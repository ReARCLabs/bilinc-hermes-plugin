"""Bilinc memory provider for Hermes.

Hosted, versioned long-term memory from Bilinc Cloud (https://bilinc.space): relevant
memories are recalled before each turn, the agent gets explicit recall / remember /
revise / forget tools, Hermes' built-in memory writes are mirrored, and finished turns
can optionally be captured. Every call goes over HTTPS to Bilinc Cloud under
``BILINC_API_KEY``; nothing runs or is stored on the local machine.

Options live in config.yaml under ``memory.bilinc`` (see README). The API key is read
only through Hermes' scoped secrets.
"""

from __future__ import annotations

import hashlib
import json
import logging
import queue
import re
import ssl
import threading
import time
import urllib.error
import urllib.request
import uuid
from typing import Any, Callable, Optional
from urllib.parse import urlsplit

from agent.memory_provider import MemoryProvider, RecallStatus, is_trivial_prompt, spawn_context_thread
from agent.secret_scope import get_secret
from tools.registry import tool_error

logger = logging.getLogger(__name__)

PLUGIN_VERSION = "1.0.0"
_DEFAULT_BASE_URL = "https://bilinc.space"
_KEY_URL = "https://bilinc.space/signup?utm_source=hermes&utm_medium=plugin"
_LOCAL_HOSTS = {"localhost", "127.0.0.1", "::1"}

_DEFAULT_OPTIONS: dict[str, Any] = {
    "base_url": "",
    "auto_recall": True,
    "recall_limit": 6,
    "capture_turns": False,
    "capture_ttl_days": 30,
    "mirror_builtin": True,
}
_BOOL_WORDS = {**dict.fromkeys(("true", "1", "yes", "on"), True), **dict.fromkeys(("false", "0", "no", "off"), False)}

_MAX_VALUE_CHARS = 6000
_MAX_PREFETCH_LINE = 320
_MAX_KEY_CHARS = 512
_READ_ATTEMPTS = 2
_PREFETCH_TIMEOUT = 2.0
_WRITE_TIMEOUT = 10.0
_RECALL_BACKOFF_SECONDS = 60.0
# Errors that will not go away by retrying: a bad key or an exhausted / inactive plan.
_BLOCKING_CODES = frozenset({"invalid_api_key", "missing_api_key", "payment_required", "entitlement_inactive"})
_BLOCKING_STATUSES = frozenset({401, 402})
_SHUTDOWN = object()

# Secret-shaped substrings are replaced before anything is sent upstream.
_SECRETISH = re.compile(
    r"(?i)(bearer\s+[a-z0-9._-]{12,}|sk-[a-z0-9_-]{20,}|pypi-[a-z0-9_-]{20,}|bil_live_[a-z0-9_-]{12,}"
    r"|gh[pousr]_[a-z0-9]{20,}|xox[abpr]-[a-z0-9-]{10,}|akia[0-9a-z]{16}"
    r"|api[_-]?key\s*[:=]\s*\S+|token\s*[:=]\s*\S+|password\s*[:=]\s*\S+)"
)


def _redact(text: Any) -> str:
    return _SECRETISH.sub("[REDACTED]", "" if text is None else str(text))


def _clip(text: Any, limit: int = _MAX_VALUE_CHARS) -> str:
    value = _redact(text)
    return value if len(value) <= limit else value[:limit] + "…[truncated]"


def _digest(*parts: Any, length: int = 16) -> str:
    raw = "\x1f".join("" if p is None else str(p) for p in parts)
    return hashlib.sha256(raw.encode("utf-8", errors="replace")).hexdigest()[:length]


def _request_id(prefix: str) -> str:
    """A fresh Idempotency-Key. Bilinc replays the stored response for a repeated key, so a key derived
    from content would turn remember -> forget -> remember of the same fact into a silent no-op."""
    return f"{prefix}-{uuid.uuid4().hex}"


def _safe_part(value: Any, limit: int = 64) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(value or "")).strip("_")
    return (cleaned or "default")[:limit]


def _as_bool(value: Any, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _BOOL_WORDS.get(value.strip().lower(), default)
    return default


def _as_int(value: Any, default: int, lo: int, hi: int) -> int:
    try:
        return max(lo, min(hi, int(value)))
    except (TypeError, ValueError):
        return default


def _load_options() -> dict[str, Any]:
    """``memory.bilinc`` from config.yaml merged over the defaults (defaults on any error)."""
    try:
        from hermes_cli.config import load_config_readonly

        block = (load_config_readonly().get("memory") or {}).get("bilinc") or {}
    except Exception:
        block = {}
    block = block if isinstance(block, dict) else {}
    return {
        "base_url": str(block.get("base_url") or "").strip(),
        "auto_recall": _as_bool(block.get("auto_recall"), _DEFAULT_OPTIONS["auto_recall"]),
        "recall_limit": _as_int(block.get("recall_limit"), _DEFAULT_OPTIONS["recall_limit"], 1, 20),
        "capture_turns": _as_bool(block.get("capture_turns"), _DEFAULT_OPTIONS["capture_turns"]),
        "capture_ttl_days": _as_int(block.get("capture_ttl_days"), _DEFAULT_OPTIONS["capture_ttl_days"], 1, 3650),
        "mirror_builtin": _as_bool(block.get("mirror_builtin"), _DEFAULT_OPTIONS["mirror_builtin"]),
    }


def _resolve_base_url(configured: str) -> str:
    """BILINC_BASE_URL (scoped) > config.yaml > default. Plain http is accepted only for localhost."""
    raw = (get_secret("BILINC_BASE_URL", "") or "").strip() or configured or _DEFAULT_BASE_URL
    base = raw.rstrip("/")
    parts = urlsplit(base)
    if parts.scheme == "https" and parts.hostname:
        return base
    if parts.scheme == "http" and (parts.hostname or "") in _LOCAL_HOSTS:
        return base
    logger.warning("Bilinc: ignoring base_url %r (https required); using %s", raw, _DEFAULT_BASE_URL)
    return _DEFAULT_BASE_URL


# ---------------------------------------------------------------------------
# HTTP client (stdlib only)
# ---------------------------------------------------------------------------


class BilincError(RuntimeError):
    """A failed Bilinc Cloud call; ``code`` is the canonical API error code when the server sent one."""

    def __init__(self, message: str, *, status: Optional[int] = None, code: Optional[str] = None,
                 retryable: bool = False) -> None:
        super().__init__(message)
        self.status, self.code, self.retryable = status, code, retryable


Transport = Callable[..., tuple[int, Any]]


def _ssl_context() -> ssl.SSLContext:
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _urllib_transport(method: str, url: str, *, headers: dict[str, str], body: Optional[bytes],
                      timeout: float) -> tuple[int, Any]:
    request = urllib.request.Request(url, data=body, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout, context=_ssl_context()) as response:  # noqa: S310 - https enforced
            status, raw = response.status, response.read().decode("utf-8", errors="replace")
    except urllib.error.HTTPError as exc:
        status, raw = exc.code, exc.read().decode("utf-8", errors="replace")
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise BilincError(f"Bilinc Cloud unreachable: {getattr(exc, 'reason', exc)}",
                          code="connection_failed", retryable=True) from exc
    try:
        payload = json.loads(raw) if raw else {}
    except json.JSONDecodeError:
        payload = {"message": raw[:200]}
    return status, payload


class _Client:
    """Minimal Bilinc Cloud client: one profile-bound key and base URL per instance."""

    def __init__(self, api_key: str, base_url: str, transport: Optional[Transport] = None) -> None:
        self._api_key = api_key.replace("Bearer ", "").strip()
        self.base_url = base_url.rstrip("/")
        self._transport = transport or _urllib_transport

    def _headers(self, idempotency_key: Optional[str]) -> dict[str, str]:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "User-Agent": f"bilinc-hermes/{PLUGIN_VERSION}",
            "X-Bilinc-Client": "hermes-plugin",
            "X-Bilinc-Client-Version": PLUGIN_VERSION,
        }
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return headers

    def _call(self, method: str, path: str, payload: Optional[dict[str, Any]], *, timeout: float,
              idempotency_key: Optional[str] = None) -> dict[str, Any]:
        body = None if payload is None else json.dumps(payload, separators=(",", ":"), default=str).encode("utf-8")
        status, data = self._transport(method, f"{self.base_url}{path}", headers=self._headers(idempotency_key),
                                       body=body, timeout=timeout)
        if 200 <= status < 300:
            return data if isinstance(data, dict) else {"result": data}
        info = data if isinstance(data, dict) else {}
        code = info.get("error") if isinstance(info.get("error"), str) else None
        message = info.get("message") if isinstance(info.get("message"), str) else None
        retryable = info.get("retryable")
        if not isinstance(retryable, bool):
            retryable = status == 429 or status >= 500
        raise BilincError(f"Bilinc Cloud {path} failed ({status}): {message or code or 'error'}",
                          status=status, code=code, retryable=retryable)

    def _read(self, path: str, payload: dict[str, Any], *, timeout: float, attempts: int) -> dict[str, Any]:
        """Reads may retry on retryable failures; writes are never retried here."""
        for attempt in range(attempts):
            try:
                return self._call("POST", path, payload, timeout=timeout)
            except BilincError as exc:
                if not exc.retryable or attempt == attempts - 1:
                    raise
                time.sleep(0.25 * (attempt + 1))
        raise AssertionError("unreachable")  # pragma: no cover

    def recall(self, query: str, *, limit: int, profile: str = "balanced", timeout: float = _WRITE_TIMEOUT,
               attempts: int = _READ_ATTEMPTS) -> dict[str, Any]:
        return self._read("/api/cloud/memory/recall",
                          {"query": _clip(query, 4096), "profile": profile, "limit": limit},
                          timeout=timeout, attempts=attempts)

    def commit(self, key: str, value: Any, *, memory_type: str = "semantic", importance: float = 0.7,
               metadata: Optional[dict[str, Any]] = None, source: str = "hermes", session_id: str = "",
               canonical: Optional[bool] = None, priority: Optional[float] = None, ttl: Optional[float] = None,
               if_absent: bool = False, idempotency_key: Optional[str] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"key": key, "value": value, "memoryType": memory_type, "importance": importance,
                                   "metadata": metadata or {}, "source": source}
        if session_id:
            payload["sessionId"] = session_id[:256]
        for name, val in (("canonical", canonical), ("priority", priority), ("ttl", ttl)):
            if val is not None:
                payload[name] = val
        if if_absent:
            payload["ifAbsent"] = True
        return self._call("POST", "/api/cloud/memory/commit", payload, timeout=_WRITE_TIMEOUT,
                          idempotency_key=idempotency_key)

    def revise(self, key: str, value: Any, *, reason: str = "", idempotency_key: Optional[str] = None) -> dict[str, Any]:
        payload: dict[str, Any] = {"key": key, "value": value, "strategy": "recency"}
        if reason:
            payload["reason"] = reason[:512]
        return self._call("POST", "/api/cloud/memory/revise", payload, timeout=_WRITE_TIMEOUT,
                          idempotency_key=idempotency_key)

    def forget(self, key: str, *, reason: str, idempotency_key: Optional[str] = None) -> dict[str, Any]:
        return self._call("POST", "/api/cloud/memory/forget", {"key": key, "reason": reason[:512]},
                          timeout=_WRITE_TIMEOUT, idempotency_key=idempotency_key)


# ---------------------------------------------------------------------------
# Formatting helpers
# ---------------------------------------------------------------------------


def _result_items(payload: dict[str, Any]) -> list[dict[str, Any]]:
    items = payload.get("results") if isinstance(payload, dict) else None
    return [item for item in (items or []) if isinstance(item, dict)]


def _value_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _compact(text: str, limit: int = _MAX_PREFETCH_LINE) -> str:
    flat = re.sub(r"\s+", " ", text or "").strip()
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


def _format_prefetch(items: list[dict[str, Any]]) -> str:
    lines = []
    for item in items:
        value = _compact(_value_text(item.get("value", item.get("content", ""))))
        if value:
            lines.append(f"- {item.get('key', 'memory')}: {value}")
    if not lines:
        return ""
    return "[Bilinc memory — recalled for this turn; may be outdated, verify before relying on it]\n" + "\n".join(lines)


def _tool_result(item: dict[str, Any]) -> dict[str, Any]:
    keep = ("key", "value", "memory_type", "importance", "score", "version", "entry_version", "source", "updated_at")
    return {k: item[k] for k in keep if k in item}


def _schema(name: str, description: str, properties: dict[str, Any], required: tuple[str, ...]) -> dict[str, Any]:
    return {"name": name, "description": description,
            "parameters": {"type": "object", "properties": properties, "required": list(required)}}


_TOOL_SCHEMAS = (
    _schema("bilinc_recall",
            "Search long-term memory in Bilinc. Returns matching memories with their keys, so they can be revised or forgotten.",
            {"query": {"type": "string", "description": "What to look for."},
             "limit": {"type": "integer", "description": "Maximum results, 1-20 (default 8)."}},
            ("query",)),
    _schema("bilinc_remember",
            "Store a new durable fact, decision or preference in Bilinc. Never overwrites: if the key already exists, "
            "use bilinc_revise instead.",
            {"content": {"type": "string", "description": "The fact to remember, self-contained."},
             "key": {"type": "string", "description": "Optional stable key such as 'pref:editor'. Generated if omitted."},
             "memory_type": {"type": "string", "enum": ["semantic", "episodic", "procedural"],
                             "description": "semantic (facts, default), episodic (events), procedural (how-to)."}},
            ("content",)),
    _schema("bilinc_revise",
            "Replace the value of an existing Bilinc memory. Bilinc keeps the previous version in its history.",
            {"key": {"type": "string", "description": "Key of the memory to change (from bilinc_recall)."},
             "content": {"type": "string", "description": "The new value."},
             "reason": {"type": "string", "description": "Why it changed (kept in the audit history)."}},
            ("key", "content")),
    _schema("bilinc_forget",
            "Remove a Bilinc memory from recall. The reason is written to the audit history.",
            {"key": {"type": "string", "description": "Key of the memory to forget (from bilinc_recall)."},
             "reason": {"type": "string", "description": "Why it should be forgotten."}},
            ("key", "reason")),
)


# ---------------------------------------------------------------------------
# Provider
# ---------------------------------------------------------------------------


class BilincMemoryProvider(MemoryProvider):
    """Bilinc Cloud memory. Every binding (key, URL, options) is made in initialize() for one profile."""

    def __init__(self, transport: Optional[Transport] = None) -> None:
        self._transport = transport
        self._client: Optional[_Client] = None
        self._options: dict[str, Any] = dict(_DEFAULT_OPTIONS)
        self._session_id = ""
        self._write_enabled = False
        self._lock = threading.Lock()
        self._prefetched: dict[str, tuple[str, int]] = {}
        self._last_count: Optional[int] = None
        self._prefetch_thread: Optional[threading.Thread] = None
        self._writes: Optional[queue.Queue] = None
        self._writer: Optional[threading.Thread] = None
        self._writer_lock = threading.Lock()
        self._warned: set[Any] = set()
        # Set after a bad key or an exhausted plan: automatic recall and writes stop for this instance
        # (explicit tool calls still go through, so the agent can report the error).
        self._blocked = False
        self._recall_paused_until = 0.0

    @property
    def name(self) -> str:
        return "bilinc"

    # -- setup ------------------------------------------------------------------

    def is_available(self) -> bool:
        return bool(get_secret("BILINC_API_KEY", ""))

    def unavailable_reason(self) -> str:
        return f"Set BILINC_API_KEY (run `hermes memory setup bilinc`; get a key at {_KEY_URL})."

    def get_config_schema(self) -> list[dict[str, Any]]:
        return [{"key": "api_key", "description": "Bilinc Cloud API key", "secret": True, "required": True,
                 "env_var": "BILINC_API_KEY", "url": _KEY_URL}]

    def initialize(self, session_id: str, **kwargs) -> None:
        self._options = _load_options()
        api_key = get_secret("BILINC_API_KEY", "") or ""
        base_url = _resolve_base_url(self._options["base_url"])
        self._client = _Client(api_key, base_url, self._transport) if api_key else None
        self._session_id = session_id or ""
        # Automatic writes only from the primary agent; subagents, cron and flush runs only read.
        self._write_enabled = (kwargs.get("agent_context") or "primary") == "primary"

    def system_prompt_block(self) -> str:
        if not self._client:
            return ""
        return ("# Bilinc memory\nActive. Relevant memories are recalled before each turn. "
                "Use bilinc_recall to search, bilinc_remember for new durable facts, bilinc_revise to change one "
                "and bilinc_forget to remove one. Store only what will matter in later sessions, and never secrets.")

    # -- recall -----------------------------------------------------------------

    def _recall_block(self, query: str, timeout: float, attempts: int) -> tuple[str, int]:
        payload = self._client.recall(query, limit=self._options["recall_limit"], profile="fast", timeout=timeout,
                                      attempts=attempts)
        items = _result_items(payload)
        return _format_prefetch(items), len(items)

    def _auto_recall_ready(self, query: str) -> bool:
        return bool(self._client and self._options["auto_recall"] and not self._blocked
                    and time.monotonic() >= self._recall_paused_until and not is_trivial_prompt(query))

    def _recall_failed(self, exc: Exception) -> None:
        # Back off so an outage costs one short wait per minute, not one per turn.
        self._recall_paused_until = time.monotonic() + _RECALL_BACKOFF_SECONDS
        self._note_failure("prefetch", exc)

    def queue_prefetch(self, query: str, *, session_id: str = "") -> None:
        if not self._auto_recall_ready(query):
            return
        if self._prefetch_thread is not None and self._prefetch_thread.is_alive():
            return
        sid = session_id or self._session_id

        def _job() -> None:
            try:
                result = self._recall_block(query, _WRITE_TIMEOUT, _READ_ATTEMPTS)
            except Exception as exc:
                self._recall_failed(exc)
                return
            with self._lock:
                self._prefetched[sid] = result

        self._prefetch_thread = spawn_context_thread(_job, name="bilinc-prefetch")
        self._prefetch_thread.start()

    def prefetch(self, query: str, *, session_id: str = "") -> str:
        self._last_count = None
        if not self._auto_recall_ready(query):
            return ""
        sid = session_id or self._session_id
        with self._lock:
            cached = self._prefetched.pop(sid, None)
        if cached is None:  # nothing queued yet (first turn): one short synchronous attempt
            try:
                cached = self._recall_block(query, _PREFETCH_TIMEOUT, 1)
            except Exception as exc:
                self._recall_failed(exc)
                return ""
        text, count = cached
        self._last_count = count if text else None
        return text

    def recall_status(self) -> Optional[RecallStatus]:
        if not self._last_count:
            return None
        return RecallStatus(provider_label="Bilinc", count=self._last_count)

    # -- background writes ---------------------------------------------------------

    def _enqueue(self, label: str, fn: Callable[[], Any]) -> None:
        with self._writer_lock:
            if self._writes is None:
                self._writes = queue.Queue()
                self._writer = spawn_context_thread(self._write_loop, args=(self._writes,), name="bilinc-writer")
                self._writer.start()
            self._writes.put((label, fn))

    def _write_loop(self, writes: queue.Queue) -> None:
        while (item := writes.get()) is not _SHUTDOWN:
            label, fn = item
            try:
                fn()
            except Exception as exc:
                self._note_failure(label, exc)

    def _note_failure(self, label: str, exc: Exception) -> None:
        code, status = getattr(exc, "code", None), getattr(exc, "status", None)
        if code in _BLOCKING_CODES or status in _BLOCKING_STATUSES:
            self._blocked = True
            if (code, status) not in self._warned:  # once per instance: a bad key or exhausted plan must not spam logs
                self._warned.add((code, status))
                logger.warning("Bilinc %s failed; automatic recall and writes are off until restart: %s", label, exc)
            return
        logger.debug("Bilinc %s failed: %s", label, exc)

    def _auto_write_ready(self, option: str) -> bool:
        return bool(self._client and self._write_enabled and self._options[option] and not self._blocked)

    def sync_turn(self, user_content: str, assistant_content: str, *, session_id: str = "", **kwargs) -> None:
        if not self._auto_write_ready("capture_turns") or not user_content:
            return
        sid = session_id or self._session_id
        key = f"hermes_session::{_safe_part(sid)}::{_digest(sid, user_content, assistant_content)}"
        value = {"user": _clip(user_content, 3000), "assistant": _clip(assistant_content, 3000)}
        ttl = float(self._options["capture_ttl_days"] * 86400)
        self._enqueue("turn capture", lambda: self._client.commit(
            key, value, memory_type="episodic", importance=0.3, source="hermes_session_summary",
            session_id=sid, canonical=False, priority=0.2, ttl=ttl,
            metadata={"kind": "turn", "plugin": "bilinc-hermes"}, idempotency_key=f"hermes-turn-{_digest(key, 'v1', length=32)}"))

    def on_memory_write(self, action: str, target: str, content: str, metadata: Optional[dict[str, Any]] = None,
                        **kwargs) -> None:
        """Mirror Hermes' built-in memory tool into Bilinc (add -> commit, replace -> commit + forget old, remove -> forget)."""
        if not self._auto_write_ready("mirror_builtin"):
            return
        target = _safe_part(target or "memory", 16)
        previous = (metadata or {}).get("previous_content") or ""

        def _key(text: str) -> str:
            return f"hermes_memory::{target}::{_digest(text)}"

        def _commit(text: str) -> None:
            self._client.commit(_key(text), _clip(text), memory_type="semantic", importance=0.8, canonical=True,
                                priority=0.6, session_id=self._session_id,
                                metadata={"kind": "builtin_memory", "target": target, "plugin": "bilinc-hermes"},
                                idempotency_key=_request_id("hermes-mem"))

        def _forget(text: str, reason: str) -> None:
            try:
                self._client.forget(_key(text), reason=reason,
                                    idempotency_key=_request_id("hermes-mem"))
            except BilincError as exc:
                if exc.status != 404:
                    raise

        if action == "add" and content:
            self._enqueue("memory mirror", lambda: _commit(content))
        elif action == "replace" and content:
            def _replace() -> None:
                _commit(content)
                if previous and previous != content:
                    _forget(previous, "Replaced in Hermes built-in memory")
            self._enqueue("memory mirror", _replace)
        elif action == "remove" and (previous or content):
            self._enqueue("memory mirror", lambda: _forget(previous or content, "Removed from Hermes built-in memory"))

    def on_session_switch(self, new_session_id: str, *, parent_session_id: str = "", reset: bool = False,
                          rewound: bool = False, **kwargs) -> None:
        with self._lock:
            self._prefetched.pop(self._session_id, None)
        self._session_id = new_session_id or self._session_id

    # -- tools -------------------------------------------------------------------------

    def get_tool_schemas(self) -> list[dict[str, Any]]:
        return [dict(schema) for schema in _TOOL_SCHEMAS]

    def handle_tool_call(self, tool_name: str, args: dict[str, Any], **kwargs) -> str:
        if not self._client:
            return tool_error("Bilinc is not configured: set BILINC_API_KEY.")
        handler = self._TOOLS.get(tool_name)
        if handler is None:
            return tool_error(f"Unknown Bilinc tool: {tool_name}")
        try:
            return json.dumps(handler(self, args or {}), ensure_ascii=False, default=str)
        except BilincError as exc:
            self._note_failure(tool_name, exc)
            hint = " Use bilinc_revise to change an existing key." if exc.code == "memory_exists" else ""
            return tool_error(f"{exc}{hint}", code=exc.code or exc.status)
        except ValueError as exc:
            return tool_error(str(exc))

    @staticmethod
    def _required(args: dict[str, Any], name: str) -> str:
        value = str(args.get(name) or "").strip()
        if not value:
            raise ValueError(f"{name} is required")
        return value

    def _tool_recall(self, args: dict[str, Any]) -> dict[str, Any]:
        query = self._required(args, "query")
        limit = _as_int(args.get("limit"), 8, 1, 20)
        items = _result_items(self._client.recall(query, limit=limit))
        return {"count": len(items), "results": [_tool_result(item) for item in items]}

    def _tool_remember(self, args: dict[str, Any]) -> dict[str, Any]:
        content = _clip(self._required(args, "content"))
        key = str(args.get("key") or "").strip()[:_MAX_KEY_CHARS] or f"hermes_note::{_digest(content)}"
        memory_type = args.get("memory_type") if args.get("memory_type") in ("semantic", "episodic", "procedural") else "semantic"
        result = self._client.commit(key, content, memory_type=memory_type, canonical=True, priority=0.6,
                                     session_id=self._session_id, metadata={"kind": "note", "plugin": "bilinc-hermes"},
                                     if_absent=True, idempotency_key=_request_id("hermes-note"))
        return {"stored": bool(result.get("success", True)), "key": key, "version": result.get("entry_version")}

    def _tool_revise(self, args: dict[str, Any]) -> dict[str, Any]:
        key = self._required(args, "key")
        content = _clip(self._required(args, "content"))
        result = self._client.revise(key, content, reason=_clip(args.get("reason") or "", 512),
                                     idempotency_key=_request_id("hermes-revise"))
        return {"revised": bool(result.get("success", True)), "key": key, "version": result.get("entry_version")}

    def _tool_forget(self, args: dict[str, Any]) -> dict[str, Any]:
        key = self._required(args, "key")
        reason = _clip(self._required(args, "reason"), 512)
        self._client.forget(key, reason=reason, idempotency_key=_request_id("hermes-forget"))
        return {"forgotten": True, "key": key}

    _TOOLS: dict[str, Callable[["BilincMemoryProvider", dict[str, Any]], dict[str, Any]]] = {
        "bilinc_recall": _tool_recall,
        "bilinc_remember": _tool_remember,
        "bilinc_revise": _tool_revise,
        "bilinc_forget": _tool_forget,
    }

    # -- teardown ---------------------------------------------------------------------

    def shutdown(self) -> None:
        if self._prefetch_thread is not None:
            self._prefetch_thread.join(timeout=2.0)
        if self._writes is not None and self._writer is not None:
            self._writes.put(_SHUTDOWN)
            self._writer.join(timeout=10.0)
        self._writes, self._writer, self._prefetch_thread, self._client = None, None, None, None


def register(ctx) -> None:
    """Register Bilinc as a Hermes memory provider (a new instance per load)."""
    ctx.register_memory_provider(BilincMemoryProvider())
