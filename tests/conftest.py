"""Load the plugin against a real hermes-agent checkout, with a fake HTTP transport.

Set HERMES_AGENT_PATH to a hermes-agent checkout (CI clones one). No network, no real keys.
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
HERMES_AGENT_PATH = Path(os.environ.get("HERMES_AGENT_PATH", ROOT.parent / "hermes-agent")).resolve()
if not (HERMES_AGENT_PATH / "agent" / "memory_provider.py").is_file():
    raise RuntimeError(f"HERMES_AGENT_PATH={HERMES_AGENT_PATH} is not a hermes-agent checkout")
sys.path.insert(0, str(HERMES_AGENT_PATH))


def _load_plugin():
    spec = importlib.util.spec_from_file_location(
        "bilinc_hermes_plugin", ROOT / "__init__.py", submodule_search_locations=[str(ROOT)]
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


plugin = _load_plugin()


class FakeTransport:
    """Records every request and answers from a per-path queue (default: 200 with an empty object)."""

    def __init__(self) -> None:
        self.calls: list[dict] = []
        self.responses: dict[str, list[tuple[int, object]]] = {}

    def respond(self, path: str, status: int, payload: object) -> None:
        self.responses.setdefault(path, []).append((status, payload))

    def __call__(self, method, url, *, headers, body, timeout):
        path = url.split("://", 1)[1].split("/", 1)[1]
        path = "/" + path
        self.calls.append({"method": method, "url": url, "path": path, "headers": dict(headers),
                           "body": json.loads(body) if body else None, "timeout": timeout})
        queued = self.responses.get(path)
        if queued:
            return queued.pop(0)
        return 200, {"success": True, "entryVersion": "v1_test"}

    def paths(self) -> list[str]:
        return [call["path"] for call in self.calls]


@pytest.fixture
def transport() -> FakeTransport:
    return FakeTransport()


@pytest.fixture
def env(monkeypatch, tmp_path):
    """Isolated HERMES_HOME with a fake key; options can be written to config.yaml."""
    home = tmp_path / "hermes-home"
    home.mkdir()
    monkeypatch.setenv("HERMES_HOME", str(home))
    monkeypatch.setenv("BILINC_API_KEY", "bil_test_fake_key_for_unit_tests")
    monkeypatch.delenv("BILINC_BASE_URL", raising=False)

    def write_options(**options) -> None:
        lines = ["memory:", "  provider: bilinc", "  bilinc:"]
        lines += [f"    {name}: {json.dumps(value)}" for name, value in options.items()]
        (home / "config.yaml").write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            from hermes_cli import config as hermes_config

            for name in ("_invalidate_config_cache", "invalidate_config_cache", "clear_config_cache"):
                if hasattr(hermes_config, name):
                    getattr(hermes_config, name)()
        except Exception:
            pass

    return {"home": home, "write_options": write_options}


@pytest.fixture
def make_provider(env, transport):
    created = []

    def _make(agent_context: str = "primary", **options):
        if options:
            env["write_options"](**options)
        provider = plugin.BilincMemoryProvider(transport=transport)
        provider.initialize("session-1", hermes_home=str(env["home"]), platform="cli", agent_context=agent_context)
        created.append(provider)
        return provider

    yield _make
    for provider in created:
        provider.shutdown()
