from __future__ import annotations

import json

from conftest import plugin


def _drain(provider) -> None:
    """Wait for queued background writes and prefetch threads."""
    if provider._prefetch_thread is not None:
        provider._prefetch_thread.join(timeout=5)
    if provider._writes is not None:
        provider._writes.put(plugin._SHUTDOWN)
        provider._writer.join(timeout=5)
        provider._writes, provider._writer = None, None


def test_availability_follows_the_key(monkeypatch, env):
    provider = plugin.BilincMemoryProvider()
    assert provider.is_available()
    monkeypatch.delenv("BILINC_API_KEY")
    assert not provider.is_available()
    assert "BILINC_API_KEY" in provider.unavailable_reason()


def test_config_schema_only_asks_for_the_key():
    schema = plugin.BilincMemoryProvider().get_config_schema()
    assert [field["env_var"] for field in schema] == ["BILINC_API_KEY"]
    assert all(field.get("secret") for field in schema)


def test_tool_names_are_prefixed():
    names = [schema["name"] for schema in plugin.BilincMemoryProvider().get_tool_schemas()]
    assert names == ["bilinc_recall", "bilinc_remember", "bilinc_revise", "bilinc_forget"]


def test_remember_is_create_only_redacted_and_idempotent(make_provider, transport):
    provider = make_provider()
    raw = provider.handle_tool_call("bilinc_remember", {"content": "deploy token: abc123 and key sk-" + "x" * 24,
                                                        "key": "pref:deploy"})
    result = json.loads(raw)
    assert result["stored"] is True and result["key"] == "pref:deploy" and result["version"] == "v1_test"
    call = transport.calls[-1]
    assert call["path"] == "/api/cloud/memory/commit"
    assert call["body"]["ifAbsent"] is True
    assert "abc123" not in call["body"]["value"] and "sk-xxxx" not in call["body"]["value"]
    assert call["headers"]["Authorization"] == "Bearer bil_test_fake_key_for_unit_tests"
    assert call["headers"]["Idempotency-Key"].startswith("hermes-note-")
    assert call["headers"]["X-Bilinc-Client"] == "hermes-plugin"


def test_remember_on_existing_key_points_to_revise(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/commit", 409, {"error": "memory_exists", "message": "Memory already exists"})
    raw = provider.handle_tool_call("bilinc_remember", {"content": "x", "key": "pref:editor"})
    assert "bilinc_revise" in raw


def test_recall_tool_returns_compact_results(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/recall", 200, {"results": [
        {"key": "pref:editor", "value": "vim", "score": 0.9, "memory_type": "semantic", "internal": "drop me"}]})
    result = json.loads(provider.handle_tool_call("bilinc_recall", {"query": "which editor", "limit": 99}))
    assert result["count"] == 1
    assert result["results"][0] == {"key": "pref:editor", "value": "vim", "memory_type": "semantic"}
    assert transport.calls[-1]["body"]["limit"] == 20


def test_forget_requires_a_reason(make_provider, transport):
    provider = make_provider()
    raw = provider.handle_tool_call("bilinc_forget", {"key": "pref:editor"})
    assert "reason is required" in raw
    assert transport.calls == []


def test_errors_become_tool_errors_not_exceptions(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/recall", 402, {"error": "payment_required", "message": "Plan limit reached"})
    raw = provider.handle_tool_call("bilinc_recall", {"query": "anything"})
    assert "Plan limit reached" in raw


def test_prefetch_formats_and_reports_recall_status(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/recall", 200, {"results": [{"key": "project:x", "value": {"stack": "postgres"}}]})
    block = provider.prefetch("what database does project x use?")
    assert "project:x" in block and "postgres" in block
    assert provider.recall_status().count == 1
    assert transport.calls[-1]["body"]["profile"] == "balanced"


def test_prefetch_fails_soft_and_skips_trivial_prompts(make_provider, transport):
    provider = make_provider()
    assert provider.prefetch("thanks!") == ""
    assert transport.calls == []
    transport.respond("/api/cloud/memory/recall", 401, {"error": "invalid_api_key"})
    assert provider.prefetch("what did we decide about pricing?") == ""
    assert provider.recall_status() is None


def test_queued_prefetch_is_consumed_next_turn(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/recall", 200, {"results": [{"key": "k", "value": "v"}]})
    provider.queue_prefetch("tell me about the launch plan", session_id="session-1")
    provider._prefetch_thread.join(timeout=5)
    assert "k: v" in provider.prefetch("next question about the launch", session_id="session-1")
    assert transport.paths().count("/api/cloud/memory/recall") == 1


def test_turn_capture_is_off_by_default(make_provider, transport):
    provider = make_provider()
    provider.sync_turn("hello there, remember my plan", "noted")
    _drain(provider)
    assert transport.calls == []


def test_turn_capture_when_enabled(make_provider, transport):
    provider = make_provider(capture_turns=True, capture_ttl_days=7)
    provider.sync_turn("my api_key=supersecret and the plan", "done", session_id="session-1")
    _drain(provider)
    body = transport.calls[-1]["body"]
    assert body["key"].startswith("hermes_session::session-1::")
    assert body["canonical"] is False and body["ttl"] == 7 * 86400
    assert "supersecret" not in json.dumps(body)


def test_subagents_never_write_automatically(make_provider, transport):
    provider = make_provider(agent_context="subagent", capture_turns=True)
    provider.sync_turn("a turn", "a reply")
    provider.on_memory_write("add", "memory", "user prefers dark mode")
    _drain(provider)
    assert transport.calls == []


def test_builtin_replace_commits_new_and_forgets_old(make_provider, transport):
    provider = make_provider()
    provider.on_memory_write("replace", "user", "prefers light mode",
                             metadata={"previous_content": "prefers dark mode"})
    _drain(provider)
    assert transport.paths() == ["/api/cloud/memory/commit", "/api/cloud/memory/forget"]
    assert transport.calls[0]["body"]["key"].startswith("hermes_memory::user::")
    assert transport.calls[1]["body"]["key"] != transport.calls[0]["body"]["key"]


def test_builtin_remove_ignores_missing_entries(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/forget", 404, {"error": "memory_not_found"})
    provider.on_memory_write("remove", "memory", "", metadata={"previous_content": "old fact"})
    _drain(provider)
    assert transport.paths() == ["/api/cloud/memory/forget"]


def test_mirroring_can_be_disabled(make_provider, transport):
    provider = make_provider(mirror_builtin=False)
    provider.on_memory_write("add", "memory", "a fact")
    _drain(provider)
    assert transport.calls == []


def test_plain_http_base_url_is_refused(monkeypatch, make_provider, transport):
    monkeypatch.setenv("BILINC_BASE_URL", "http://evil.example.com")
    provider = make_provider()
    provider.handle_tool_call("bilinc_recall", {"query": "x"})
    assert transport.calls[-1]["url"].startswith("https://bilinc.space/")


def test_localhost_base_url_is_allowed(monkeypatch, make_provider, transport):
    monkeypatch.setenv("BILINC_BASE_URL", "http://localhost:3000/")
    provider = make_provider()
    provider.handle_tool_call("bilinc_recall", {"query": "x"})
    assert transport.calls[-1]["url"] == "http://localhost:3000/api/cloud/memory/recall"


def test_repeated_writes_get_distinct_idempotency_keys(make_provider, transport):
    provider = make_provider()
    provider.handle_tool_call("bilinc_remember", {"content": "same fact", "key": "fact:a"})
    provider.handle_tool_call("bilinc_forget", {"key": "fact:a", "reason": "test"})
    provider.handle_tool_call("bilinc_remember", {"content": "same fact", "key": "fact:a"})
    keys = [call["headers"]["Idempotency-Key"] for call in transport.calls]
    assert len(keys) == 3 and len(set(keys)) == 3


def test_outage_costs_one_short_attempt_then_backs_off(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/recall", 503, {"error": "cloud_runtime_unavailable", "retryable": True})
    assert provider.prefetch("what is the deploy plan?") == ""
    assert len(transport.calls) == 1
    assert transport.calls[0]["timeout"] <= 6.0  # under Hermes' 8 s provider budget
    assert provider.prefetch("and the rollback plan?") == ""
    provider.queue_prefetch("and the rollback plan?")
    assert len(transport.calls) == 1


def test_plan_limit_stops_automatic_calls(make_provider, transport):
    provider = make_provider(capture_turns=True)
    transport.respond("/api/cloud/memory/recall", 402, {"error": "payment_required"})
    assert provider.prefetch("what did we decide?") == ""
    provider.sync_turn("a turn", "a reply")
    provider.on_memory_write("add", "memory", "a fact")
    provider.queue_prefetch("next question here")
    _drain(provider)
    assert len(transport.calls) == 1
    # explicit tools still reach the server, so the agent can report the error
    provider.handle_tool_call("bilinc_recall", {"query": "x"})
    assert len(transport.calls) == 2


def test_register_hands_out_a_fresh_instance():
    registered = []

    class Ctx:
        def register_memory_provider(self, provider):
            registered.append(provider)

    plugin.register(Ctx())
    plugin.register(Ctx())
    assert len(registered) == 2 and registered[0] is not registered[1]


def test_recall_tool_accepts_several_queries_and_dedupes(make_provider, transport):
    provider = make_provider()
    both = {"results": [{"key": "person:anna", "value": "Anna Rao works at Fieldline"},
                        {"key": "pref:editor", "value": "vim"}]}
    transport.respond("/api/cloud/memory/recall", 200, both)
    transport.respond("/api/cloud/memory/recall", 200, both)  # queries run in parallel; same answer for each
    result = json.loads(provider.handle_tool_call("bilinc_recall", {"queries": ["Anna Rao employer", "editor"]}))
    assert [g["query"] for g in result["searches"]] == ["Anna Rao employer", "editor"]
    assert [r["key"] for r in result["searches"][0]["results"]] == ["person:anna", "pref:editor"]
    assert result["searches"][1]["results"] == []  # same keys are not repeated
    assert result["count"] == 2
    assert len(transport.calls) == 2


def test_recall_tool_requires_a_query(make_provider, transport):
    provider = make_provider()
    assert "query is required" in provider.handle_tool_call("bilinc_recall", {"queries": []})
    assert transport.calls == []


def test_long_values_are_cut_around_the_matching_part(make_provider, transport):
    provider = make_provider()
    filler = "unrelated planning chatter. " * 200
    turn = {"user": filler + "The Lisbon offsite budget is 14,200 EUR. " + filler, "assistant": "Noted."}
    transport.respond("/api/cloud/memory/recall", 200, {"results": [{"key": "hermes_session::s1::abc", "value": turn}]})
    result = json.loads(provider.handle_tool_call("bilinc_recall", {"query": "Lisbon offsite budget"}))
    item = result["results"][0]
    assert item["truncated"] is True and len(item["value"]) <= 1502
    assert "14,200 EUR" in item["value"]
    transport.respond("/api/cloud/memory/recall", 200, {"results": [{"key": "hermes_session::s1::abc", "value": turn}]})
    block = provider.prefetch("what is the Lisbon offsite budget?")
    assert "14,200 EUR" in block
    assert len(block) < 1000


def test_captured_turns_read_as_dialogue(make_provider, transport):
    provider = make_provider()
    transport.respond("/api/cloud/memory/recall", 200, {"results": [
        {"key": "hermes_session::s1::abc", "value": {"user": "I moved to Porto", "assistant": "Got it."}}]})
    result = json.loads(provider.handle_tool_call("bilinc_recall", {"query": "where do I live"}))
    assert result["results"][0]["value"] == "User: I moved to Porto\nAssistant: Got it."


def test_system_prompt_does_not_promise_recall_that_may_not_happen(make_provider):
    block = make_provider().system_prompt_block()
    assert "recalled before each turn" not in block
    assert "bilinc_recall before acting" in block



def test_clip_keeps_marker_inside_limit():
    out = plugin._clip("x" * 5000, 4096)
    assert len(out) == 4096 and out.endswith("…[truncated]")
    assert plugin._clip("short", 4096) == "short"
    assert len(plugin._clip("y" * 600, 512)) == 512
