"""LLM provider tests. Groq is mocked at the HTTP layer; no real API call is made."""

import asyncio
import json
from types import SimpleNamespace as NS

import httpx
import pytest
from mcp.client import Client

from research_lab import agent
from research_lab.agent import Controller, Limits
from research_lab.flowlab import Flowlab
from research_lab.llm import (GROQ_BASE_URL, GROQ_MODEL, Anthropic, LLMError, LLMNotConfigured, OpenAICompatible,
                              ToolCall, ToolResult, Turn, llm_from_env)
from research_lab.mcp_server import build_server
from research_lab.store import Store
from tests.test_agent import SETUP
from tests.test_mcp_server import FakeFlowlab

TOOLS = [{"name": "get_campaign", "description": "state",
          "input_schema": {"type": "object", "properties": {"campaign_id": {"type": "string"}},
                           "required": ["campaign_id"]}}]


def completion(content=None, tool_calls=None, finish="stop"):
    msg = {"role": "assistant", "content": content}
    if tool_calls:
        msg["tool_calls"] = tool_calls
    return {"choices": [{"index": 0, "message": msg, "finish_reason": finish}],
            "usage": {"prompt_tokens": 100, "completion_tokens": 20}}


def call(id_, name, args):
    return {"id": id_, "type": "function",
            "function": {"name": name, "arguments": args if isinstance(args, str) else json.dumps(args)}}


class FakeGroq:
    """Scripted /chat/completions; records request bodies."""

    def __init__(self, responses):
        self.responses, self.requests = list(responses), []

    def handler(self, request):
        assert request.url.path == "/openai/v1/chat/completions"
        assert request.headers["authorization"] == "Bearer gsk-test"
        self.requests.append(json.loads(request.content))
        r = self.responses.pop(0)
        return r if isinstance(r, httpx.Response) else httpx.Response(200, json=r)


def groq(fake, **kw):
    return OpenAICompatible("gsk-test", GROQ_MODEL, GROQ_BASE_URL, name="groq",
                            extra={"reasoning_effort": "medium"}, transport=httpx.MockTransport(fake.handler), **kw)


# --- selection / configuration ---------------------------------------------------------

def test_groq_is_default_and_needs_no_anthropic_key():
    p = llm_from_env({"GROQ_API_KEY": "gsk-test"})
    assert isinstance(p, OpenAICompatible) and p.name == "groq" and p.model == "openai/gpt-oss-120b"
    assert str(p.http.base_url).rstrip("/") == "https://api.groq.com/openai/v1"
    assert p.extra == {"reasoning_effort": "medium"}


def test_groq_configuration_overrides():
    p = llm_from_env({"RESEARCH_LLM_PROVIDER": "groq", "GROQ_API_KEY": "k", "GROQ_MODEL": "m",
                      "GROQ_BASE_URL": "https://example.test/v1/", "GROQ_REASONING_EFFORT": "high"})
    assert p.model == "m" and str(p.http.base_url).rstrip("/") == "https://example.test/v1"
    assert p.extra["reasoning_effort"] == "high"
    assert llm_from_env({"GROQ_API_KEY": "k"}, model="cli-model").model == "cli-model"


def test_missing_groq_key():
    with pytest.raises(LLMNotConfigured, match="GROQ_API_KEY"):
        llm_from_env({"RESEARCH_LLM_PROVIDER": "groq", "ANTHROPIC_API_KEY": "present-but-irrelevant"})


def test_unknown_provider():
    with pytest.raises(LLMNotConfigured):
        llm_from_env({"RESEARCH_LLM_PROVIDER": "gemini"})


def test_anthropic_provider_still_loads(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")  # constructing the client makes no request
    p = llm_from_env({"RESEARCH_LLM_PROVIDER": "anthropic"})
    assert isinstance(p, Anthropic) and p.model == "claude-opus-5-5"


# --- request shape ---------------------------------------------------------------------

def test_request_uses_only_function_tools():
    fake = FakeGroq([completion("hi")])
    groq(fake).step("SYS", TOOLS, [{"role": "user", "text": "q"}])
    body = fake.requests[0]
    assert body["model"] == "openai/gpt-oss-120b" and body["tool_choice"] == "auto"
    assert body["reasoning_effort"] == "medium"
    assert body["messages"] == [{"role": "system", "content": "SYS"}, {"role": "user", "content": "q"}]
    assert body["tools"] == [{"type": "function", "function": {"name": "get_campaign", "description": "state",
                                                               "parameters": TOOLS[0]["input_schema"]}}]
    assert all(t["type"] == "function" for t in body["tools"])  # no Groq browser/code tools


# --- response parsing ------------------------------------------------------------------

def test_parse_tool_calls():
    t = OpenAICompatible.parse(completion(None, [call("c1", "get_campaign", {"campaign_id": "x"}),
                                                 call("c2", "search_evidence", '{"query": "sicn"}')], "tool_calls"))
    assert t.stop == "tool_calls" and [c.name for c in t.tool_calls] == ["get_campaign", "search_evidence"]
    assert t.tool_calls[0].arguments == {"campaign_id": "x"} and t.tool_calls[1].arguments == {"query": "sicn"}


@pytest.mark.parametrize("args,err", [('{"query": ', "invalid JSON"), ("[1, 2]", "not a JSON object")])
def test_parse_bad_arguments_are_flagged(args, err):
    t = OpenAICompatible.parse(completion(None, [call("c1", "search_evidence", args)], "tool_calls"))
    assert t.tool_calls[0].arguments is None and err in t.tool_calls[0].parse_error


@pytest.mark.parametrize("finish,stop", [("stop", "end"), ("length", "max_tokens"),
                                         ("content_filter", "refusal"), ("weird", "other")])
def test_parse_finish_reasons(finish, stop):
    t = OpenAICompatible.parse(completion("text", None, finish))
    assert t.stop == stop and t.text == "text" and t.tool_calls == []


def test_parse_malformed_body():
    with pytest.raises(LLMError, match="malformed"):
        OpenAICompatible.parse({"error": "x"})


def test_tool_results_returned_to_model():
    fake = FakeGroq([completion("done")])
    turn = Turn("", [ToolCall("c1", "get_campaign", {"campaign_id": "x"})], "tool_calls")
    history = [{"role": "user", "text": "q"}, {"role": "assistant", "turn": turn},
               {"role": "tool_results", "results": [ToolResult("c1", '{"ok": 1}'), ]}]
    groq(fake).step("SYS", TOOLS, history)
    msgs = fake.requests[0]["messages"]
    assert msgs[2] == {"role": "assistant", "content": None, "tool_calls": [
        {"id": "c1", "type": "function", "function": {"name": "get_campaign", "arguments": '{"campaign_id": "x"}'}}]}
    assert msgs[3] == {"role": "tool", "tool_call_id": "c1", "content": '{"ok": 1}'}


# --- transport behaviour ---------------------------------------------------------------

def test_retries_rate_limit_then_succeeds(monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake = FakeGroq([httpx.Response(429, headers={"retry-after": "1"}), completion("ok")])
    assert groq(fake).step("S", TOOLS, [{"role": "user", "text": "q"}]).text == "ok" and len(fake.requests) == 2


def test_client_error_is_not_retried():
    fake = FakeGroq([httpx.Response(400, json={"error": {"code": "tool_use_failed"}})])
    with pytest.raises(LLMError, match="400"):
        groq(fake).step("S", TOOLS, [{"role": "user", "text": "q"}])
    assert len(fake.requests) == 1


def test_usage_recorded():
    fake = FakeGroq([completion("ok")])
    p = groq(fake)
    p.step("S", TOOLS, [{"role": "user", "text": "q"}])
    assert p.usage == [{"input": 100, "output": 20}]


# --- Anthropic translation (fake client, no SDK call) ----------------------------------

def test_anthropic_translation():
    blocks = [NS(type="thinking", thinking=""), NS(type="text", text="hm"),
              NS(type="tool_use", id="tu1", name="get_campaign", input={"campaign_id": "x"})]
    created = {}

    class FakeClient:
        class beta:
            class messages:
                @staticmethod
                def create(**kw):
                    created.update(kw)
                    return NS(content=blocks, stop_reason="tool_use",
                              usage=NS(input_tokens=1, output_tokens=2, cache_read_input_tokens=0,
                                       cache_creation_input_tokens=0))
    p = Anthropic(client=FakeClient())
    t = p.step("S", TOOLS, [{"role": "user", "text": "q"}])
    assert t.stop == "tool_calls" and t.tool_calls[0].arguments == {"campaign_id": "x"} and t.text == "hm"
    msgs = Anthropic.messages([{"role": "user", "text": "q"}, {"role": "assistant", "turn": t},
                               {"role": "tool_results", "results": [ToolResult("tu1", "{}", True)]}])
    assert msgs[1]["content"] is blocks  # thinking blocks replayed verbatim
    assert msgs[2]["content"] == [{"type": "tool_result", "tool_use_id": "tu1", "content": "{}", "is_error": True}]
    assert created["cache_control"] == {"type": "ephemeral"} and created["fallbacks"] == "default"


# --- agent loop over mocked Groq + MCP -------------------------------------------------

def run_agent(tmp_path, monkeypatch, responses, limits=Limits()):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake_fl, fake = FakeFlowlab(), FakeGroq(responses)
    server = build_server(Store(tmp_path), Flowlab(transport=httpx.MockTransport(fake_fl.handler)), 0.05)

    async def go():
        async with Client(server) as s:
            ctl = Controller(s, groq(fake), limits)
            return await ctl.research("q?", SETUP, "brief")
    return asyncio.run(go()), fake


def test_agent_loop_with_groq(tmp_path, monkeypatch):
    run, fake = run_agent(tmp_path, monkeypatch, [
        completion(None, [call("c1", "get_campaign", {})], "tool_calls"),
        completion("## Hypothesis\nnone yet")])
    assert run.stop_reason == "agent_finished" and run.final_text.startswith("## Hypothesis") and run.llm_turns == 2
    tool_msg = fake.requests[1]["messages"][-1]
    assert tool_msg["role"] == "tool" and tool_msg["tool_call_id"] == "c1"
    assert json.loads(tool_msg["content"])["campaign"]["id"] == run.campaign_id
    names = {t["function"]["name"] for t in fake.requests[0]["tools"]}
    assert "run_experiment" in names and not names & agent.CONTROLLER_TOOLS


def test_agent_handles_unparseable_arguments(tmp_path, monkeypatch):
    run, fake = run_agent(tmp_path, monkeypatch, [
        completion(None, [call("c1", "search_evidence", '{"query": ')], "tool_calls"), completion("done")])
    err = json.loads(fake.requests[1]["messages"][-1]["content"])["error"]
    assert err["kind"] == "invalid_arguments" and "invalid JSON" in err["message"]
    assert run.stop_reason == "agent_finished" and run.tool_calls == 1


def test_agent_stops_on_llm_error(tmp_path, monkeypatch):
    run, _ = run_agent(tmp_path, monkeypatch, [httpx.Response(401, text="bad key")])
    assert run.stop_reason.startswith("llm_error") and run.llm_turns == 1


def test_agent_turn_limit_with_groq(tmp_path, monkeypatch):
    loop = completion(None, [call("c", "get_campaign", {})], "tool_calls")
    run, fake = run_agent(tmp_path, monkeypatch, [loop] * 5, Limits(max_llm_turns=3))
    assert run.stop_reason == "turn_limit" and len(fake.requests) == 3


def test_cli_fails_fast_without_groq_key(monkeypatch, capsys):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    monkeypatch.setenv("RESEARCH_LLM_PROVIDER", "groq")
    args = NS(model=None, store="unused", question="q", geometry="g", length_unit="mm", up_axis="z",
              demo_objective="m:min:1", search="1,2,0.1", max_turns=1, max_experiments=1)
    assert asyncio.run(agent.amain(args)) == 2
    assert "GROQ_API_KEY" in capsys.readouterr().err


# --- context budget (serialization only; tool results themselves are untouched) -------

def test_serialize_does_not_mutate_result():
    from research_lab.agent import serialize
    out = {"url": "u", "passages": ["x" * 2000] * 6, "tier": 1, "title": "t", "status": "ok"}
    before = json.dumps(out)
    text = serialize("fetch_source", out, 3000)
    assert json.dumps(out) == before and len(text) <= 3000 + len(" ...[truncated]")
    assert len(json.loads(text)["passages"]) == 4


def test_fit_history_elides_oldest_but_keeps_latest():
    from research_lab.agent import fit_history
    big = lambda i: {"role": "tool_results", "results": [ToolResult(f"c{i}", "y" * 3000, False, "fetch_source")]}
    history = [{"role": "user", "text": "q"}] + [big(i) for i in range(5)]
    fit_history(history, 7000)
    assert history[-1]["results"][0].content == "y" * 3000
    assert json.loads(history[1]["results"][0].content)["elided"] is True


def test_think_tags_stripped():
    t = OpenAICompatible.parse(completion("<think>long private reasoning</think>Final answer."))
    assert t.text == "Final answer."


def test_pinned_parameters_must_be_mesh_fields():
    from pydantic import ValidationError
    from research_lab.store import SearchSpace
    with pytest.raises(ValidationError, match="not Flowlab mesh parameters"):
        SearchSpace(lower=6, upper=12, rel_tolerance=0.1, fixed={"optimizations": ["optimize"]})
    assert SearchSpace(lower=6, upper=12, rel_tolerance=0.1, fixed={"intent": "quality"}).fixed == {"intent": "quality"}


def test_activity_events_emitted(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake_fl, fake = FakeFlowlab(), FakeGroq([completion(None, [call("c1", "plan_next_experiment", {})], "tool_calls"),
                                            completion("done")])
    server = build_server(Store(tmp_path), Flowlab(transport=httpx.MockTransport(fake_fl.handler)), 0.05)

    async def go():
        async with Client(server) as s:
            await Controller(s, groq(fake), Limits(), on_event=events.append).research("q", SETUP, "b")
    asyncio.run(go())
    kinds = [e["kind"] for e in events]
    assert kinds[:2] == ["campaign_created", "geometry_ready"] and "tool" in kinds and kinds[-1] == "stopped"


def test_llm_schema_requires_hypothesis_and_drops_nulls():
    from research_lab.agent import llm_schema
    raw = {"type": "object", "title": "x", "required": ["campaign_id", "parameters"], "properties": {
        "campaign_id": {"type": "string", "title": "Campaign Id"},
        "parameters": {"type": "object", "additionalProperties": True},
        "hypothesis": {"anyOf": [{"type": "string"}, {"type": "null"}], "default": None, "title": "Hypothesis"}}}
    s = llm_schema("run_experiment", raw)
    assert set(s["required"]) == {"campaign_id", "parameters", "hypothesis", "rationale"}
    assert s["properties"]["hypothesis"]["type"] == "string" and "title" not in s
    assert llm_schema("get_campaign", {"properties": {"x": {"anyOf": [{"type": "integer"}, {"type": "null"}],
                                                          "default": None}}}) == {"properties": {"x": {"type": "integer"}}}
