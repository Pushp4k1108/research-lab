"""Agent loop tests: scripted LLM, fake Bright Data, mocked Flowlab REST. No network."""

import asyncio
from types import SimpleNamespace as NS

import httpx
import pytest
from mcp.client import Client

from research_lab.agent import Controller, Limits, finish
from research_lab.flowlab import Flowlab
from research_lab.llm import ToolCall, Turn
from research_lab.mcp_server import build_server
from research_lab.store import DEMO_BASIS, Store
from tests.test_mcp_server import CAD, FakeFlowlab

GMSH_DOC = "https://gmsh.info/doc/texinfo/gmsh.html"
FORUM = "https://www.cfd-online.com/Forums/meshing/12345-quality.html"
PAGE = ("Gmsh can compute element quality measures. The signed inverse condition number (SICN) is a "
        "quality measure for elements; values near 1 indicate well-shaped elements and negative values "
        "indicate invalid elements. The gamma measure is the inscribed to circumscribed radius ratio.")
SETUP = {"geometry_path": str(CAD), "length_unit": "mm", "up_axis": "z"}


class FakeBrightData:
    """Conforms to research.ResearchProvider; normalizes through the shared page()."""
    name = "fake"

    def __init__(self):
        self.searches, self.fetches = [], []

    def search(self, query, n):
        self.searches.append(query)
        return [{"title": "Gmsh reference manual", "url": GMSH_DOC, "snippet": "Gmsh element quality measures",
                 "provider": self.name},
                {"title": "mesh quality thread", "url": FORUM, "snippet": "people use minSICN > 0.3 usually",
                 "provider": self.name}]

    def fetch(self, url):
        from research_lab.research import page
        self.fetches.append(url)
        return page(self.name, url, url, "text/html", f"<html><title>Gmsh manual</title><p>{PAGE}</p></html>")


def tool(name, **inp):
    return NS(type="tool_use", id=f"tu-{name}-{id(inp)}", name=name, input=inp)


def text(t):
    return NS(type="text", text=t)


class ScriptedLLM:
    """LLMProvider that replays a fixed list of turns (lists of blocks). Records what it was sent."""
    name, model = "scripted", "none"

    def __init__(self, turns, repeat_last=False):
        self.turns, self.repeat_last, self.seen, self.usage = list(turns), repeat_last, [], []

    def step(self, system, tools, history):
        self.seen.append(history[-1])
        blocks = self.turns.pop(0) if self.turns else (self.last if self.repeat_last else [text("done")])
        if callable(blocks):  # turn built from the conversation so far
            blocks = blocks(history)
        self.last = blocks
        calls = [ToolCall(b.id, b.name, dict(b.input)) for b in blocks if b.type == "tool_use"]
        return Turn(text="\n".join(b.text for b in blocks if b.type == "text"), tool_calls=calls,
                    stop="tool_calls" if calls else "end")


def results_of(llm, turn):
    """Tool results the controller returned after a given LLM turn (0-based)."""
    import json
    return [json.loads(r.content) for r in llm.seen[turn + 1]["results"]]


OBJ = dict(metric="minSICN", stat="min", threshold=0.005, threshold_basis="demo",
           search_lower=8, search_upper=12, rel_tolerance=0.05)
RUN12 = dict(parameters={"target_element_size": 12.0}, hypothesis="the upper bound violates the demo threshold",
             rationale="planner baseline")
RUN8 = dict(parameters={"target_element_size": 8.0}, hypothesis="a finer mesh improves the worst element",
            rationale="planner lower-bound probe")


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)
    fake, bd = FakeFlowlab(), FakeBrightData()
    server = build_server(Store(tmp_path), Flowlab(transport=httpx.MockTransport(fake.handler)), 0.05, bd)
    return NS(fake=fake, bd=bd, server=server, root=tmp_path)


def run(env, turns, limits=Limits(), repeat_last=False):
    llm = ScriptedLLM(turns, repeat_last)

    async def go():
        async with Client(env.server) as s:
            ctl = Controller(s, llm, limits)
            r = await ctl.research("q?", SETUP, "brief")
            report, state = await finish(ctl)
            return r, report, state
    r, report, state = asyncio.run(go())
    return llm, r, report, state


def flowlab_generation_posts(env):
    return sum(c == ("POST", "/api/generations") for c in env.fake.calls)


def test_campaign_created_by_controller_and_inspected_by_agent(env):
    llm, r, _, state = run(env, [[tool("get_campaign")]])
    got = results_of(llm, 0)[0]
    assert got["campaign"]["id"] == r.campaign_id and got["campaign"]["geometry"]["model_id"]
    assert r.stop_reason == "agent_finished" and state["campaign"]["geometry"] is not None


def test_controller_tools_hidden_from_agent(env):
    llm, r, _, _ = run(env, [[tool("create_campaign", question="x", max_experiments=5)]])
    assert results_of(llm, 0)[0]["error"]["kind"] == "unknown_tool"


def test_evidence_search_is_cached(env):
    llm, *_ = run(env, [[tool("search_evidence", query="gmsh SICN")], [tool("search_evidence", query="gmsh SICN")]])
    first = results_of(llm, 0)[0]
    assert [h["tier"] for h in first["results"]] == [1, 5]
    assert env.bd.searches == ["gmsh SICN"]  # second call served from cache


def test_evidence_preserves_provenance(env):
    excerpt = "values near 1 indicate well-shaped elements and negative values indicate invalid elements"
    turns = [[tool("search_evidence", query="gmsh SICN")],
             [tool("fetch_source", url=GMSH_DOC, keywords=["SICN"])],
             [tool("record_evidence", claim="SICN near 1 means well-shaped; negative means invalid",
                   claim_kind="metric_definition", url=GMSH_DOC, excerpt=excerpt,
                   applicability="Flowlab reports minSICN from Gmsh", reasoning="official Gmsh docs")]]
    llm, _, report, state = run(env, turns)
    e = state["evidence"][0]
    assert e["admissibility"] == "admissible" and e["excerpt_verified"] and e["retrieved_via"] == "fetched_page"
    assert e["source"]["url"] == GMSH_DOC and e["source"]["tier"] == 1
    assert e["source"]["source_type"] == "official_technical_documentation" and e["recorded_at"]
    assert "[SOURCED] (metric_definition)" in report


def test_snippets_and_forums_are_leads_only(env):
    turns = [[tool("search_evidence", query="q")],
             [tool("record_evidence", claim="minSICN > 0.3 is used", claim_kind="numerical_threshold", url=FORUM,
                   excerpt="minSICN > 0.3", metric="minSICN", threshold=0.3, applicability="a", reasoning="r")]]
    _, _, _, state = run(env, turns)
    e = state["evidence"][0]
    assert e["admissibility"] == "lead_only" and e["retrieved_via"] == "search_snippet"


def test_unsupported_threshold_is_not_sourced(env):
    turns = [[tool("search_evidence", query="q")], [tool("fetch_source", url=GMSH_DOC)],
             # number 0.3 does not appear in the excerpt -> cannot be inferred
             [tool("record_evidence", claim="SICN should exceed 0.3", claim_kind="numerical_threshold", url=GMSH_DOC,
                   excerpt="values near 1 indicate well-shaped elements", metric="minSICN", threshold=0.3,
                   applicability="a", reasoning="r")],
             [tool("fix_objective", **{**OBJ, "threshold": 0.3, "threshold_basis": "ev-001"})],
             [tool("record_evidence", claim="fabricated", claim_kind="metric_relevance", url="https://doi.org/10.1/x",
                   excerpt="anything", applicability="a", reasoning="r")],
             [tool("fix_objective", **OBJ)]]
    llm, _, report, state = run(env, turns)
    assert state["evidence"][0]["admissibility"] == "rejected"
    assert "cannot be inferred" in state["evidence"][0]["gate_reasons"][0]
    assert results_of(llm, 3)[0]["error"]["kind"] == "invalid_arguments"
    assert state["evidence"][1]["admissibility"] == "rejected"  # never retrieved
    assert state["campaign"]["objective_basis"] == DEMO_BASIS
    assert "[DEMO ASSUMPTION] " + DEMO_BASIS in report


def test_experiment_requires_hypothesis(env):
    turns = [[tool("fix_objective", **OBJ)],
             [tool("run_experiment", parameters={"target_element_size": 12.0}, hypothesis="", rationale="r")]]
    llm, *_ = run(env, turns)
    assert results_of(llm, 1)[0]["error"]["kind"] == "invalid_arguments" and flowlab_generation_posts(env) == 0


def test_experiment_runs_through_mcp_and_stores_real_measurement(env):
    turns = [[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)],
             [text("Elements were 99 and minSICN was 0.9.")]]  # LLM text cannot change the measurement
    llm, r, report, state = run(env, turns)
    seen = results_of(llm, 1)[0]
    assert seen["outcome"] == "ok" and seen["element_count"] == 5357 and seen["feasible"] is False
    assert seen["planner_after"]["next_parameters"] == {"target_element_size": 8.0}
    e = state["experiments"][0]
    assert e["result"]["element_count"] == 5357 and e["hypothesis"] == RUN12["hypothesis"]
    assert flowlab_generation_posts(env) == 1 and r.experiments_run == 1
    assert "| 5357 |" in report and "minSICN min 0.004104" in report


def test_agent_cannot_fabricate_results(env):
    turns = [[tool("store_result", element_count=1, metrics={"minSICN": {"min": 1}})],
             [tool("record_interpretation", experiment_id="nope-001", interpretation="great", next_decision="x")]]
    llm, r, _, state = run(env, turns)
    assert results_of(llm, 0)[0]["error"]["kind"] == "unknown_tool"
    assert results_of(llm, 1)[0]["error"]["kind"] == "not_found"
    assert state["experiments"] == [] and state["notes"] == []


def test_infrastructure_error_is_not_engineering_result(env):
    env.fake.generation = "500"
    turns = [[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)]]
    llm, _, _, state = run(env, turns)
    seen = results_of(llm, 1)[0]
    assert seen["outcome"] == "infra_error" and seen["metrics"] == {} and seen["feasible"] is None
    assert state["budget_used"] == 0


def test_planner_is_authoritative_on_parameters(env):
    turns = [[tool("fix_objective", **OBJ)],
             [tool("run_experiment", parameters={"target_element_size": 9.0}, hypothesis="h", rationale="r")]]
    llm, *_ = run(env, turns)
    err = results_of(llm, 1)[0]["error"]
    assert err["kind"] == "parameters_not_planned" and err["expected"] == {"target_element_size": 12.0}
    assert flowlab_generation_posts(env) == 0


def test_cannot_plan_blocks_experiments(env):
    llm, r, _, _ = run(env, [[tool("run_experiment", **RUN12)]])  # no objective fixed
    err = results_of(llm, 0)[0]["error"]
    assert err["kind"] == "planner_refused" and err["decision"]["status"] == "cannot_plan"
    assert err["decision"]["reason"] == "no_objective" and flowlab_generation_posts(env) == 0


def test_stop_is_enforced_and_agent_stops(env):
    turns = [[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)], [tool("run_experiment", **RUN8)],
             [tool("run_experiment", parameters={"target_element_size": 10.0}, hypothesis="h", rationale="r")],
             [text("## Hypothesis\n...")]]
    llm, r, report, state = run(env, turns)
    assert results_of(llm, 2)[0]["planner_after"]["status"] == "stop"
    assert results_of(llm, 2)[0]["planner_after"]["reason"] == "infeasible_in_range"
    err = results_of(llm, 3)[0]["error"]
    assert err["kind"] == "planner_refused" and err["decision"]["status"] == "stop"
    assert flowlab_generation_posts(env) == 2 and r.stop_reason == "agent_finished"
    assert "planner: stop — infeasible_in_range" in report


def test_terminal_planner_stop_is_hard_controller_barrier(env):
    events = []

    async def go():
        llm = ScriptedLLM([
            [tool("fix_objective", **OBJ)],
            [tool("run_experiment", **RUN12)],
            [tool("plan_next_experiment"), tool("fetch_source", url=GMSH_DOC)],
        ])

        async with Client(env.server) as s:
            ctl = Controller(
                s,
                llm,
                Limits(max_experiments=1),
                on_event=events.append,
            )
            r = await ctl.research("q?", SETUP, "brief")
            return llm, r

    llm, r = asyncio.run(go())

    tool_events = [e for e in events if e["kind"] == "tool"]

    assert r.stop_reason.startswith("planner_stop:")
    assert not any(e["tool"] == "fetch_source" for e in tool_events)
    assert env.bd.fetches == []
    assert r.llm_turns == 3


def test_campaign_budget_enforced(env):
    turns = [[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)], [tool("run_experiment", **RUN8)]]
    llm, r, _, _ = run(env, turns, Limits(max_experiments=1))
    err = results_of(llm, 2)[0]["error"]
    assert err["kind"] == "experiment_limit" and flowlab_generation_posts(env) == 1


def test_campaign_budget_from_planner(env):
    # controller cap above the campaign budget: the planner's budget stop still applies
    async def go():
        llm = ScriptedLLM([[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)],
                           [tool("run_experiment", **RUN8)]])
        async with Client(env.server) as s:
            ctl = Controller(s, llm, Limits(max_experiments=5))
            ctl.limits = Limits(max_experiments=5)
            orig = ctl.call

            async def small_budget(name, args):  # campaign created with budget 1
                if name == "create_campaign":
                    args = {**args, "max_experiments": 1}
                return await orig(name, args)
            ctl.call = small_budget
            await ctl.research("q", SETUP, "b")
            return llm
    llm = asyncio.run(go())
    err = results_of(llm, 2)[0]["error"]
    assert err["kind"] == "planner_refused" and err["decision"]["reason"] == "budget_exhausted"


def test_turn_limit(env):
    llm, r, _, _ = run(env, [[tool("get_campaign")]], Limits(max_llm_turns=3), repeat_last=True)
    assert r.stop_reason == "turn_limit" and r.llm_turns == 3 and r.final_text is None


def test_tool_call_limit(env):
    llm, r, _, _ = run(env, [[tool("get_campaign"), tool("get_campaign"), tool("get_campaign")]],
                       Limits(max_tool_calls=2))
    assert results_of(llm, 0)[2]["error"]["kind"] == "limit"


def test_history_survives_on_disk(env):
    turns = [[tool("search_evidence", query="gmsh SICN")], [tool("fetch_source", url=GMSH_DOC)],
             [tool("record_evidence", claim="c", claim_kind="metric_relevance", url=GMSH_DOC,
                   excerpt="Gmsh can compute element quality measures", applicability="a", reasoning="r")],
             [tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)]]
    _, r, _, _ = run(env, turns)
    store = Store(env.root)  # fresh reader over the same files
    exps, evid = store.experiments(r.campaign_id), store.evidence(r.campaign_id)
    assert len(exps) == 1 and exps[0].result.generation_id and exps[0].flowlab_existing is False
    assert evid[0]["admissibility"] == "admissible"
    assert store.campaign(r.campaign_id).objective.threshold == 0.005


def test_interpretation_recorded(env):
    def interpret(messages):
        cid = messages[0]["text"].split("Campaign: ")[1].split(" ")[0]
        return [tool("record_interpretation", experiment_id=f"{cid}-001",
                     interpretation="coarse mesh misses the demo threshold", next_decision="probe lower bound")]
    turns = [[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)], interpret, [text("summary")]]
    _, r, report, state = run(env, turns)
    assert state["notes"][0]["kind"] == "agent_interpretation" and state["notes"][0]["experiment_id"].endswith("-001")
    assert "- [AGENT] after " in report and "coarse mesh misses the demo threshold" in report
    assert state["experiments"][0]["result"]["element_count"] == 5357  # unchanged by the note


def test_fix_objective_rejects_stat_suffixed_metric(env):
    llm, *_ = run(env, [[tool("fix_objective", **{**OBJ, "metric": "minSICN.min"})]])
    err = results_of(llm, 0)[0]["error"]
    assert err["kind"] == "invalid_arguments" and "'minSICN'" in err["message"]
