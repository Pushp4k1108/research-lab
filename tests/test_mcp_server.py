"""MCP tool tests against a mocked Flowlab REST API (real recorded responses). No live server."""

import asyncio
import copy
import json
from pathlib import Path

import httpx
import pytest
from mcp.client import Client

from research_lab.flowlab import Flowlab
from research_lab.mcp_server import build_server
from research_lab.store import Store

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"
CAD = Path(__file__).resolve().parents[1] / "fixtures" / "smoke" / "01_geometry.json"  # any existing file
TOOLS = {"create_campaign", "setup_geometry", "run_experiment", "compare_experiments", "search_evidence",
         "fetch_source", "record_evidence", "fix_objective", "record_interpretation",
         "plan_next_experiment", "get_campaign"}
OBJ = {"metric": "minSICN", "stat": "min", "threshold": 0.005}  # test-only threshold


def rec(name):
    return json.loads((SMOKE / f"{name}.json").read_text())


class FakeFlowlab:
    """Routes recorded Flowlab responses; `generation` decides what POST /api/generations does."""

    def __init__(self):
        self.calls = []
        self.generation = "accept"  # accept | dedup | failed | 409 | 500 | never
        self.polls = 0

    def handler(self, request: httpx.Request) -> httpx.Response:
        m, p = request.method, request.url.path
        self.calls.append((m, p))
        if (m, p) == ("POST", "/api/geometry"):
            return httpx.Response(201, json=rec("01_geometry"))
        if (m, p) == ("POST", "/api/models"):
            return httpx.Response(201, json=rec("02_model_v1"))
        if m == "POST" and p.endswith("/inventory"):
            return httpx.Response(201, json=rec("03_inventory")["envelope"])
        if (m, p) == ("POST", "/api/generations"):
            cfg = json.loads(request.content)["configuration"]
            if self.generation == "409":
                return httpx.Response(409, json={"detail": {"reason": "unresolved_bindings", "groups": []}})
            if self.generation == "500":
                return httpx.Response(500, text="Internal Server Error")
            if self.generation == "dedup":
                return httpx.Response(200, json=self.envelope(cfg))
            self.cfg = cfg
            return httpx.Response(202, json=rec("04_generation_request_size12")["body"])
        if (m, p) == ("GET", "/api/generations"):
            self.polls += 1
            if self.generation == "never" or self.polls < 2:
                return httpx.Response(200, json={"generations": []})  # unresolved, not failure
            return httpx.Response(200, json={"generations": [self.envelope(self.cfg)]})
        if m == "GET" and p.startswith("/api/generations/"):
            return httpx.Response(200, json=self.envelope(self.cfg))
        return httpx.Response(404, json={"detail": "unrouted"})

    def envelope(self, cfg):
        env = copy.deepcopy(rec("05_generation_size12"))
        env["generation"]["configuration"].update(cfg)
        if self.generation == "failed":
            env["status"] = env["generation"]["status"] = "failed"
            env["generation"]["error"] = {"code": "cell_budget_exceeded", "message": "x", "detail": {}}
            env["generation"]["quality_report"] = None
        self.cfg = cfg
        return env


@pytest.fixture
def fake():
    return FakeFlowlab()


@pytest.fixture
def server(tmp_path, fake, monkeypatch):
    monkeypatch.setattr("time.sleep", lambda s: None)  # polling interval
    return build_server(Store(tmp_path), Flowlab(transport=httpx.MockTransport(fake.handler)), poll_timeout=0.05)


def call(server, name, args):
    async def go():
        async with Client(server) as c:
            return await c.call_tool(name, args)
    r = asyncio.run(go())
    return r if r.is_error else r.structured_content


def ready(server, **kw):
    args = {"question": "q", "max_experiments": 5, "objective": OBJ,
            "search_lower": 8, "search_upper": 12, "rel_tolerance": 0.05, **kw}
    cid = call(server, "create_campaign", args)["campaign_id"]
    call(server, "setup_geometry", {"campaign_id": cid, "geometry_path": str(CAD), "length_unit": "mm", "up_axis": "z"})
    return cid


def test_tool_registration(server):
    async def go():
        async with Client(server) as c:
            return await c.list_tools()
    tools = {t.name: t for t in asyncio.run(go()).tools}
    assert set(tools) == TOOLS
    assert set(tools["setup_geometry"].input_schema["required"]) == {"campaign_id", "geometry_path", "length_unit", "up_axis"}


def test_create_and_get_campaign(server):
    out = call(server, "create_campaign", {"question": "q", "max_experiments": 3})
    got = call(server, "get_campaign", {"campaign_id": out["campaign_id"]})
    assert got["campaign"] == out["campaign"] and got["experiments"] == [] and got["budget_used"] == 0
    assert got["campaign"]["objective"] is None and got["campaign"]["search"] is None


@pytest.mark.parametrize("args,kind", [
    ({"question": "q", "max_experiments": 6}, "invalid_arguments"),
    ({"question": "q", "max_experiments": 2, "search_lower": 1}, "invalid_arguments"),
    ({"question": "q", "max_experiments": 2, "pinned_parameters": {"intent": "quality"}}, "invalid_arguments"),
    ({"question": "q", "max_experiments": 2, "search_lower": 5, "search_upper": 1, "rel_tolerance": 0.1},
     "invalid_arguments"),
])
def test_create_campaign_rejections(server, args, kind):
    assert call(server, "create_campaign", args)["error"]["kind"] == kind


@pytest.mark.parametrize("name,args", [
    ("create_campaign", {"question": "q"}),
    ("create_campaign", {"question": "q", "max_experiments": "many"}),
    ("create_campaign", {"question": "q", "max_experiments": 2, "objective": {"metric": "m", "stat": "median", "threshold": 1}}),
    ("setup_geometry", {"campaign_id": "x", "geometry_path": "p", "length_unit": "furlong", "up_axis": "z"}),
    ("run_experiment", {"campaign_id": "x", "parameters": "size=10"}),
])
def test_malformed_arguments_are_mcp_errors(server, name, args):
    r = call(server, name, args)
    assert r.is_error


def test_unknown_campaign(server):
    for name in ("get_campaign", "compare_experiments", "plan_next_experiment"):
        assert call(server, name, {"campaign_id": "nope"})["error"]["kind"] == "not_found"


def test_setup_geometry_records_flowlab_ids(server, fake):
    cid = ready(server)
    g = call(server, "get_campaign", {"campaign_id": cid})["campaign"]["geometry"]
    inv = rec("03_inventory")["envelope"]
    assert g == {"geometry_ref": rec("01_geometry")["geometry_ref"], "model_id": inv["model_id"],
                 "model_version": inv["version"], "snapshot_hash": inv["snapshot_hash"]}
    again = call(server, "setup_geometry", {"campaign_id": cid, "geometry_path": str(CAD), "length_unit": "mm", "up_axis": "z"})
    assert again["error"]["kind"] == "invalid_state"


def test_run_before_setup_is_invalid_state(server):
    cid = call(server, "create_campaign", {"question": "q", "max_experiments": 2})["campaign_id"]
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 10}})
    assert r["error"]["kind"] == "invalid_state"


def test_run_experiment_accept_poll_normalize(server, fake):
    cid = ready(server)
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12},
                                        "hypothesis": "h"})
    assert r["outcome"] == "ok" and r["repeat"] is False
    res = r["experiment"]["result"]
    assert res["element_count"] == 5357 and res["node_count"] is None
    assert set(res["metrics"]) == {"gamma", "minSICN"} and res["feasible"] is False  # 0.0041 < 0.005
    assert fake.polls == 2  # first poll empty, second resolved
    assert r["experiment"]["hypothesis"] == "h"


def test_repeat_is_free(server, fake):
    cid = ready(server)
    call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    n = len(fake.calls)
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12.0}})
    assert r["repeat"] is True and len(fake.calls) == n
    assert call(server, "get_campaign", {"campaign_id": cid})["budget_used"] == 1


def test_parameters_must_match_search(server):
    cid = ready(server, pinned_parameters={"intent": "quality"})
    for params in ({"target_element_size": 10}, {"target_element_size": 50, "intent": "quality"},
                   {"target_element_size": True, "intent": "quality"}):
        r = call(server, "run_experiment", {"campaign_id": cid, "parameters": params})
        assert r["error"]["kind"] == "invalid_arguments"


def test_budget_enforced(server, fake):
    cid = ready(server, max_experiments=1)
    call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    n = len(fake.calls)
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 10}})
    assert r["error"]["kind"] == "budget_exhausted" and len(fake.calls) == n  # Flowlab not called
    assert call(server, "plan_next_experiment", {"campaign_id": cid})["reason"] == "budget_exhausted"


def test_failed_generation_is_engineering_outcome(server, fake):
    cid = ready(server)
    fake.generation = "failed"
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    res = r["experiment"]["result"]
    assert r["outcome"] == "failed" and res["error_code"] == "cell_budget_exceeded"
    assert res["metrics"] == {} and call(server, "get_campaign", {"campaign_id": cid})["budget_used"] == 1


def test_infra_error_is_not_evidence_and_free(server, fake):
    cid = ready(server)
    fake.generation = "500"
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    assert r["outcome"] == "infra_error" and r["experiment"]["result"]["metrics"] == {}
    assert call(server, "get_campaign", {"campaign_id": cid})["budget_used"] == 0
    call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    d = call(server, "plan_next_experiment", {"campaign_id": cid})
    assert d["status"] == "stop" and d["reason"] == "infrastructure_unavailable"


def test_refused_is_not_retried(server, fake):
    cid = ready(server)
    fake.generation = "409"
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    assert r["outcome"] == "refused" and r["experiment"]["result"]["error_code"] == "unresolved_bindings"
    assert sum(c == ("POST", "/api/generations") for c in fake.calls) == 1
    assert call(server, "plan_next_experiment", {"campaign_id": cid})["status"] == "cannot_plan"


def test_poll_timeout_is_unresolved(server, fake):
    cid = ready(server)
    fake.generation = "never"
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    assert r["outcome"] == "unresolved" and r["experiment"]["result"]["input_hash"].startswith("sha256:")
    assert call(server, "plan_next_experiment", {"campaign_id": cid})["reason"] == "unresolved_pending"


def test_dedup_200_path(server, fake):
    cid = ready(server)
    fake.generation = "dedup"
    r = call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 10}})
    assert r["outcome"] == "ok" and fake.polls == 0


def test_compare_and_plan_tools(server, fake):
    cid = ready(server)
    assert call(server, "plan_next_experiment", {"campaign_id": cid})["next_parameters"] == {"target_element_size": 12.0}
    call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 12}})
    fake.generation, fake.polls = "accept", 0
    env10 = rec("05_generation_size10_dedup")
    fake.envelope = lambda cfg: env10  # serve the real size-10 result
    call(server, "run_experiment", {"campaign_id": cid, "parameters": {"target_element_size": 10}})
    s = call(server, "compare_experiments", {"campaign_id": cid})
    assert s["feasible"] == [f"{cid}-002"] and s["infeasible"] == [f"{cid}-001"]
    assert s["trend"]["series"]["element_count"]["direction"] == "non_increasing"
    d = call(server, "plan_next_experiment", {"campaign_id": cid, "agent_suggests_stop": True})
    assert d["status"] == "propose" and d["next_parameters"] == {"target_element_size": 10.95}
    assert d["agent_suggested_stop"] is True and d["decided_by"] == "planner"
