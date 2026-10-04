import copy
import json
from pathlib import Path

import httpx
import pytest

from research_lab.flowlab import FlowlabError, PollTimeout
from research_lab.results import Constraint, classify_exception, normalize_generation

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


def real(name):
    return json.loads((SMOKE / f"05_generation_{name}.json").read_text())


# Thresholds below are test inputs only, not engineering claims.
@pytest.mark.parametrize("name,elements", [("size10_dedup", 5971), ("size12", 5357)])
def test_real_ok_generation(name, elements):
    r = normalize_generation(real(name))
    assert r.outcome == "ok"
    assert r.element_count == elements
    assert r.node_count is None  # Flowlab's quality report carries no node_count
    assert set(r.metrics) == {"gamma", "minSICN"}  # names exactly as returned
    assert r.metrics["minSICN"].orientation == "higher_is_better"
    assert r.metrics["minSICN"].model_extra == {"scope": None}  # unknown fields preserved
    assert r.dimension == 3 and r.measured_by == "4.15.2" and r.findings == []
    assert r.feasible is None and r.feasibility_reason == "no_constraint"


def test_real_configuration_and_hash_carried():
    r = normalize_generation(real("size12"))
    assert r.configuration["target_element_size"] == 12
    assert r.input_hash.startswith("sha256:") and r.mesh_ref.startswith("sha256:")


def test_threshold_respects_orientation():
    env = real("size12")  # minSICN.min = 0.0041, mean = 0.4935
    assert normalize_generation(env, Constraint(metric="minSICN", stat="min", threshold=0.001)).feasible is True
    assert normalize_generation(env, Constraint(metric="minSICN", stat="min", threshold=0.1)).feasible is False
    assert normalize_generation(env, Constraint(metric="minSICN", stat="mean", threshold=0.4)).feasible is True


def test_missing_metric_is_not_a_pass():
    r = normalize_generation(real("size12"), Constraint(metric="minSIGE", stat="min", threshold=0.0))
    assert r.feasible is None and r.feasibility_reason == "metric_missing:minSIGE"


def test_unknown_orientation_is_not_a_pass():
    env = copy.deepcopy(real("size12"))
    env["generation"]["quality_report"]["metrics"]["gamma"]["orientation"] = None
    assert normalize_generation(env, Constraint(metric="gamma", stat="min", threshold=0.0)).feasible is None


def test_failed_generation_has_no_quality_data():
    env = copy.deepcopy(real("size12"))
    env["status"] = env["generation"]["status"] = "failed"
    env["generation"]["error"] = {"code": "mesh_failed", "message": "x", "detail": {}}
    env["generation"]["quality_report"] = None
    r = normalize_generation(env, Constraint(metric="minSICN", stat="min", threshold=0.0))
    assert r.outcome == "failed" and r.error_code == "mesh_failed"
    assert r.metrics == {} and r.element_count is None and r.feasible is False


def test_ok_without_quality_report_is_contract_breach():
    env = copy.deepcopy(real("size12"))
    env["generation"]["quality_report"] = None
    r = normalize_generation(env)
    assert r.outcome == "infra_error" and r.error_code == "contract_breach" and r.feasible is None


def test_request_errors():
    r = classify_exception(FlowlabError(409, {"reason": "unresolved_bindings", "groups": []}, "/api/generations"))
    assert r.outcome == "refused" and r.error_code == "unresolved_bindings"
    assert classify_exception(FlowlabError(422, [{"msg": "bad"}], "/x")).error_code == "http_422"
    assert classify_exception(FlowlabError(500, "boom", "/x")).outcome == "infra_error"
    assert classify_exception(httpx.ConnectError("down")).outcome == "infra_error"
    assert classify_exception(PollTimeout("sha256:abc")).outcome == "unresolved"
