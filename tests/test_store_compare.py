import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_lab.compare import pair, summarize, trend
from research_lab.flowlab import FlowlabError, PollTimeout
from research_lab.results import Constraint, classify_exception, normalize_generation
from research_lab.store import Geometry, Store, StoreError

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"


def envelope(name):
    return json.loads((SMOKE / f"05_generation_{name}.json").read_text())


@pytest.fixture
def store(tmp_path):
    return Store(tmp_path)


@pytest.fixture
def two(store):
    """Campaign holding the real size-10 and size-12 results."""
    c = store.create_campaign("q", max_experiments=5)
    store.append(c.id, {"target_element_size": 10}, normalize_generation(envelope("size10_dedup")), hypothesis="h")
    store.append(c.id, {"target_element_size": 12}, normalize_generation(envelope("size12")))
    return c


def failed_result():
    env = copy.deepcopy(envelope("size12"))
    env["status"] = env["generation"]["status"] = "failed"
    env["generation"]["error"] = {"code": "mesh_failed", "message": "x", "detail": {}}
    env["generation"]["quality_report"] = None
    env["generation"]["configuration"]["target_element_size"] = 50
    return normalize_generation(env)


# --- store -------------------------------------------------------------------

def test_budget_cap_is_five():
    with pytest.raises(ValidationError):
        Store(Path("/nonexistent")).create_campaign("q", max_experiments=6)


def test_roundtrip_preserves_result(store, two):
    exps = store.experiments(two.id)
    assert [e.experiment_id for e in exps] == [f"{two.id}-001", f"{two.id}-002"]
    assert exps[0].hypothesis == "h" and exps[0].timestamp
    r = exps[1].result
    assert r == normalize_generation(envelope("size12"))  # lossless, incl. extra metric fields
    assert r.metrics["minSICN"].model_extra == {"scope": None}
    assert r.node_count is None


def test_append_only_file(store, two):
    path = store.root / two.id / "experiments.jsonl"
    before = path.read_text()
    store.append(two.id, {"target_element_size": 14}, classify_exception(PollTimeout("sha256:x")))
    assert path.read_text().startswith(before)


def test_budget_counts_only_engineering_outcomes(store, two):
    store.append(two.id, {"target_element_size": 14}, classify_exception(FlowlabError(500, "x", "/")))
    store.append(two.id, {"target_element_size": 14}, classify_exception(PollTimeout("sha256:x")))
    store.append(two.id, {"target_element_size": 14},
                 classify_exception(FlowlabError(409, {"reason": "unresolved_bindings"}, "/")))
    store.append(two.id, {"target_element_size": 50}, failed_result())
    assert store.budget_used(two.id) == 3  # two ok + one failed


def test_repeat_ignores_non_evidence(store, two):
    assert store.find_repeat(two.id, {"target_element_size": 12.0}).number == 2
    store.append(two.id, {"target_element_size": 14}, classify_exception(FlowlabError(503, "x", "/")))
    assert store.find_repeat(two.id, {"target_element_size": 14}) is None


def test_geometry_set_once(store, two):
    g = Geometry(geometry_ref="sha256:a", model_id="m", model_version=2, snapshot_hash="sha256:s")
    store.set_geometry(two.id, g)
    store.set_geometry(two.id, g)  # idempotent
    with pytest.raises(StoreError):
        store.set_geometry(two.id, g.model_copy(update={"model_version": 3}))
    assert store.campaign(two.id).geometry == g


def test_unknown_campaign(store):
    with pytest.raises(StoreError):
        store.experiments("nope")


# --- compare -----------------------------------------------------------------

def test_real_pair_deltas(store, two):
    a, b = store.experiments(two.id)
    p = pair(a, b)
    assert p["comparable"] and p["changed_parameters"] == {"target_element_size": [10, 12]}
    assert p["element_count"]["abs"] == 5357 - 5971
    assert p["node_count"] is None  # not reported by Flowlab, so no delta
    assert set(p["metrics"]) == {"gamma", "minSICN"}
    assert p["metrics"]["minSICN"]["min"]["to"] == b.result.metrics["minSICN"].min


def test_pair_with_failed_is_not_comparable(store, two):
    f = store.append(two.id, {"target_element_size": 50}, failed_result())
    p = pair(store.experiments(two.id)[0], f)
    assert p["comparable"] is False and "element_count" not in p


def test_real_trend(store, two):
    t = trend(store.experiments(two.id), "target_element_size")
    assert t["status"] == "ok" and t["values"] == [10, 12]
    assert t["series"]["element_count"]["direction"] == "non_increasing"
    assert t["series"]["node_count"]["direction"] == "not_measured"


def test_trend_excludes_non_ok_and_flags_confounding(store, two):
    store.append(two.id, {"target_element_size": 50}, failed_result())
    assert trend(store.experiments(two.id), "target_element_size")["values"] == [10, 12]
    env = copy.deepcopy(envelope("size12"))
    env["generation"]["configuration"].update(target_element_size=14, intent="quality")
    store.append(two.id, {"target_element_size": 14, "intent": "quality"}, normalize_generation(env))
    assert trend(store.experiments(two.id), "target_element_size")["status"] == "confounded"


def test_trend_insufficient(store):
    c = store.create_campaign("q", max_experiments=1)
    assert trend(store.experiments(c.id), "target_element_size")["status"] == "insufficient_points"


def test_summary_without_objective_has_no_verdicts(store, two):
    s = summarize(store.experiments(two.id))
    assert "feasible" not in s and "best_feasible_min_elements" not in s
    assert s["by_outcome"] == {"ok": [f"{two.id}-001", f"{two.id}-002"]}


def test_summary_with_objective(store, two):
    store.append(two.id, {"target_element_size": 50}, failed_result())
    store.append(two.id, {"target_element_size": 14}, classify_exception(FlowlabError(502, "x", "/")))
    # Test thresholds only; minSICN.min is 0.0075 (size 10) and 0.0041 (size 12).
    s = summarize(store.experiments(two.id), Constraint(metric="minSICN", stat="min", threshold=0.005))
    assert s["feasible"] == [f"{two.id}-001"] and s["infeasible"] == [f"{two.id}-002"]
    assert s["best_feasible_min_elements"] == f"{two.id}-001"
    assert set(s["by_outcome"]) == {"ok", "failed", "infra_error"}
    s = summarize(store.experiments(two.id), Constraint(metric="minSICN", stat="mean", threshold=0.4))
    assert s["best_feasible_min_elements"] == f"{two.id}-002"  # fewer elements
    s = summarize(store.experiments(two.id), Constraint(metric="absent", stat="min", threshold=0.0))
    assert s["feasible"] == [] and len(s["undetermined"]) == 2 and s["best_feasible_min_elements"] is None
