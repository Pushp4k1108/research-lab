import copy
import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from research_lab.flowlab import FlowlabError, PollTimeout
from research_lab.planner import plan
from research_lab.results import Constraint, classify_exception, normalize_generation
from research_lab.store import Campaign, Experiment, SearchSpace

SMOKE = Path(__file__).resolve().parents[1] / "fixtures" / "smoke"
REAL = {name: json.loads((SMOKE / f"05_generation_{name}.json").read_text()) for name in ("size10_dedup", "size12")}
# Test-only threshold: real minSICN.min is 0.0075 at size 10 and 0.0041 at size 12.
OBJ = Constraint(metric="minSICN", stat="min", threshold=0.005)
SEARCH = SearchSpace(lower=2, upper=20, rel_tolerance=0.1)


def campaign(objective=OBJ, search=SEARCH, max_experiments=5):
    return Campaign(id="c", question="q", objective=objective, max_experiments=max_experiments,
                    created_at="t", search=search)


def exp(n, result, parameters):
    return Experiment(experiment_id=f"c-{n:03d}", number=n, timestamp="t", parameters=parameters, result=result)


def ok(n, size, min_sicn, elements, **extra):
    env = copy.deepcopy(REAL["size12"])
    env["generation"]["configuration"]["target_element_size"] = size
    env["generation"]["quality_report"]["element_count"] = elements
    env["generation"]["quality_report"]["metrics"]["minSICN"]["min"] = min_sicn
    return exp(n, normalize_generation(env), {"target_element_size": size, **extra})


def failed(n, size):
    env = copy.deepcopy(REAL["size12"])
    env["status"] = env["generation"]["status"] = "failed"
    env["generation"]["error"] = {"code": "mesh_failed", "message": "x", "detail": {}}
    env["generation"]["quality_report"] = None
    return exp(n, normalize_generation(env), {"target_element_size": size})


def other(n, size, exc):
    return exp(n, classify_exception(exc), {"target_element_size": size})


def real_history():
    return [exp(1, normalize_generation(REAL["size10_dedup"]), {"target_element_size": 10}),
            exp(2, normalize_generation(REAL["size12"]), {"target_element_size": 12})]


def test_search_space_validation():
    with pytest.raises(ValidationError):
        SearchSpace(lower=5, upper=5, rel_tolerance=0.1)
    with pytest.raises(ValidationError):
        SearchSpace(lower=1, upper=5, rel_tolerance=0.1, fixed={"target_element_size": 3})


def test_empty_campaign_proposes_upper_bound():
    d = plan(campaign(), [])
    assert d.status == "propose" and d.reason == "baseline_upper_bound"
    assert d.next_parameters == {"target_element_size": 20.0}
    assert d.budget == {"used": 0, "max": 5, "remaining": 5}


def test_fixed_parameters_are_carried():
    s = SearchSpace(lower=2, upper=20, rel_tolerance=0.1, fixed={"intent": "quality"})
    assert plan(campaign(search=s), []).next_parameters == {"intent": "quality", "target_element_size": 20.0}


@pytest.mark.parametrize("kw,reason", [({"objective": None}, "no_objective"), ({"search": None}, "no_search_space")])
def test_cannot_plan_without_explicit_objective_or_search(kw, reason):
    d = plan(campaign(**kw), [])
    assert d.status == "cannot_plan" and d.reason == reason and d.next_parameters is None


def test_one_valid_feasible_below_upper_probes_upper():
    d = plan(campaign(), [ok(1, 10, 0.01, 6000)])
    assert d.reason == "probe_upper_bound" and d.best_feasible == "c-001"


def test_one_valid_infeasible_probes_lower():
    d = plan(campaign(), [ok(1, 20, 0.001, 3000)])
    assert d.reason == "probe_lower_bound" and d.next_parameters == {"target_element_size": 2.0}


def test_feasible_at_upper_stops():
    d = plan(campaign(), [ok(1, 20, 0.01, 3000)])
    assert d.status == "stop" and d.reason == "objective_satisfied_at_upper_bound" and d.best_feasible == "c-001"


def test_infeasible_everywhere_stops():
    d = plan(campaign(), [ok(1, 20, 0.001, 3000), ok(2, 2, 0.001, 90000)])
    assert d.status == "stop" and d.reason == "infeasible_in_range" and d.best_feasible is None


def test_real_two_points_bisect():
    d = plan(campaign(search=SearchSpace(lower=10, upper=12, rel_tolerance=0.5)), real_history())
    # width (12-10)/10 = 0.2 <= 0.5, but two points never count as a narrowed search
    assert d.status == "propose" and d.reason == "bisect_bracket"
    assert d.bracket == {"feasible_max": 10.0, "infeasible_min": 12.0} and d.best_feasible == "c-001"
    assert d.next_parameters == {"target_element_size": 10.95}  # sqrt(120), 4 s.f.


def test_interval_narrowed_needs_three_points():
    h = [ok(1, 20, 0.001, 3000), ok(2, 2, 0.01, 90000), ok(3, 10, 0.01, 6000), ok(4, 10.5, 0.001, 5800)]
    d = plan(campaign(), h)
    assert d.status == "stop" and d.reason == "interval_narrowed" and d.best_feasible == "c-003"


def test_repeated_consistent_values_are_merged():
    h = [ok(1, 20, 0.001, 3000), ok(2, 20, 0.001, 3000)]
    assert plan(campaign(), h).reason == "probe_lower_bound"


def test_repeated_conflicting_values_cannot_plan():
    h = [ok(1, 20, 0.001, 3000), ok(2, 20, 0.01, 3000)]
    assert plan(campaign(), h).reason == "inconsistent_repeats"


def test_resolution_limit():
    s = SearchSpace(lower=10, upper=10.002, rel_tolerance=0.0001)
    h = [ok(1, 10.002, 0.001, 5999), ok(2, 10, 0.01, 6000)]
    assert plan(campaign(search=s), h).reason == "resolution_limit"


def test_confounded_history():
    d = plan(campaign(), [ok(1, 20, 0.001, 3000, intent="quality")])
    assert d.status == "cannot_plan" and d.reason == "confounded_history"


def test_out_of_bounds_history():
    assert plan(campaign(), [ok(1, 50, 0.001, 900)]).reason == "history_out_of_bounds"


def test_non_monotonic_feasibility():
    h = [ok(1, 20, 0.01, 3000), ok(2, 10, 0.001, 6000)]
    assert plan(campaign(), h).reason == "non_monotonic_feasibility"


def test_non_monotonic_element_count():
    h = [ok(1, 20, 0.001, 7000), ok(2, 10, 0.01, 6000)]
    assert plan(campaign(), h).reason == "non_monotonic_element_count"


def test_missing_metric_cannot_plan():
    d = plan(campaign(objective=Constraint(metric="minSIGE", stat="min", threshold=0.1)), real_history())
    assert d.status == "cannot_plan" and d.reason == "undetermined_feasibility"


def test_failed_is_infeasible_without_metrics():
    d = plan(campaign(), [failed(1, 20)])
    assert d.reason == "probe_lower_bound" and d.budget["used"] == 1


def test_single_infra_error_is_ignored_and_free():
    d = plan(campaign(), [other(1, 20, FlowlabError(502, "x", "/"))])
    assert d.reason == "baseline_upper_bound" and d.budget["used"] == 0


def test_two_infra_errors_stop():
    h = [other(1, 20, FlowlabError(502, "x", "/")), other(2, 20, FlowlabError(503, "x", "/"))]
    d = plan(campaign(), h)
    assert d.status == "stop" and d.reason == "infrastructure_unavailable"


def test_refused_and_unresolved_block():
    r = plan(campaign(), [other(1, 20, FlowlabError(409, {"reason": "unresolved_bindings"}, "/"))])
    assert r.status == "cannot_plan" and r.reason == "request_refused:unresolved_bindings"
    u = plan(campaign(), [other(1, 20, PollTimeout("sha256:x"))])
    assert u.status == "cannot_plan" and u.reason == "unresolved_pending"


def test_budget_exhausted():
    d = plan(campaign(max_experiments=2, search=SearchSpace(lower=10, upper=12, rel_tolerance=0.01)), real_history())
    assert d.status == "stop" and d.reason == "budget_exhausted" and d.best_feasible == "c-001"
    assert d.budget["remaining"] == 0


def test_agent_stop_is_recorded_not_obeyed():
    d = plan(campaign(), [], agent_suggests_stop=True)
    assert d.status == "propose" and d.agent_suggested_stop and d.decided_by == "planner"
