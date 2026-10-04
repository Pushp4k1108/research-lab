"""Deterministic experiment planner: continue, stop, or cannot_plan. No LLM involvement.

Objective (MVP): fewest elements subject to one explicit metric constraint, searched over one
bounded parameter (target_element_size). The search assumes, and checks against the data, that
  * element_count does not increase as the size grows, and
  * feasibility switches once: feasible below some size, infeasible above it.
If the observations contradict either assumption the planner refuses (cannot_plan) instead of
extrapolating. Strategy: test the upper bound, then the lower bound, then bisect the
feasible/infeasible bracket geometrically until it is narrower than rel_tolerance.

The planner is authoritative. An agent's wish to stop is recorded, never acted on here.
"""

from __future__ import annotations

import math
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict

from research_lab.results import check
from research_lab.store import BUDGET_OUTCOMES, Campaign, Experiment

Status = Literal["propose", "stop", "cannot_plan"]
MIN_POINTS_FOR_NARROWED = 3  # a bracket from two points is not a converged search
SIG_DIGITS = 4  # proposals are rounded so requested parameters are reproducible


class Decision(BaseModel):
    model_config = ConfigDict(frozen=True)
    status: Status
    reason: str
    decided_by: Literal["planner"] = "planner"
    agent_suggested_stop: bool = False
    next_parameters: dict[str, Any] | None = None
    best_feasible: str | None = None  # experiment_id
    bracket: dict[str, float | None] = {}
    budget: dict[str, int]
    notes: list[str] = []


def _round(v: float) -> float:
    return float(f"{v:.{SIG_DIGITS}g}")


def plan(campaign: Campaign, experiments: list[Experiment], agent_suggests_stop: bool = False) -> Decision:
    used = sum(e.result.outcome in BUDGET_OUTCOMES for e in experiments)
    budget = {"used": used, "max": campaign.max_experiments, "remaining": max(campaign.max_experiments - used, 0)}

    def decide(status: Status, reason: str, **kw) -> Decision:
        return Decision(status=status, reason=reason, agent_suggested_stop=agent_suggests_stop, budget=budget, **kw)

    # --- non-evidence at the tail: these block planning without being measurements --------
    tail = [e.result.outcome for e in experiments]
    if tail[-2:] == ["infra_error", "infra_error"]:
        return decide("stop", "infrastructure_unavailable",
                      notes=["two consecutive infrastructure errors; not engineering evidence"])
    if tail and tail[-1] == "refused":
        return decide("cannot_plan", f"request_refused:{experiments[-1].result.error_code}",
                      notes=["Flowlab refused the request: fix the model/configuration, do not retry"])
    if tail and tail[-1] == "unresolved":
        return decide("cannot_plan", "unresolved_pending",
                      notes=[f"re-poll input_hash {experiments[-1].result.input_hash} before planning"])

    objective, search = campaign.objective, campaign.search
    if objective is None:
        return decide("cannot_plan", "no_objective")
    if search is None:
        return decide("cannot_plan", "no_search_space")

    # --- validate the engineering history ------------------------------------------------
    valid = [e for e in experiments if e.result.outcome in BUDGET_OUTCOMES]
    p = search.parameter
    for e in valid:
        if p not in e.parameters or e.parameters != search.parameters(e.parameters[p]):
            return decide("cannot_plan", "confounded_history",
                          notes=[f"{e.experiment_id} parameters differ from the pinned search: {e.parameters}"])
        if not search.lower <= e.parameters[p] <= search.upper:
            return decide("cannot_plan", "history_out_of_bounds", notes=[e.experiment_id])

    # size -> feasible?  ok runs use the campaign objective; failed runs mean "no mesh",
    # which cannot satisfy the objective. Their (absent) metrics are never read.
    verdict: dict[float, bool] = {}
    for e in valid:
        size = float(e.parameters[p])
        if e.result.outcome == "ok":
            f, why = check(e.result.metrics, objective)
            if f is None:
                return decide("cannot_plan", "undetermined_feasibility", notes=[f"{e.experiment_id}: {why}"])
        else:
            f = False
        if verdict.get(size, f) != f:
            return decide("cannot_plan", "inconsistent_repeats", notes=[f"{p}={size} gave conflicting outcomes"])
        verdict[size] = f

    ok_sorted = sorted((e for e in valid if e.result.outcome == "ok"), key=lambda e: e.parameters[p])
    counts = [e.result.element_count for e in ok_sorted]
    if None in counts:
        return decide("cannot_plan", "element_count_missing")
    if any(b > a for a, b in zip(counts, counts[1:])):
        return decide("cannot_plan", "non_monotonic_element_count",
                      notes=["element_count rose with size; 'largest feasible size' no longer implies fewest elements"])

    feasible = sorted(s for s, f in verdict.items() if f)
    infeasible = sorted(s for s, f in verdict.items() if not f)
    if feasible and infeasible and feasible[-1] > infeasible[0]:
        return decide("cannot_plan", "non_monotonic_feasibility",
                      notes=[f"infeasible at {infeasible[0]} but feasible at {feasible[-1]}"])

    f_max = feasible[-1] if feasible else None
    i_min = infeasible[0] if infeasible else None
    bracket = {"feasible_max": f_max, "infeasible_min": i_min}
    best = next((e.experiment_id for e in valid if e.result.outcome == "ok"
                 and float(e.parameters[p]) == f_max), None) if f_max is not None else None
    common = dict(bracket=bracket, best_feasible=best)

    if budget["remaining"] == 0:
        return decide("stop", "budget_exhausted", **common)

    def propose(value: float, reason: str) -> Decision:
        value = _round(value)
        if value in verdict:
            return decide("stop", "resolution_limit", **common,
                          notes=[f"next {p} rounds to an already-tested value at {SIG_DIGITS} significant digits"])
        return decide("propose", reason, next_parameters=search.parameters(value), **common)

    if f_max is None:
        if not verdict:
            return propose(search.upper, "baseline_upper_bound")
        if search.lower in verdict:
            return decide("stop", "infeasible_in_range", **common,
                          notes=[f"objective not met anywhere tested, including the lower bound {search.lower}"])
        return propose(search.lower, "probe_lower_bound")
    if i_min is None:
        if f_max == search.upper:
            return decide("stop", "objective_satisfied_at_upper_bound", **common,
                          notes=["feasible at the largest allowed size; no larger size may be tried"])
        return propose(search.upper, "probe_upper_bound")

    width = (i_min - f_max) / f_max
    if width <= search.rel_tolerance and len(verdict) >= MIN_POINTS_FOR_NARROWED:
        return decide("stop", "interval_narrowed", **common, notes=[f"relative bracket width {width:.4g}"])
    return propose(math.sqrt(f_max * i_min), "bisect_bracket")
