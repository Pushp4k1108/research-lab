"""Deterministic comparison of stored experiments.

Reports observed deltas and trends only. Feasibility and "best" appear only when the
campaign supplies an objective; nothing here judges a metric good or bad on its own.
"""

from __future__ import annotations

from typing import Any

from research_lab.results import Constraint, check
from research_lab.store import Experiment

STATS = ("min", "mean", "max")


def _delta(a: float | None, b: float | None) -> dict[str, float] | None:
    if a is None or b is None:
        return None
    out = {"from": a, "to": b, "abs": b - a}
    if a != 0:
        out["rel"] = (b - a) / abs(a)
    return out


def _config(e: Experiment) -> dict[str, Any]:
    # Flowlab's echoed configuration (defaults filled) when it exists, else what was requested.
    return e.result.configuration or e.parameters


def pair(a: Experiment, b: Experiment) -> dict[str, Any]:
    """Change from a to b. Quantities are compared only between two ok results."""
    ca, cb = _config(a), _config(b)
    out: dict[str, Any] = {
        "from": a.experiment_id, "to": b.experiment_id,
        "outcomes": [a.result.outcome, b.result.outcome],
        "changed_parameters": {k: [ca.get(k), cb.get(k)] for k in sorted(ca.keys() | cb.keys())
                               if ca.get(k) != cb.get(k)},
    }
    if a.result.outcome != "ok" or b.result.outcome != "ok":
        out["comparable"] = False
        return out
    ra, rb = a.result, b.result
    shared = sorted(ra.metrics.keys() & rb.metrics.keys())
    out.update(
        comparable=True,
        element_count=_delta(ra.element_count, rb.element_count),
        node_count=_delta(ra.node_count, rb.node_count),
        metrics={m: {s: _delta(getattr(ra.metrics[m], s), getattr(rb.metrics[m], s)) for s in STATS}
                 for m in shared},
        metrics_only_in={a.experiment_id: sorted(ra.metrics.keys() - rb.metrics.keys()),
                         b.experiment_id: sorted(rb.metrics.keys() - ra.metrics.keys())},
    )
    return out


def _direction(values: list[float]) -> str:
    steps = [b - a for a, b in zip(values, values[1:])]
    if all(s == 0 for s in steps):
        return "constant"
    if all(s >= 0 for s in steps):
        return "non_decreasing"
    if all(s <= 0 for s in steps):
        return "non_increasing"
    return "non_monotonic"


def trend(experiments: list[Experiment], parameter: str) -> dict[str, Any]:
    """How each observed quantity moves as `parameter` increases, over ok results only.

    Refused as "confounded" when any other parameter also differs between the points.
    """
    ok = [e for e in experiments if e.result.outcome == "ok" and parameter in _config(e)]
    points = sorted(ok, key=lambda e: _config(e)[parameter])
    if len(points) < 2:
        return {"parameter": parameter, "status": "insufficient_points", "points": len(points)}
    others = [{k: v for k, v in _config(e).items() if k != parameter} for e in points]
    if any(o != others[0] for o in others[1:]):
        return {"parameter": parameter, "status": "confounded", "points": len(points)}
    if len({_config(e)[parameter] for e in points}) < len(points):
        return {"parameter": parameter, "status": "duplicate_values", "points": len(points)}

    series: dict[str, list[float | None]] = {
        "element_count": [e.result.element_count for e in points],
        "node_count": [e.result.node_count for e in points],
    }
    shared = set.intersection(*(set(e.result.metrics) for e in points))
    for m in sorted(shared):
        for s in STATS:
            series[f"{m}.{s}"] = [getattr(e.result.metrics[m], s) for e in points]
    return {
        "parameter": parameter, "status": "ok",
        "values": [_config(e)[parameter] for e in points],
        "experiments": [e.experiment_id for e in points],
        "series": {k: {"values": v, "direction": _direction(v) if None not in v else "not_measured"}
                   for k, v in series.items()},
    }


def summarize(experiments: list[Experiment], objective: Constraint | None = None,
              parameter: str = "target_element_size") -> dict[str, Any]:
    by_outcome: dict[str, list[str]] = {}
    for e in experiments:
        by_outcome.setdefault(e.result.outcome, []).append(e.experiment_id)
    ok = [e for e in experiments if e.result.outcome == "ok"]
    out: dict[str, Any] = {
        "count": len(experiments),
        "by_outcome": by_outcome,
        "history": [{"experiment_id": e.experiment_id, "parameters": e.parameters,
                     "outcome": e.result.outcome, "generation_id": e.result.generation_id,
                     "element_count": e.result.element_count, "node_count": e.result.node_count,
                     "error_code": e.result.error_code, "feasible": e.result.feasible}
                    for e in experiments],
        "consecutive": [pair(a, b) for a, b in zip(ok, ok[1:])],
        "trend": trend(experiments, parameter),
    }
    if objective is not None:
        # Feasibility re-derived from the campaign objective, not trusted from stored results.
        verdicts = {e.experiment_id: check(e.result.metrics, objective) for e in ok}
        feasible = [e for e in ok if verdicts[e.experiment_id][0] is True]
        out["objective"] = objective.model_dump()
        out["feasible"] = [e.experiment_id for e in feasible]
        out["infeasible"] = [i for i, (f, _) in verdicts.items() if f is False]
        out["undetermined"] = [i for i, (f, _) in verdicts.items() if f is None]
        out["verdicts"] = {i: r for i, (_, r) in verdicts.items()}
        counted = [e for e in feasible if e.result.element_count is not None]
        best = min(counted, key=lambda e: (e.result.element_count, e.number), default=None)
        out["best_feasible_min_elements"] = best.experiment_id if best else None
    return out
