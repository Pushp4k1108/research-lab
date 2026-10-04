"""Normalize Flowlab Generation responses into experiment results, deterministically.

Outcomes:
  ok          -- Generation status "ok"; its quality report is engineering data.
  failed      -- Generation status "failed"; evidence the configuration fails, no quality data.
  refused     -- Flowlab rejected the request (404/409/422): fix the model/request, don't retry.
  unresolved  -- no Generation appeared before the poll deadline; not a failure.
  infra_error -- 5xx, transport fault, or a response breaking the API contract; never evidence.

Metric names come from the response; none are assumed. A threshold is supplied by the
caller -- this module holds none.
"""

from __future__ import annotations

from typing import Any, Literal

import httpx
from pydantic import BaseModel, ConfigDict

from research_lab.flowlab import FlowlabError, PollTimeout

Outcome = Literal["ok", "failed", "refused", "unresolved", "infra_error"]
Stat = Literal["min", "max", "mean"]


class Metric(BaseModel):
    # extra="allow": fields Flowlab adds later (e.g. "scope") are kept, not dropped.
    model_config = ConfigDict(frozen=True, extra="allow")
    min: float
    max: float
    mean: float
    orientation: str | None = None


class Constraint(BaseModel):
    model_config = ConfigDict(frozen=True)
    metric: str
    stat: Stat
    threshold: float


class Result(BaseModel):
    model_config = ConfigDict(frozen=True)
    outcome: Outcome
    configuration: dict[str, Any] | None = None
    input_hash: str | None = None
    generation_id: str | None = None
    element_count: int | None = None
    node_count: int | None = None
    dimension: int | None = None
    bounding_box: dict[str, list[float]] | None = None
    metrics: dict[str, Metric] = {}
    findings: list[dict[str, Any]] = []
    unmeasured: dict[str, Any] = {}
    measured_by: str | None = None
    mesh_ref: str | None = None
    error_code: str | None = None
    error_message: str | None = None
    feasible: bool | None = None
    feasibility_reason: str


def check(metrics: dict[str, Metric], constraint: Constraint | None) -> tuple[bool | None, str]:
    """Feasibility of an ok result. Missing data or unknown orientation is None, never a pass."""
    if constraint is None:
        return None, "no_constraint"
    m = metrics.get(constraint.metric)
    if m is None:
        return None, f"metric_missing:{constraint.metric}"
    value = getattr(m, constraint.stat)
    if m.orientation == "higher_is_better":
        ok = value >= constraint.threshold
    elif m.orientation == "lower_is_better":
        ok = value <= constraint.threshold
    else:
        return None, f"orientation_unknown:{m.orientation}"
    op = ">=" if m.orientation == "higher_is_better" else "<="
    return ok, f"{constraint.metric}.{constraint.stat}={value:.6g} {op} {constraint.threshold:g}: {ok}"


def _contract_breach(reason: str, envelope: dict) -> Result:
    return Result(outcome="infra_error", generation_id=envelope.get("generation_id"),
                  error_code="contract_breach", error_message=reason, feasibility_reason="not_evidence")


def normalize_generation(envelope: dict, constraint: Constraint | None = None) -> Result:
    """Normalize a GET /api/generations/{id} (or list entry) envelope."""
    doc = envelope.get("generation")
    status = envelope.get("status")
    if not isinstance(doc, dict) or status not in ("ok", "failed") or doc.get("status") != status:
        return _contract_breach(f"unexpected envelope status={status!r}", envelope)
    common = dict(configuration=doc.get("configuration"), input_hash=doc.get("input_hash"),
                  generation_id=envelope.get("generation_id"))
    if status == "failed":
        err = doc.get("error") or {}
        return Result(outcome="failed", **common, error_code=err.get("code"),
                      error_message=err.get("message"),
                      # A configuration that yields no mesh cannot satisfy a quality objective.
                      feasible=False, feasibility_reason="generation_failed")
    q = doc.get("quality_report")
    if not isinstance(q, dict):
        return _contract_breach("ok generation without quality_report", envelope)
    try:
        metrics = {name: Metric.model_validate(v) for name, v in (q.get("metrics") or {}).items()}
    except ValueError as e:
        return _contract_breach(f"unparseable metrics: {e}", envelope)
    feasible, reason = check(metrics, constraint)
    return Result(outcome="ok", **common,
                  element_count=q.get("element_count"), node_count=q.get("node_count"),
                  dimension=q.get("dimension"), bounding_box=q.get("bounding_box"), metrics=metrics,
                  findings=q.get("findings") or [], unmeasured=q.get("unmeasured") or {},
                  measured_by=q.get("measured_by"), mesh_ref=doc.get("mesh_ref"),
                  feasible=feasible, feasibility_reason=reason)


def classify_exception(exc: Exception, configuration: dict | None = None) -> Result:
    """Map a request/poll exception to a non-evidence outcome."""
    if isinstance(exc, PollTimeout):
        return Result(outcome="unresolved", configuration=configuration, input_hash=str(exc),
                      feasibility_reason="unresolved")
    if isinstance(exc, FlowlabError) and exc.status < 500:
        detail = exc.detail if isinstance(exc.detail, dict) else {"message": exc.detail}
        return Result(outcome="refused", configuration=configuration,
                      error_code=detail.get("reason") or f"http_{exc.status}",
                      error_message=str(detail.get("message") or detail), feasibility_reason="refused")
    if isinstance(exc, (FlowlabError, httpx.HTTPError)):
        return Result(outcome="infra_error", configuration=configuration,
                      error_code=f"http_{exc.status}" if isinstance(exc, FlowlabError) else type(exc).__name__,
                      error_message=str(exc), feasibility_reason="not_evidence")
    raise exc
