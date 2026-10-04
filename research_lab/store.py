"""Append-only local campaign store. Research Lab state only; Flowlab state stays in Flowlab.

Layout:  <root>/<campaign_id>/campaign.json      written at creation, geometry set once
         <root>/<campaign_id>/experiments.jsonl  one Experiment per line, append-only
"""

from __future__ import annotations

import hashlib
import json
import os
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from research_lab.results import Constraint, Result

MAX_EXPERIMENTS = 5
DEMO_BASIS = "demo/user-supplied threshold; unsupported by retrieved evidence"
# Outcomes that are engineering observations and therefore spend budget.
BUDGET_OUTCOMES = frozenset({"ok", "failed"})


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


class Geometry(BaseModel):
    model_config = ConfigDict(frozen=True)
    geometry_ref: str
    model_id: str
    model_version: int
    snapshot_hash: str


# Flowlab MeshConfiguration fields a search may pin (public API contract; target_element_size is
# the searched parameter). Anything else would be refused by Flowlab with a 422.
MESH_FIELDS = frozenset({"dimension", "min_element_size", "curvature_resolution", "feature_angle_degrees",
                         "element_order", "intent", "boundary_layers", "cell_budget", "seed"})


class SearchSpace(BaseModel):
    """Explicit bounded search over one mesh parameter; every other parameter is pinned."""

    model_config = ConfigDict(frozen=True)
    parameter: str = "target_element_size"
    lower: float = Field(gt=0)
    upper: float = Field(gt=0)
    rel_tolerance: float = Field(gt=0, lt=1)  # stop when (infeasible - feasible) / feasible <= this
    fixed: dict[str, Any] = {}  # the other requested parameters, identical for every experiment

    @model_validator(mode="after")
    def _ordered(self):
        if self.upper <= self.lower:
            raise ValueError("upper must exceed lower")
        if self.parameter in self.fixed:
            raise ValueError("searched parameter cannot also be fixed")
        unknown = set(self.fixed) - MESH_FIELDS
        if unknown:
            raise ValueError(f"not Flowlab mesh parameters: {sorted(unknown)}; allowed: {sorted(MESH_FIELDS)}")
        return self

    def parameters(self, value: float) -> dict[str, Any]:
        return {**self.fixed, self.parameter: value}


class Campaign(BaseModel):
    model_config = ConfigDict(frozen=True)
    id: str
    question: str
    objective: Constraint | None = None  # None: compare only, no feasibility verdicts
    max_experiments: int = Field(ge=1, le=MAX_EXPERIMENTS)
    created_at: str
    geometry: Geometry | None = None
    search: SearchSpace | None = None  # None: no directed search, planner cannot propose
    # Where the objective's threshold comes from: DEMO_BASIS, or "evidence:<id>" of an
    # admissible numerical_threshold record. Never left implicit.
    objective_basis: str | None = None
    # Template/configuration concept: which template created this campaign and the
    # original uploaded geometry filename (for display only; Flowlab IDs stay in geometry).
    template: str | None = None
    geometry_name: str | None = None


class Experiment(BaseModel):
    model_config = ConfigDict(frozen=True)
    experiment_id: str
    number: int
    timestamp: str
    parameters: dict[str, Any]  # as requested; Flowlab's echoed config is result.configuration
    hypothesis: str | None = None
    rationale: str | None = None
    result: Result
    # True: Flowlab answered from an existing Generation (HTTP 200); False: newly meshed (202).
    flowlab_existing: bool | None = None


class StoreError(Exception):
    pass


class Store:
    def __init__(self, root: Path):
        self.root = Path(root)

    def _dir(self, campaign_id: str) -> Path:
        d = self.root / campaign_id
        if not (d / "campaign.json").is_file():
            raise StoreError(f"no campaign {campaign_id}")
        return d

    def _write_campaign(self, campaign: Campaign) -> None:
        d = self.root / campaign.id
        d.mkdir(parents=True, exist_ok=True)
        tmp = d / "campaign.json.tmp"
        tmp.write_text(campaign.model_dump_json(indent=2))
        os.replace(tmp, d / "campaign.json")

    def create_campaign(self, question: str, max_experiments: int, objective: Constraint | None = None,
                        search: SearchSpace | None = None, template: str | None = None,
                        geometry_name: str | None = None) -> Campaign:
        campaign = Campaign(id=uuid.uuid4().hex[:12], question=question, objective=objective,
                            max_experiments=max_experiments, created_at=_now(), search=search,
                            objective_basis=DEMO_BASIS if objective is not None else None,
                            template=template, geometry_name=geometry_name)
        self._write_campaign(campaign)
        return campaign

    def campaign(self, campaign_id: str) -> Campaign:
        return Campaign.model_validate_json((self._dir(campaign_id) / "campaign.json").read_text())

    def set_geometry(self, campaign_id: str, geometry: Geometry) -> Campaign:
        campaign = self.campaign(campaign_id)
        if campaign.geometry is not None and campaign.geometry != geometry:
            raise StoreError("campaign geometry is already set")
        updated = campaign.model_copy(update={"geometry": geometry})
        self._write_campaign(updated)
        return updated

    def set_objective(self, campaign_id: str, objective: Constraint, basis: str,
                      search: SearchSpace | None) -> Campaign:
        """Fix the objective once, before any experiment has been recorded."""
        campaign = self.campaign(campaign_id)
        if campaign.objective is not None:
            raise StoreError("campaign objective is already fixed")
        if self.experiments(campaign_id):
            raise StoreError("objective must be fixed before the first experiment")
        updated = campaign.model_copy(update={"objective": objective, "objective_basis": basis,
                                              "search": search or campaign.search})
        self._write_campaign(updated)
        return updated

    # --- append-only side records: evidence, agent notes ---------------------------------

    def _records(self, campaign_id: str, name: str) -> list[dict[str, Any]]:
        path = self._dir(campaign_id) / f"{name}.jsonl"
        if not path.exists():
            return []
        return [json.loads(line) for line in path.read_text().splitlines() if line]

    def _append_record(self, campaign_id: str, name: str, record: dict[str, Any]) -> dict[str, Any]:
        with open(self._dir(campaign_id) / f"{name}.jsonl", "a") as f:
            f.write(json.dumps(record, sort_keys=True) + "\n")
        return record

    def evidence(self, campaign_id: str) -> list[dict[str, Any]]:
        return self._records(campaign_id, "evidence")

    def append_evidence(self, campaign_id: str, record: dict[str, Any]) -> dict[str, Any]:
        n = len(self.evidence(campaign_id)) + 1
        return self._append_record(campaign_id, "evidence", {**record, "id": f"ev-{n:03d}", "recorded_at": _now()})

    def notes(self, campaign_id: str) -> list[dict[str, Any]]:
        return self._records(campaign_id, "notes")

    def append_note(self, campaign_id: str, record: dict[str, Any]) -> dict[str, Any]:
        return self._append_record(campaign_id, "notes", {**record, "recorded_at": _now()})

    # --- per-campaign research cache (provider responses) ---------------------------------

    def cache_get(self, campaign_id: str, kind: str, key: str) -> Any | None:
        path = self._dir(campaign_id) / "cache" / kind / f"{hashlib.sha256(key.encode()).hexdigest()}.json"
        return json.loads(path.read_text()) if path.exists() else None

    def cache_put(self, campaign_id: str, kind: str, key: str, value: Any) -> None:
        d = self._dir(campaign_id) / "cache" / kind
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{hashlib.sha256(key.encode()).hexdigest()}.json").write_text(json.dumps({"key": key, **value}))

    def experiments(self, campaign_id: str) -> list[Experiment]:
        path = self._dir(campaign_id) / "experiments.jsonl"
        if not path.exists():
            return []
        return [Experiment.model_validate_json(line) for line in path.read_text().splitlines() if line]

    def append(self, campaign_id: str, parameters: dict[str, Any], result: Result,
               hypothesis: str | None = None, rationale: str | None = None,
               flowlab_existing: bool | None = None) -> Experiment:
        number = len(self.experiments(campaign_id)) + 1
        exp = Experiment(experiment_id=f"{campaign_id}-{number:03d}", number=number, timestamp=_now(),
                         parameters=parameters, hypothesis=hypothesis, rationale=rationale, result=result,
                         flowlab_existing=flowlab_existing)
        with open(self._dir(campaign_id) / "experiments.jsonl", "a") as f:
            f.write(exp.model_dump_json() + "\n")
        return exp

    def budget_used(self, campaign_id: str) -> int:
        return sum(e.result.outcome in BUDGET_OUTCOMES for e in self.experiments(campaign_id))

    def find_repeat(self, campaign_id: str, parameters: dict[str, Any]) -> Experiment | None:
        """Earliest stored engineering observation for identical requested parameters."""
        for e in self.experiments(campaign_id):
            if e.result.outcome in BUDGET_OUTCOMES and e.parameters == parameters:
                return e
        return None

