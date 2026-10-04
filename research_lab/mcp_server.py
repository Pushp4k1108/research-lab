"""Research Lab MCP server (stdio). Thin tool layer over store, planner, compare and the Flowlab client.

Every tool returns a JSON object. Expected conditions are data, not exceptions:
  {"error": {"kind": ..., "message": ...}}  kinds: not_found, invalid_state, invalid_arguments,
                                                   budget_exhausted, refused, infra_error
  run_experiment -> {"outcome": ok|failed|refused|unresolved|infra_error, "repeat": bool, "experiment": ...}
  plan_next_experiment -> the planner Decision verbatim (propose | stop | cannot_plan)
Malformed arguments are rejected by the MCP layer's schema validation before a tool runs.

Run:  python -m research_lab.mcp_server
Env:  FLOWLAB_BASE_URL (default http://localhost:8000), RESEARCH_LAB_STORE (default ./campaigns),
      RESEARCH_LAB_POLL_TIMEOUT seconds (default 1200), RESEARCH_PROVIDER + provider vars (see research.py)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import httpx
from mcp.server.mcpserver import MCPServer
from pydantic import ValidationError

from research_lab import evidence as ev
from research_lab.research import (AccessRestricted, NotConfigured, ResearchError, ResearchProvider,
                                   SearchUnsupported, UnsupportedContent, provider_from_env)
from research_lab.compare import summarize
from research_lab.flowlab import Flowlab, FlowlabError, PollTimeout
from research_lab.planner import plan
from research_lab.results import Constraint, classify_exception, normalize_generation
from research_lab.store import DEMO_BASIS, Geometry, SearchSpace, Store, StoreError

LengthUnit = Literal["m", "mm", "cm", "in", "ft"]  # Flowlab's Units.length vocabulary
Axis = Literal["x", "y", "z"]


def _error(kind: str, message: str, **extra) -> dict[str, Any]:
    return {"error": {"kind": kind, "message": message, **extra}}


def _flowlab_error(exc: Exception) -> dict[str, Any]:
    r = classify_exception(exc)
    kind = "refused" if r.outcome == "refused" else "infra_error"
    return _error(kind, r.error_message or "", code=r.error_code)


def _passages(text: str, keywords: list[str], width: int, limit: int) -> list[str]:
    """Deterministic keyword windows over a page, so the agent reads passages, not pages."""
    low, spans = text.lower(), []
    for k in keywords:
        start = 0
        while (i := low.find(k.lower(), start)) != -1 and len(spans) < 4 * limit:
            spans.append((max(i - width, 0), min(i + len(k) + width, len(text))))
            start = i + len(k)
    merged: list[list[int]] = []
    for a, b in sorted(spans):
        if merged and a <= merged[-1][1]:
            merged[-1][1] = max(merged[-1][1], b)
        else:
            merged.append([a, b])
    return [text[a:b] for a, b in merged[:limit]]


def build_server(store: Store, flowlab: Flowlab, poll_timeout: float = 1200.0,
                 research: ResearchProvider | None = None) -> MCPServer:
    mcp = MCPServer("research-lab")

    def research_error(e: ResearchError) -> dict[str, Any]:
        kind = {NotConfigured: "not_configured", SearchUnsupported: "unsupported",
                UnsupportedContent: "unsupported", AccessRestricted: "access_restricted"}.get(type(e), "infra_error")
        return _error(kind, str(e), provider=getattr(research, "name", None))

    def load(campaign_id: str):
        try:
            return store.campaign(campaign_id), None
        except StoreError as e:
            return None, _error("not_found", str(e))

    @mcp.tool()
    def create_campaign(question: str, max_experiments: int, objective: Constraint | None = None,
                        search_lower: float | None = None, search_upper: float | None = None,
                        rel_tolerance: float | None = None,
                        pinned_parameters: dict[str, Any] | None = None,
                        template: str | None = None,
                        geometry_name: str | None = None) -> dict[str, Any]:
        """Create a campaign. max_experiments is 1..5. objective is an explicit metric constraint
        {metric, stat: min|mean|max, threshold}; the server supplies none. Give search_lower,
        search_upper and rel_tolerance together to define a target_element_size search; every
        other mesh parameter is pinned via pinned_parameters. template/geometry_name are
        display-only provenance recorded on the campaign."""
        bounds = (search_lower, search_upper, rel_tolerance)
        try:
            if all(b is None for b in bounds):
                if pinned_parameters:
                    return _error("invalid_arguments", "pinned_parameters requires a search range")
                search = None
            elif any(b is None for b in bounds):
                return _error("invalid_arguments", "search_lower, search_upper, rel_tolerance go together")
            else:
                search = SearchSpace(lower=search_lower, upper=search_upper, rel_tolerance=rel_tolerance,
                                     fixed=pinned_parameters or {})
            c = store.create_campaign(question, max_experiments, objective, search,
                                    template=template, geometry_name=geometry_name)
        except ValidationError as e:
            return _error("invalid_arguments", str(e))
        return {"campaign_id": c.id, "campaign": c.model_dump()}

    @mcp.tool()
    def setup_geometry(campaign_id: str, geometry_path: str, length_unit: LengthUnit,
                       up_axis: Axis, handedness: Literal["right", "left"] = "right") -> dict[str, Any]:
        """Upload an unmodified CAD file to Flowlab, create a model with zero boundary groups,
        derive its inventory, and record the Flowlab IDs on the campaign. Units and frame are
        required because Flowlab refuses to default them."""
        c, err = load(campaign_id)
        if err:
            return err
        if c.geometry is not None:
            return _error("invalid_state", "campaign geometry already set", geometry=c.geometry.model_dump())
        path = Path(geometry_path).expanduser()
        if not path.is_file():
            return _error("invalid_arguments", f"no such file: {path}")
        try:
            geometry = flowlab.upload_geometry(path)
            model = flowlab.create_model({
                "units": {"length": length_unit},
                "coord_system": {"up_axis": up_axis, "handedness": handedness},
                "geometry_ref": geometry["geometry_ref"], "boundary_groups": [], "intent_params": {},
            })
            inventory_status, measured = flowlab.derive_inventory(model["model_id"])
        except (FlowlabError, httpx.HTTPError) as e:
            return _flowlab_error(e)
        g = Geometry(geometry_ref=geometry["geometry_ref"], model_id=measured["model_id"],
                     model_version=measured["version"], snapshot_hash=measured["snapshot_hash"])
        c = store.set_geometry(campaign_id, g)
        return {"campaign_id": campaign_id, "geometry": g.model_dump(), "upload": geometry,
                "inventory_http_status": inventory_status,
                "geometry_provenance": measured["model"].get("geometry_provenance")}

    @mcp.tool()
    def run_experiment(campaign_id: str, parameters: dict[str, Any], hypothesis: str | None = None,
                       rationale: str | None = None) -> dict[str, Any]:
        """Run one mesh Generation with explicit MeshConfiguration parameters and record it.
        Identical parameters return the stored result without spending budget. With a search
        range, parameters must be the pinned parameters plus an in-range target_element_size."""
        c, err = load(campaign_id)
        if err:
            return err
        if c.geometry is None:
            return _error("invalid_state", "setup_geometry has not been run for this campaign")
        if c.search is not None:
            p = c.search.parameter
            value = parameters.get(p)
            if (isinstance(value, bool) or not isinstance(value, (int, float)) or parameters != c.search.parameters(value)
                    or not c.search.lower <= value <= c.search.upper):
                return _error("invalid_arguments", "parameters must equal the pinned search parameters "
                              f"plus {p} in [{c.search.lower}, {c.search.upper}]",
                              expected=c.search.parameters("<value>"))
        repeat = store.find_repeat(campaign_id, parameters)
        if repeat is not None:
            return {"outcome": repeat.result.outcome, "repeat": True, "experiment": repeat.model_dump()}
        used = store.budget_used(campaign_id)
        if used >= c.max_experiments:
            return _error("budget_exhausted", f"{used} of {c.max_experiments} experiments used")
        existing = None
        try:
            code, body = flowlab.request_generation(c.geometry.model_id, parameters, c.geometry.model_version)
            existing = code == 200
            if "generation_id" not in body:  # 202 accepted: poll by input_hash
                body, _ = flowlab.await_generation(body["input_hash"], timeout=poll_timeout)
            result = normalize_generation(flowlab.get_generation(body["generation_id"]), c.objective)
        except (FlowlabError, httpx.HTTPError, PollTimeout) as e:
            result = classify_exception(e, configuration=parameters)
        exp = store.append(campaign_id, parameters, result, hypothesis, rationale, flowlab_existing=existing)
        return {"outcome": result.outcome, "repeat": False, "experiment": exp.model_dump()}

    @mcp.tool()
    def compare_experiments(campaign_id: str) -> dict[str, Any]:
        """Deterministic comparison: history, outcomes, deltas, trend; feasibility only under the
        campaign's explicit objective. No interpretation."""
        c, err = load(campaign_id)
        if err:
            return err
        return summarize(store.experiments(campaign_id), c.objective)

    @mcp.tool()
    def plan_next_experiment(campaign_id: str, agent_suggests_stop: bool = False) -> dict[str, Any]:
        """The deterministic planner's decision (propose | stop | cannot_plan). Authoritative;
        agent_suggests_stop is recorded on the decision but does not change it."""
        c, err = load(campaign_id)
        if err:
            return err
        return plan(c, store.experiments(campaign_id), agent_suggests_stop).model_dump()

    @mcp.tool()
    def get_campaign(campaign_id: str) -> dict[str, Any]:
        """Campaign state, budget use and full experiment history."""
        c, err = load(campaign_id)
        if err:
            return err
        return {"campaign": c.model_dump(), "budget_used": store.budget_used(campaign_id),
                "experiments": [e.model_dump() for e in store.experiments(campaign_id)],
                "evidence": store.evidence(campaign_id), "notes": store.notes(campaign_id)}

    # --- evidence ------------------------------------------------------------------------

    @mcp.tool()
    def search_evidence(campaign_id: str, query: str, max_results: int = 6) -> dict[str, Any]:
        """Live web search through the configured research provider. Results are leads with a
        deterministic source tier, not evidence. Identical queries are served from cache. Some
        providers have no search (kind "unsupported"): use fetch_source on a known URL instead."""
        c, err = load(campaign_id)
        if err:
            return err
        cached = store.cache_get(campaign_id, "search", query)
        if cached is None:
            if research is None:
                return _error("not_configured", "no research provider configured (RESEARCH_PROVIDER)")
            try:
                hits = research.search(query, max_results)
            except ResearchError as e:
                return research_error(e)
            for h in hits:
                store.cache_put(campaign_id, "snippet", h["url"], {"title": h["title"], "snippet": h["snippet"],
                                                                    "provider": h["provider"], "query": query})
            store.cache_put(campaign_id, "search", query, {"results": hits})
            cached = {"results": hits}
        results = [{**h, **dict(zip(("tier", "source_type", "host"), ev.classify(h["url"])))}
                   for h in cached["results"]]
        return {"query": query, "results": results}

    @mcp.tool()
    def fetch_source(campaign_id: str, url: str, keywords: list[str] | None = None,
                     max_chars: int = 4000) -> dict[str, Any]:
        """Fetch an http(s) URL -- from search_evidence results or one you know -- through the
        configured provider. With keywords, returns only passages around keyword matches.
        Excerpts recorded as evidence must be verbatim from this text."""
        c, err = load(campaign_id)
        if err:
            return err
        page = store.cache_get(campaign_id, "page", url)
        if page is None:
            if research is None:
                return _error("not_configured", "no research provider configured (RESEARCH_PROVIDER)")
            try:
                page = research.fetch(url)
            except ResearchError as e:
                return research_error(e)
            found = store.cache_get(campaign_id, "snippet", url)
            page["discovered_via"] = f"search:{found['query']}" if found and found.get("query") else "agent_supplied_url"
            store.cache_put(campaign_id, "page", url, page)
        tier, source_type, host = ev.classify(url)
        text = page.get("text", "")
        body = (_passages(text, keywords, 300, 6) if keywords else [text[:max_chars]])
        return {"url": url, "title": page.get("title"), "status": page.get("status"), "tier": tier,
                "source_type": source_type, "host": host, "total_chars": len(text),
                "passages": [p[:max_chars] for p in body]}

    @mcp.tool()
    def record_evidence(campaign_id: str, claim: str, claim_kind: ev.ClaimKind, url: str, excerpt: str,
                        applicability: str, reasoning: str, metric: str | None = None,
                        threshold: float | None = None, title: str | None = None,
                        publisher: str | None = None, published: str | None = None,
                        supports: list[str] | None = None, contradicts: list[str] | None = None) -> dict[str, Any]:
        """Record a claim with provenance. The gate (tier, verbatim-excerpt check, threshold-stated
        check) is applied by code; admissibility is returned, never chosen by the caller.
        claim_kind distinguishes metric_relevance, metric_definition and numerical_threshold."""
        c, err = load(campaign_id)
        if err:
            return err
        existing = {e["id"]: e for e in store.evidence(campaign_id)}
        page = store.cache_get(campaign_id, "page", url)
        snip = store.cache_get(campaign_id, "snippet", url)
        # Tier follows where the text actually came from (after redirects), never the caller.
        g = ev.gate(claim_kind=claim_kind, url=(page or {}).get("final_url") or url, excerpt=excerpt, page=page,
                    snippet=snip["snippet"] if snip else None, metric=metric, threshold=threshold,
                    existing=existing, supports=supports or [], contradicts=contradicts or [])
        record = {"claim": claim, "claim_kind": claim_kind, "metric": metric, "threshold": threshold,
                  "source": {"url": url, "title": title or (page or {}).get("title") or (snip or {}).get("title"),
                             "publisher": publisher, "published": published, "tier": g["tier"],
                             "source_type": g["source_type"], "host": g["host"],
                             "agent_reported_fields": [k for k, v in (("title", title), ("publisher", publisher),
                                                                      ("published", published)) if v],
                             "retrieval": {k: (page or snip or {}).get(k) for k in
                                           ("provider", "retrieved_at", "final_url", "discovered_via", "query")}},
                  "excerpt": excerpt, "applicability": applicability, "reasoning": reasoning,
                  "supports": supports or [], "contradicts": contradicts or [],
                  **{k: g[k] for k in ("admissibility", "gate_reasons", "retrieved_via", "excerpt_verified", "contested")}}
        return store.append_evidence(campaign_id, record)

    @mcp.tool()
    def fix_objective(campaign_id: str, metric: str, stat: Literal["min", "mean", "max"], threshold: float,
                      threshold_basis: str, search_lower: float, search_upper: float, rel_tolerance: float,
                      pinned_parameters: dict[str, Any] | None = None) -> dict[str, Any]:
        """Fix the objective and search once, before any experiment. threshold_basis is "demo"
        (recorded as an unsupported demo/user-supplied threshold) or an evidence id whose record
        is an admissible, uncontested numerical_threshold for this metric and value."""
        c, err = load(campaign_id)
        if err:
            return err
        if "." in metric and metric.rsplit(".", 1)[-1] in ("min", "mean", "max"):
            return _error("invalid_arguments", f"metric names are bare (e.g. {metric.rsplit('.', 1)[0]!r}); "
                          "give the statistic in stat")
        if threshold_basis == "demo":
            basis = DEMO_BASIS
        else:
            records = {e["id"]: e for e in store.evidence(campaign_id)}
            e = records.get(threshold_basis)
            contested = e is not None and (e["contested"] or any(threshold_basis in r["contradicts"]
                                                                 for r in records.values()))
            if (e is None or e["admissibility"] != "admissible" or e["claim_kind"] != "numerical_threshold"
                    or e["metric"] != metric or e["threshold"] != threshold or contested):
                return _error("invalid_arguments", "threshold_basis must be 'demo' or an admissible, "
                              "uncontested numerical_threshold evidence id for this metric and value")
            basis = f"evidence:{threshold_basis}"
        try:
            objective = Constraint(metric=metric, stat=stat, threshold=threshold)
            search = SearchSpace(lower=search_lower, upper=search_upper, rel_tolerance=rel_tolerance,
                                 fixed=pinned_parameters or {})
            c = store.set_objective(campaign_id, objective, basis, search)
        except ValidationError as e:
            return _error("invalid_arguments", str(e))
        except StoreError as e:
            return _error("invalid_state", str(e))
        return {"campaign": c.model_dump()}

    @mcp.tool()
    def record_interpretation(campaign_id: str, experiment_id: str, interpretation: str,
                              next_decision: str) -> dict[str, Any]:
        """Attach the agent's interpretation of a stored experiment. Labeled as agent text;
        it never alters the measured result."""
        c, err = load(campaign_id)
        if err:
            return err
        if experiment_id not in {e.experiment_id for e in store.experiments(campaign_id)}:
            return _error("not_found", f"no experiment {experiment_id}")
        return store.append_note(campaign_id, {"kind": "agent_interpretation", "experiment_id": experiment_id,
                                               "interpretation": interpretation, "next_decision": next_decision})

    return mcp


def main() -> None:
    store = Store(Path(os.environ.get("RESEARCH_LAB_STORE", "campaigns")))
    flowlab = Flowlab(os.environ.get("FLOWLAB_BASE_URL", "http://localhost:8000"))
    build_server(store, flowlab, float(os.environ.get("RESEARCH_LAB_POLL_TIMEOUT", "1200")),
                 provider_from_env()).run("stdio")


if __name__ == "__main__":
    main()
