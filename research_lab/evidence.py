"""Deterministic evidence gate: source tier, excerpt verification, admissibility.

The LLM proposes a claim, its kind, an excerpt and its applicability. Code decides:
  * which tier the source belongs to (by host, never by the agent's say-so);
  * whether the excerpt actually occurs in what was retrieved (fetched page, else snippet);
  * whether a numerical threshold is literally stated in the excerpt;
  * admissible / lead_only / rejected.
Search snippets are leads, never admissible on their own.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any, Literal
from urllib.parse import urlparse

ClaimKind = Literal["metric_relevance", "metric_definition", "numerical_threshold", "other"]

# (tier, source_type). Tier 1 best. Hosts match exactly or as a parent domain.
TIERS: list[tuple[int, str, tuple[str, ...]]] = [
    (1, "peer_reviewed_or_standard", ("doi.org", "sciencedirect.com", "link.springer.com", "springer.com",
                                      "ieeexplore.ieee.org", "dl.acm.org", "onlinelibrary.wiley.com",
                                      "tandfonline.com", "iso.org", "astm.org", "sae.org", "aiaa.org",
                                      "arc.aiaa.org", "asmedigitalcollection.asme.org", "mdpi.com")),
    (1, "official_technical_documentation", ("gmsh.info",)),
    (2, "university_government_research", ("nasa.gov", "nist.gov", "sandia.gov", "osti.gov", "llnl.gov",
                                           "cern.ch", "arxiv.org", "hal.science", "researchgate.net")),
    (3, "engineering_organization", ("nafems.org", "asme.org", "ansys.com", "comsol.com", "siemens.com",
                                     "sw.siemens.com", "cadence.com", "pointwise.com", "simscale.com",
                                     "openfoam.org", "openfoam.com", "salome-platform.org")),
    (5, "forum", ("stackexchange.com", "stackoverflow.com", "reddit.com", "quora.com")),
    (4, "blog_or_practitioner", ("medium.com", "wordpress.com", "blogspot.com", "substack.com",
                                 "github.com", "github.io", "wikipedia.org")),
]
FORUM_PATH = re.compile(r"/(forums?|threads?|community)/", re.I)


def classify(url: str) -> tuple[int, str, str]:
    """(tier, source_type, host). Unknown hosts are tier 5 'unclassified'."""
    parsed = urlparse(url)
    host = (parsed.hostname or "").lower().removeprefix("www.")
    for tier, kind, hosts in TIERS:
        if any(host == h or host.endswith("." + h) for h in hosts):
            return tier, kind, host
    if FORUM_PATH.search(parsed.path):
        return 5, "forum", host
    if re.search(r"\.(edu|ac\.[a-z]{2}|edu\.[a-z]{2})$", host) or host.endswith(".gov") or ".gov." in host:
        return 2, "university_government_research", host
    return 5, "unclassified", host


def _norm(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).lower()
    s = s.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                   "–": "-", "—": "-", "−": "-"}))
    return re.sub(r"\s+", " ", s).strip()


def contains(haystack: str, excerpt: str) -> bool:
    return bool(excerpt.strip()) and _norm(excerpt) in _norm(haystack)


def numbers_in(text: str) -> set[float]:
    return {float(m) for m in re.findall(r"(?<![\w.])-?\d+(?:\.\d+)?(?:[eE]-?\d+)?", _norm(text))}


def gate(*, claim_kind: str, url: str, excerpt: str, page: dict | None, snippet: str | None,
         metric: str | None, threshold: float | None, existing: dict[str, dict],
         supports: list[str], contradicts: list[str]) -> dict[str, Any]:
    """Return the deterministic gate fields of an evidence record."""
    tier, source_type, host = classify(url)
    reasons: list[str] = []
    out = {"tier": tier, "source_type": source_type, "host": host, "contested": False}

    unknown = [i for i in supports + contradicts if i not in existing]
    if unknown:
        return {**out, "admissibility": "rejected", "retrieved_via": None, "excerpt_verified": False,
                "gate_reasons": [f"unknown evidence ids: {unknown}"]}

    if page is not None and page.get("text"):
        via, verified = "fetched_page", contains(page["text"], excerpt)
    elif snippet is not None:
        via, verified = "search_snippet", contains(snippet, excerpt)
    else:
        return {**out, "admissibility": "rejected", "retrieved_via": None, "excerpt_verified": False,
                "gate_reasons": ["source was not retrieved in this campaign"]}
    out.update(retrieved_via=via, excerpt_verified=verified)
    if not verified:
        return {**out, "admissibility": "rejected", "gate_reasons": [f"excerpt not found in {via}"]}

    admissibility = "admissible"
    if via == "search_snippet":
        admissibility = "lead_only"
        reasons.append("search snippet only; fetch the source to verify")
    if tier >= 4:
        admissibility = "lead_only"
        reasons.append(f"tier {tier} ({source_type}) is a research lead, not supporting evidence")
    if claim_kind == "numerical_threshold":
        if metric is None or threshold is None:
            return {**out, "admissibility": "rejected", "gate_reasons": ["threshold claim needs metric and threshold"]}
        if threshold not in numbers_in(excerpt):
            return {**out, "admissibility": "rejected",
                    "gate_reasons": [f"threshold {threshold:g} is not stated in the excerpt; it cannot be inferred"]}
    contested_by = [i for i in contradicts if existing[i].get("admissibility") == "admissible"]
    if contested_by:
        out["contested"] = True
        reasons.append(f"contradicts admissible evidence {contested_by}")
    return {**out, "admissibility": admissibility, "gate_reasons": reasons or ["passed"]}
