"""Research templates: configuration concept over the existing meshing workflow.

Only ``mesh_optimization`` is implemented. The registry is deliberately
data-driven so later templates can be added without rewriting UI/backend:

    TEMPLATES = {"mesh_optimization": MeshOptimization, ...}

A template describes the creation-form defaults and how to turn those form
values into the EXISTING backend contracts (Store campaign + MCP
setup_geometry + agent brief). No planner/evidence/LLM behaviour lives here.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Template:
    id: str
    label: str
    description: str
    # Form defaults for the existing backend contracts.
    default_question: str
    default_length_unit: str = "mm"
    default_up_axis: str = "z"
    default_max_experiments: int = 5
    default_demo_objective: str = "minSICN:min:0.005"
    default_search_lower: float = 6.0
    default_search_upper: float = 12.0
    default_rel_tolerance: float = 0.1
    allowed_units: tuple[str, ...] = ("m", "mm", "cm", "in", "ft")
    allowed_axes: tuple[str, ...] = ("x", "y", "z")
    allowed_extensions: tuple[str, ...] = (".step", ".stp")
    max_upload_bytes: int = 10 * 1024 * 1024


MESH_OPTIMIZATION = Template(
    id="mesh_optimization",
    label="Mesh Optimization",
    description=(
        "How does mesh resolution affect mesh quality for this geometry, and what "
        "mesh resolution provides an appropriate trade-off between mesh quality "
        "and element count?"
    ),
    default_question=(
        "How does mesh resolution affect mesh quality for this geometry, and what "
        "mesh resolution provides an appropriate trade-off between mesh quality "
        "and element count?"
    ),
)

TEMPLATES: dict[str, Template] = {MESH_OPTIMIZATION.id: MESH_OPTIMIZATION}


def get_template(template_id: str) -> Template:
    try:
        return TEMPLATES[template_id]
    except KeyError:
        raise ValueError(f"unknown template {template_id!r} (available: {sorted(TEMPLATES)})") from None


def list_templates() -> list[dict[str, Any]]:
    return [
        {"id": t.id, "label": t.label, "description": t.description,
         "defaults": {"question": t.default_question, "length_unit": t.default_length_unit,
                       "up_axis": t.default_up_axis, "max_experiments": t.default_max_experiments,
                       "demo_objective": t.default_demo_objective,
                       "search_lower": t.default_search_lower, "search_upper": t.default_search_upper,
                       "rel_tolerance": t.default_rel_tolerance},
         "allowed": {"units": list(t.allowed_units), "axes": list(t.allowed_axes),
                     "extensions": list(t.allowed_extensions)},
         "max_upload_bytes": t.max_upload_bytes}
        for t in TEMPLATES.values()
    ]


def demo_brief(demo_objective: str, lower: float, upper: float, rel_tolerance: float) -> str:
    """User-supplied demo objective brief for the agent (never presented as sourced)."""
    metric, stat, thr = demo_objective.split(":")
    return (
        f'User-supplied demo objective (not evidence): fewest elements subject to metric="{metric}", '
        f'stat="{stat}", threshold={thr}; search target_element_size in [{lower}, {upper}] with '
        f'rel_tolerance {rel_tolerance}; no pinned_parameters. Call fix_objective with exactly these values and '
        f'threshold_basis="demo" unless admissible evidence states a threshold for this metric.'
    )
