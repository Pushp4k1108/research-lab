"""Real Flowlab smoke test. Two stages so the mesh size is chosen from measured geometry.

    python scripts/smoke_test.py setup
    python scripts/smoke_test.py mesh --size <target_element_size>

Every raw response is saved under fixtures/smoke/ for later tests.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from research_lab.flowlab import Flowlab, FlowlabError

ROOT = Path(__file__).resolve().parents[1]
GEOMETRY = ROOT / "fixtures" / "Turbine.stp"
OUT = ROOT / "fixtures" / "smoke"


def save(name: str, data) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"{name}.json").write_text(json.dumps(data, indent=2, sort_keys=True))


def load(name: str):
    return json.loads((OUT / f"{name}.json").read_text())


def timed(fn, *a, **kw):
    t = time.monotonic()
    out = fn(*a, **kw)
    return out, round(time.monotonic() - t, 2)


def setup(fl: Flowlab) -> None:
    geometry, t_upload = timed(fl.upload_geometry, GEOMETRY)
    save("01_geometry", geometry)
    draft = {
        # The STEP header declares SI_UNIT(.MILLI.,.METRE.).
        "units": {"length": "mm"},
        "coord_system": {"up_axis": "z", "handedness": "right"},
        "geometry_ref": geometry["geometry_ref"],
        "boundary_groups": [],
        "intent_params": {},
    }
    model, t_model = timed(fl.create_model, draft)
    save("02_model_v1", model)
    try:
        (code, inventory), t_inv = timed(fl.derive_inventory, model["model_id"])
    except FlowlabError as e:
        save("03_inventory_error", {"status": e.status, "detail": e.detail})
        print(f"inventory refused: {e}")
        return
    save("03_inventory", {"http_status": code, "envelope": inventory})
    save("timing_setup", {"upload_s": t_upload, "model_s": t_model, "inventory_s": t_inv})
    print(json.dumps({"geometry": geometry, "model_id": model["model_id"],
                      "inventory_http": code, "version": inventory["version"],
                      "snapshot_hash": inventory["snapshot_hash"],
                      "provenance": inventory["model"].get("geometry_provenance"),
                      "timing": [t_upload, t_model, t_inv]}, indent=2))


def mesh(fl: Flowlab, size: float) -> None:
    model = load("03_inventory")["envelope"]
    configuration = {"target_element_size": size}
    t0 = time.monotonic()
    try:
        code, body = fl.request_generation(model["model_id"], configuration, model["version"])
    except FlowlabError as e:
        save("04_generation_request_error", {"status": e.status, "detail": e.detail})
        print(f"generation refused: {e}")
        return
    save("04_generation_request", {"http_status": code, "body": body})
    if code == 202:
        envelope, polls = fl.await_generation(body["input_hash"])
    else:
        envelope, polls = body, 0
    t_total = round(time.monotonic() - t0, 2)
    full = fl.get_generation(envelope["generation_id"])
    save("05_generation", full)
    save("timing_mesh", {"request_to_result_s": t_total, "polls": polls, "http_status": code})
    g = full["generation"]
    q = g.get("quality_report") or {}
    print(json.dumps({"http": code, "input_hash": g["input_hash"], "generation_id": full["generation_id"],
                      "status": full["status"], "polls": polls, "seconds": t_total,
                      "error": g.get("error"),
                      "quality": {k: q.get(k) for k in ("element_count", "node_count", "dimension",
                                                         "bounding_box", "metrics", "unmeasured",
                                                         "findings", "measured_by")}}, indent=2))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("stage", choices=["setup", "mesh"])
    p.add_argument("--size", type=float)
    p.add_argument("--base-url", default="http://localhost:8000")
    a = p.parse_args()
    fl = Flowlab(a.base_url)
    setup(fl) if a.stage == "setup" else mesh(fl, a.size)
