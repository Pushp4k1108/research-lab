"""Research Lab web application: dashboard + campaign creation + live workspace.

Shows research state that already exists on disk -- no sample data, no fake
progress. Campaign creation uploads a STEP/STP geometry and starts the
EXISTING agent/MCP/Flowlab pipeline in a background thread; the workspace
polls the existing store files. Mesh inspection links out to Flowlab's own
viewer (FLOWLAB_PUBLIC_URL, default http://localhost:8000).

    python -m research_lab.ui [--store campaigns] [--port 8765]
"""

from __future__ import annotations

import argparse
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from research_lab.compare import summarize
from research_lab.planner import plan
from research_lab.store import Store, StoreError
from research_lab.templates import demo_brief, get_template, list_templates


HTML = Path(__file__).with_name("ui.html")


def _safe_jsonl(path: Path, tail: int | None = None) -> list[dict]:
    """Read a live-appended JSONL file; skip torn trailing lines, never 500."""
    if not path.exists():
        return []
    try:
        text = path.read_text()
    except OSError:
        return []
    lines = [line for line in text.splitlines() if line.strip()]
    if tail:
        lines = lines[-tail:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except ValueError:
            continue  # half-written line from the live agent; next poll gets it
    return out


def _jsonl(path: Path, tail: int | None = None) -> list[dict]:
    return _safe_jsonl(path, tail)


def campaigns(store: Store, runner=None) -> list[dict]:
    running = set(runner.running_campaigns()) if runner is not None else set()
    out = []
    for d in store.root.iterdir() if store.root.exists() else []:
        if not (d / "campaign.json").is_file():
            continue
        try:
            c = store.campaign(d.name)
        except ValueError:  # campaign file that fails today's validation: not shown
            continue
        run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
        n_exp = len(store.experiments(c.id))
        if runner is not None and runner.last_error(c.id) and run is None and c.id not in running:
            status = "error"
        elif run is not None:
            status = "finished"
        elif c.id in running or (d / "activity.jsonl").exists():
            status = "running"
        else:
            status = "created"
        latest = None
        exps = store.experiments(c.id)
        if exps:
            r = exps[-1].result
            latest = {"experiment_id": exps[-1].experiment_id, "outcome": r.outcome,
                      "element_count": r.element_count,
                      "minSICN_min": r.metrics.get("minSICN").min if "minSICN" in r.metrics else None,
                      "generation_id": r.generation_id}
        out.append({"id": c.id, "question": c.question, "created_at": c.created_at,
                    "template": c.template, "geometry_name": c.geometry_name,
                    "status": status, "running": c.id in running,
                    "experiments": n_exp, "max_experiments": c.max_experiments,
                    "finished": run is not None,
                    "stop_reason": run and run.get("stop_reason"),
                    "runner_error": runner.last_error(c.id) if runner is not None else None,
                    "latest": latest})
    return sorted(out, key=lambda c: c["created_at"], reverse=True)


def campaign_view(store: Store, cid: str, flowlab_url: str, runner=None) -> dict:
    c = store.campaign(cid)
    exps = store.experiments(cid)
    d = store.root / cid
    running = runner.is_running(cid) if runner is not None else False
    run = json.loads((d / "run.json").read_text()) if (d / "run.json").exists() else None
    return {
        "campaign": c.model_dump(),
        "experiments": [e.model_dump() for e in exps],
        "evidence": store.evidence(cid),
        "notes": store.notes(cid),
        "activity": _jsonl(d / "activity.jsonl", tail=300),
        "run": run,
        "running": running,
        "runner_error": runner.last_error(cid) if runner is not None else None,
        "report": (d / "report.md").read_text() if (d / "report.md").exists() else None,
        "next": plan(c, exps).model_dump(),  # deterministic planner, recomputed from stored state
        "summary": summarize(exps, c.objective),
        "flowlab_url": flowlab_url.rstrip("/"),
    }


def make_handler(store: Store, flowlab_url: str, runner=None,
                 upload_dir: Path | None = None):
    uploads = Path(upload_dir) if upload_dir is not None else store.root / "_uploads"
    uploads.mkdir(parents=True, exist_ok=True)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, code: int, body: bytes, ctype: str) -> None:
            self.send_response(code)
            self.send_header("Content-Type", ctype)
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, code: int, obj) -> None:
            self._send(code, json.dumps(obj, default=str).encode(), "application/json")

        def _read_body(self, limit: int = 12 * 1024 * 1024) -> bytes:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError:
                n = 0
            if n <= 0 or n > limit:
                raise ValueError("missing or oversized request body")
            return self.rfile.read(n)

        def do_GET(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path == "/":
                return self._send(200, HTML.read_bytes(), "text/html; charset=utf-8")
            if path == "/api/templates":
                return self._json(200, list_templates())
            if path == "/api/campaigns":
                return self._json(200, campaigns(store, runner))
            if path.startswith("/api/campaigns/"):
                cid = path.rsplit("/", 1)[1]
                if not cid.isalnum():
                    return self._json(400, {"error": "bad campaign id"})
                try:
                    return self._json(200, campaign_view(store, cid, flowlab_url, runner))
                except (StoreError, ValueError) as e:
                    return self._json(404, {"error": str(e)[:300]})
            self._json(404, {"error": "not found"})

        def do_POST(self):  # noqa: N802
            path = self.path.split("?")[0]
            if path != "/api/campaigns":
                return self._json(404, {"error": "not found"})
            try:
                fields, files = _parse_multipart(self.headers.get("Content-Type") or "",
                                                 self._read_body())
            except ValueError as e:
                return self._json(400, {"error": str(e)[:300]})
            try:
                created = create_campaign_from_form(store, runner, uploads, fields, files)
            except ValueError as e:
                return self._json(400, {"error": str(e)[:500]})
            return self._json(201, created)

        def log_message(self, *a):
            pass
    return Handler


def _parse_multipart(content_type: str, body: bytes) -> tuple[dict, dict]:
    """Minimal multipart/form-data parser (stdlib only, single file field)."""
    if "multipart/form-data" not in content_type or "boundary=" not in content_type:
        raise ValueError("expected multipart/form-data")
    boundary = content_type.split("boundary=", 1)[1].strip().strip('"').encode()
    if not boundary or len(body) > 12 * 1024 * 1024:
        raise ValueError("bad boundary or oversized body")
    fields: dict[str, str] = {}
    files: dict[str, dict] = {}
    for part in body.split(b"--" + boundary):
        if b"Content-Disposition" not in part:
            continue
        head, _, content = part.partition(b"\r\n\r\n")
        if head == part:
            head, _, content = part.partition(b"\n\n")
        content = content.removesuffix(b"\r\n").removesuffix(b"\n")
        if content.endswith(b"--"):
            content = content[:-2]
        name = _disposition_attr(head, "name")
        filename = _disposition_attr(head, "filename")
        if not name:
            continue
        if filename is not None:
            files[name] = {"filename": filename, "content": content}
        else:
            fields[name] = content.decode("utf-8", errors="replace").strip()
    return fields, files


def _disposition_attr(head: bytes, attr: str) -> str | None:
    try:
        text = head.decode("latin-1")
    except ValueError:
        return None
    import re
    m = re.search(attr + r'="([^"]*)"', text)
    return m.group(1) if m else None


def create_campaign_from_form(store: Store, runner, uploads: Path,
                              fields: dict, files: dict) -> dict:
    """Validate the Mesh Optimization form, store geometry, start the agent.

    Creates the campaign record synchronously (validation -> 400, nothing
    started), then launches the existing agent loop in the background.
    Raises ValueError with a user-facing message on validation failure.
    """
    from research_lab.store import SearchSpace

    template = get_template(fields.get("template") or "mesh_optimization")
    if template.id != "mesh_optimization":
        raise ValueError(f"template {template.id!r} is not implemented yet")
    question = (fields.get("question") or "").strip()
    if len(question) < 10:
        raise ValueError("research question is required (min 10 characters)")
    length_unit = (fields.get("length_unit") or template.default_length_unit).strip()
    if length_unit not in template.allowed_units:
        raise ValueError(f"length_unit must be one of {list(template.allowed_units)}")
    up_axis = (fields.get("up_axis") or template.default_up_axis).strip()
    if up_axis not in template.allowed_axes:
        raise ValueError(f"up_axis must be one of {list(template.allowed_axes)}")
    try:
        max_experiments = int(fields.get("max_experiments") or template.default_max_experiments)
    except ValueError:
        raise ValueError("experiment budget must be an integer") from None
    if not 1 <= max_experiments <= 5:
        raise ValueError("experiment budget must be 1..5")
    demo_objective = (fields.get("demo_objective") or template.default_demo_objective).strip()
    try:
        metric, stat, thr = demo_objective.split(":")
        threshold = float(thr)
        if stat not in ("min", "mean", "max") or not metric or "." in metric:
            raise ValueError
    except ValueError:
        raise ValueError("demo objective must be metric:stat:threshold, e.g. minSICN:min:0.005") from None
    try:
        lower = float(fields.get("search_lower") or template.default_search_lower)
        upper = float(fields.get("search_upper") or template.default_search_upper)
        tol = float(fields.get("rel_tolerance") or template.default_rel_tolerance)
    except ValueError:
        raise ValueError("search bounds/tolerance must be numbers") from None
    if not (upper > lower > 0) or not (0 < tol < 1):
        raise ValueError("need 0 < lower < upper and 0 < rel_tolerance < 1")
    upload = files.get("geometry")
    if upload is None or not upload.get("content"):
        raise ValueError("a STEP/STP geometry file is required")
    filename = Path(upload["filename"]).name
    ext = Path(filename).suffix.lower()
    if ext not in template.allowed_extensions:
        raise ValueError(f"geometry must be a STEP file ({list(template.allowed_extensions)})")
    if len(upload["content"]) > template.max_upload_bytes:
        raise ValueError("geometry file exceeds 10 MB")
    try:
        search = SearchSpace(lower=lower, upper=upper, rel_tolerance=tol)
    except ValueError as e:
        raise ValueError(f"invalid search space: {e}") from None
    campaign = store.create_campaign(question, max_experiments, template=template.id,
                                     geometry_name=filename, search=search)
    uploads.mkdir(parents=True, exist_ok=True)
    geometry_path = uploads / f"{campaign.id}{ext or '.step'}"
    geometry_path.write_bytes(upload["content"])
    if runner is None:
        raise ValueError("campaign created but no runner is configured; restart the UI")
    setup = {"geometry_path": str(geometry_path), "length_unit": length_unit, "up_axis": up_axis}
    brief = demo_brief(demo_objective, lower, upper, tol)
    try:
        status = runner.start(campaign.id, str(geometry_path), template, setup, brief)
    except ValueError as e:
        raise ValueError(str(e)) from None
    except Exception as e:
        raise ValueError(f"agent failed to start ({type(e).__name__}: {e})"[:300]) from None
    return {"campaign_id": campaign.id, "campaign": campaign.model_dump(),
            "running": status.running, "started_at": status.started_at}


def main() -> None:
    from research_lab.agent import load_dotenv
    from research_lab.runner import default_runner
    load_dotenv(Path(os.environ.get("RESEARCH_LAB_ENV", ".env")))
    p = argparse.ArgumentParser()
    p.add_argument("--store", default=os.environ.get("RESEARCH_LAB_STORE", "campaigns"))
    p.add_argument("--port", type=int, default=8765)
    p.add_argument("--flowlab-url", default=os.environ.get("FLOWLAB_PUBLIC_URL", "http://localhost:8000"))
    a = p.parse_args()
    store = Store(Path(a.store))
    runner = default_runner(store)
    server = ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(store, a.flowlab_url, runner))
    print(f"Research Lab UI: http://127.0.0.1:{a.port}  (store: {a.store})")
    server.serve_forever()



if __name__ == "__main__":
    main()
