"""UI data layer: reads only real stored state."""

import json
import threading
import time

from research_lab.results import Constraint, normalize_generation
from research_lab.store import SearchSpace, Store
from research_lab.ui import _parse_multipart, campaign_view, campaigns, create_campaign_from_form
from tests.test_store_compare import envelope


def _form(**over):
    fields = {"template": "mesh_optimization",
              "question": "How does mesh resolution affect mesh quality for this geometry?",
              "length_unit": "mm", "up_axis": "z", "max_experiments": "2",
              "demo_objective": "minSICN:min:0.005",
              "search_lower": "6", "search_upper": "12", "rel_tolerance": "0.1"}
    fields.update(over)
    return fields, {"geometry": {"filename": "cube.step", "content": b"fake-step-bytes"}}


class _Runner:
    def __init__(self):
        self.calls = []

    def running_campaigns(self):
        return []

    def is_running(self, cid):
        return False

    def last_error(self, cid):
        return None

    def start(self, cid, path, template, setup, brief):
        self.calls.append((cid, path, template, setup, brief))

        class S:
            running, started_at = True, "now"
        return S()


def test_campaign_view_from_store(tmp_path):
    s = Store(tmp_path)
    c = s.create_campaign("q", 5, Constraint(metric="minSICN", stat="min", threshold=0.005),
                          SearchSpace(lower=8, upper=12, rel_tolerance=0.05))
    s.append(c.id, {"target_element_size": 12}, normalize_generation(envelope("size12")), "h", "r", True)
    (tmp_path / c.id / "activity.jsonl").write_text(json.dumps({"kind": "planner", "decision": {}}) + "\n")
    v = campaign_view(s, c.id, "http://flowlab:8000/")
    assert v["flowlab_url"] == "http://flowlab:8000"
    assert v["experiments"][0]["result"]["generation_id"] and v["experiments"][0]["flowlab_existing"] is True
    assert v["next"]["status"] == "propose" and v["next"]["next_parameters"] == {"target_element_size": 8.0}
    assert v["activity"][0]["kind"] == "planner" and v["report"] is None and v["run"] is None
    assert campaigns(s)[0]["experiments"] == 1


def test_invalid_campaign_skipped(tmp_path):
    (tmp_path / "bad").mkdir()
    (tmp_path / "bad" / "campaign.json").write_text('{"id": "bad"}')
    assert campaigns(Store(tmp_path)) == []


def test_templates_registry():
    from research_lab.templates import demo_brief, get_template, list_templates
    assert [t["id"] for t in list_templates()] == ["mesh_optimization"]
    assert "minSICN" in demo_brief("minSICN:min:0.005", 6, 12, 0.1)
    try:
        get_template("cfd")
    except ValueError as e:
        assert "unknown template" in str(e)
    else:
        raise AssertionError("unknown template accepted")


def test_create_campaign_happy_path(tmp_path):
    s = Store(tmp_path)
    r = _Runner()
    out = create_campaign_from_form(s, r, tmp_path / "up", *_form())
    cid = out["campaign_id"]
    assert r.calls and r.calls[0][0] == cid
    assert r.calls[0][2].id == "mesh_optimization"
    assert (tmp_path / "up" / f"{cid}.step").read_bytes() == b"fake-step-bytes"
    c = s.campaign(cid)  # arbitrary geometry, not Turbine.stp
    assert c.template == "mesh_optimization" and c.geometry_name == "cube.step"
    assert c.search.lower == 6 and c.max_experiments == 2
    # Dashboard + workspace reflect real state, no fabrication.
    row = [x for x in campaigns(s, r) if x["id"] == cid][0]
    assert row["status"] in ("created", "running") and row["geometry_name"] == "cube.step"
    assert campaigns(s, r)[0]["max_experiments"] == 2
    v = campaign_view(s, cid, "http://flowlab:8000/", r)
    assert v["campaign"]["template"] == "mesh_optimization" and v["experiments"] == []
    assert v["running"] is False and v["report"] is None


def test_create_campaign_validation(tmp_path):
    s = Store(tmp_path)
    for kw, msg in [({"question": "short"}, "research question"),
                    ({"length_unit": "furlong"}, "length_unit"),
                    ({"max_experiments": "9"}, "budget"),
                    ({"demo_objective": "nonsense"}, "objective"),
                    ({"search_lower": "12", "search_upper": "6"}, "lower")]:
        try:
            create_campaign_from_form(s, _Runner(), tmp_path / "up", *_form(**kw))
        except ValueError as e:
            assert msg in str(e), (kw, e)
        else:
            raise AssertionError(f"accepted invalid form: {kw}")
    try:  # wrong file type is never uploaded to Flowlab
        f, files = _form()
        files["geometry"]["filename"] = "part.stl"
        create_campaign_from_form(s, _Runner(), tmp_path / "up", f, files)
    except ValueError as e:
        assert "STEP" in str(e)
    else:
        raise AssertionError("accepted non-STEP geometry")
    try:  # only mesh_optimization exists
        create_campaign_from_form(s, _Runner(), tmp_path / "up", *_form(template="cfd"))
    except ValueError as e:
        assert "unknown template" in str(e)
    else:
        raise AssertionError("accepted unknown template")


def test_multipart_parser_roundtrip():
    b = b"--B\r\nContent-Disposition: form-data; name=\"question\"\r\n\r\nhello\r\n--B\r\nContent-Disposition: form-data; name=\"geometry\"; filename=\"cube.stp\"\r\n\r\nBYTES\r\n--B--\r\n"
    fields, files = _parse_multipart("multipart/form-data; boundary=B", b)
    assert fields == {"question": "hello"} and files["geometry"]["filename"] == "cube.stp"
    assert files["geometry"]["content"] == b"BYTES"


def test_rendered_form_options_and_defaults():
    """Regression: the served form has usable options/defaults (works pre-JS)."""
    import re
    from research_lab.templates import get_template, list_templates
    from research_lab.ui import HTML

    (tpl,) = list_templates()
    assert tpl["id"] == "mesh_optimization" and tpl["label"] == "Mesh Optimization"
    t = get_template("mesh_optimization")
    assert set(tpl["allowed"]["units"]) == {"m", "mm", "cm", "in", "ft"}
    assert set(tpl["allowed"]["axes"]) == {"x", "y", "z"}
    assert t.default_length_unit == "mm" and t.default_up_axis == "z"
    html = HTML.read_text()
    for sel, options, default in (("tpl", ["mesh_optimization"], "mesh_optimization"),
                                  ("flu", ["m", "mm", "cm", "in", "ft"], "mm"),
                                  ("faxis", ["x", "y", "z"], "z")):
        m = re.search(rf'<select id="{sel}"[^>]*>(.*?)</select>', html, re.S)
        assert m, f"missing select #{sel}"
        for opt in options:
            assert re.search(rf'<option value="{opt}"[^>]*>', m.group(1)), f"#{sel} missing {opt}"
        dm = re.search(r'<option value="([^"]+)" selected>', m.group(1))
        assert dm and dm.group(1) == default, f"#{sel} default is not {default}"
    for fid, default in (("fq", t.default_question), ("fobj", t.default_demo_objective),
                         ("flo", "6"), ("fhi", "12"), ("ftol", "0.1"), ("fbudget", "5")):
        m = re.search(rf'id="{fid}"[^>]*(?:value="([^"]*)"|>(.*?)</textarea>)', html, re.S)
        assert m, f"missing field #{fid}"
        assert (m.group(1) if m.group(1) is not None else m.group(2)) == default, f"#{fid} default"
    assert "syncTemplateForm(templates)" in html and "$(\"tpl\").onchange" in html


def test_post_campaign_real_stp_file(tmp_path):
    """End-to-end POST /api/campaigns multipart with a REAL .stp file."""
    import threading
    import urllib.request
    from http.server import ThreadingHTTPServer
    from pathlib import Path
    from research_lab.templates import get_template
    from research_lab.ui import make_handler

    t = get_template("mesh_optimization")
    s, started = Store(tmp_path), {}

    class StubRunner:
        def running_campaigns(self):
            return []

        def is_running(self, cid):
            return False

        def last_error(self, cid):
            return None

        def start(self, cid, path, template, setup, brief):
            started.update(cid=cid, path=path, tid=template.id, setup=setup, brief=brief)

            class S:
                running, started_at = True, "now"
            return S()

    srv = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(s, "http://flowlab:8000",
                                                             StubRunner(), tmp_path / "up"))
    th = threading.Thread(target=srv.serve_forever, daemon=True)
    th.start()
    try:
        port = srv.server_address[1]
        assert urllib.request.urlopen(f"http://127.0.0.1:{port}/api/templates", timeout=5).status == 200
        payload = (Path(__file__).resolve().parents[1] / "fixtures" / "Turbine.stp").read_bytes()[:50000]
        assert payload
        fields = {"template": "mesh_optimization", "question": t.default_question,
                  "length_unit": t.default_length_unit, "up_axis": t.default_up_axis,
                  "max_experiments": str(t.default_max_experiments),
                  "demo_objective": t.default_demo_objective,
                  "search_lower": str(t.default_search_lower),
                  "search_upper": str(t.default_search_upper),
                  "rel_tolerance": str(t.default_rel_tolerance)}
        B = "BOUNDARY123"

        def part(name, val, filename=None):
            head = f'--{B}\r\nContent-Disposition: form-data; name="{name}"'
            if filename:
                head += f'; filename="{filename}"\r\nContent-Type: application/octet-stream'
            return head.encode() + b"\r\n\r\n" + val + b"\r\n"

        body = b"".join(part(k, str(v).encode()) for k, v in fields.items())
        body += part("geometry", payload, "cube.stp") + f"--{B}--\r\n".encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}/api/campaigns", data=body,
                                     headers={"Content-Type": f"multipart/form-data; boundary={B}"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            assert resp.status == 201
            cid = json.loads(resp.read())["campaign_id"]
        assert started["cid"] == cid and started["tid"] == "mesh_optimization"
        assert Path(started["path"]).read_bytes() == payload
        assert started["setup"] == {"geometry_path": started["path"],
                                    "length_unit": "mm", "up_axis": "z"}
        c = s.campaign(cid)
        assert c.template == "mesh_optimization" and c.geometry_name == "cube.stp"
        assert (c.search.lower, c.search.upper, c.search.rel_tolerance) == (6.0, 12.0, 0.1)
    finally:
        srv.shutdown()
        th.join(timeout=5)



def test_runner_runs_existing_pipeline(tmp_path):
    """Runner.start launches a background thread; research_existing is the loop entry."""
    import threading
    import time
    from research_lab.runner import Runner
    from research_lab.templates import get_template

    s = Store(tmp_path)
    c = s.create_campaign("q", 2, template="mesh_optimization", geometry_name="cube.step")
    geo = tmp_path / "cube.step"
    geo.write_bytes(b"step")
    started = threading.Event()

    class FastRunner(Runner):
        def _research(self, campaign_id, geometry_path, template, setup, brief, max_llm_turns):
            assert setup["geometry_path"] == geometry_path and template.id == "mesh_optimization"
            (self.store.root / campaign_id / "activity.jsonl").write_text(
                json.dumps({"kind": "geometry_ready", "source": "cube.step"}) + "\n")
            started.set()
    fr = FastRunner(store=s, flowlab_base_url="http://x")
    st = fr.start(c.id, str(geo), get_template("mesh_optimization"),
                  {"geometry_path": str(geo), "length_unit": "mm", "up_axis": "z"}, "brief")
    assert st.running and fr.is_running(c.id)
    try:
        fr.start(c.id, str(geo), get_template("mesh_optimization"),
                 {"geometry_path": str(geo), "length_unit": "mm", "up_axis": "z"}, "brief")
    except ValueError as e:
        assert "already running" in str(e)
    else:
        raise AssertionError("double start allowed")
    assert started.wait(timeout=5)
    deadline = time.time() + 5
    while fr.is_running(c.id) and time.time() < deadline:
        time.sleep(0.01)
    assert campaigns(s, fr)[0]["status"] == "running"  # activity exists, run.json not yet
    assert campaign_view(s, c.id, "http://f/", fr)["activity"][0]["kind"] == "geometry_ready"


def test_research_existing_reuses_loop(tmp_path):
    """Controller.research_existing runs the full loop on a pre-created campaign."""
    import asyncio
    import httpx
    from mcp.client import Client
    from research_lab.agent import Controller, Limits
    from research_lab.flowlab import Flowlab
    from research_lab.mcp_server import build_server
    from tests.test_agent import ScriptedLLM, text, tool, OBJ, RUN12, SETUP
    from tests.test_mcp_server import FakeFlowlab

    s = Store(tmp_path)
    c = s.create_campaign("q", 2, template="mesh_optimization")
    geo = tmp_path / "cube.step"
    geo.write_bytes(b"step")

    async def go():
        fake = FakeFlowlab()
        fl = Flowlab(transport=httpx.MockTransport(fake.handler))
        llm = ScriptedLLM([[tool("fix_objective", **OBJ)], [tool("run_experiment", **RUN12)],
                            [text("done")]])
        async with Client(build_server(s, fl, 1200.0, None)) as session:
            ctl = Controller(session, llm, Limits(5, 60, 2))
            run = await ctl.research_existing(c.id, "q", {**SETUP, "geometry_path": str(geo)},
                                              "brief")
            assert run.campaign_id == c.id and run.stop_reason == "agent_finished"
            assert len(s.experiments(c.id)) == 1
    asyncio.run(go())
