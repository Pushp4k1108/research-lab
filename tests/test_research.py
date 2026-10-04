"""Research provider tests: interface contract, DirectWeb, Bright Data, selection, gate parity. Mocked HTTP only."""

import ast
import asyncio
import inspect
import json
from pathlib import Path

import httpx
import pytest
from mcp.client import Client

from research_lab import evidence as ev
from research_lab.brightdata import BrightData, BrightDataError
from research_lab.flowlab import Flowlab
from research_lab.mcp_server import build_server
from research_lab.research import (AccessRestricted, DirectWeb, NotConfigured, ResearchError, SearchUnsupported,
                                   Unconfigured, UnsupportedContent, provider_from_env)
from research_lab.store import Store

PKG = Path(__file__).resolve().parents[1] / "research_lab"
URL = "https://gmsh.info/doc/texinfo/gmsh.html"
HTML = ("<html><head><title>Gmsh manual</title><script>x()</script></head><body><nav>menu</nav>"
        "<p>The signed inverse condition number (SICN) is a quality measure; values near 1 indicate "
        "well-shaped elements.</p></body></html>")
EXCERPT = "values near 1 indicate well-shaped elements"
PAGE_KEYS = {"url", "final_url", "title", "text", "status", "content_type", "provider", "retrieved_at"}
ALLOW_ALL = "User-agent: *\nAllow: /\n"


def direct(routes, **kw):
    """DirectWeb over a mock: routes maps path -> response or callable(request)."""
    def handler(request):
        r = routes.get(request.url.path, httpx.Response(404))
        return r(request) if callable(r) else r
    return DirectWeb(transport=httpx.MockTransport(handler), **kw)


def ok_routes(body=HTML, ctype="text/html; charset=utf-8"):
    return {"/robots.txt": httpx.Response(200, text=ALLOW_ALL),
            "/doc/texinfo/gmsh.html": httpx.Response(200, text=body, headers={"content-type": ctype})}


def brightdata(handler):
    return BrightData("tok", "serp_zone", "unlocker_zone", transport=httpx.MockTransport(handler))


def bd_handler(request):
    body = json.loads(request.content)
    assert request.headers["authorization"] == "Bearer tok"
    if body["zone"] == "serp_zone":
        assert "brd_json=1" in body["url"]
        return httpx.Response(200, text=json.dumps({"organic": [
            {"title": "Gmsh manual", "link": URL, "description": "SICN quality measure"},
            {"title": "no link"}]}))
    return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})


# --- contract --------------------------------------------------------------------------

@pytest.mark.parametrize("provider", [DirectWeb(), BrightData("t", "s", "u"), Unconfigured("x", "r")])
def test_providers_share_interface(provider):
    assert isinstance(provider.name, str)
    assert list(inspect.signature(provider.search).parameters) == ["query", "n"]
    assert list(inspect.signature(provider.fetch).parameters) == ["url"]


def test_brightdata_errors_are_research_errors():
    assert issubclass(BrightDataError, ResearchError)


# --- DirectWeb -------------------------------------------------------------------------

def test_direct_fetch_normalizes():
    p = direct(ok_routes()).fetch(URL)
    assert set(p) == PAGE_KEYS and p["provider"] == "direct" and p["status"] == "ok"
    assert p["title"] == "Gmsh manual" and EXCERPT in p["text"]
    assert "x()" not in p["text"] and "menu" not in p["text"]  # script/nav stripped


def test_direct_plain_text():
    p = direct(ok_routes("plain   text\nbody", "text/plain")).fetch(URL)
    assert p["text"] == "plain text body" and p["content_type"] == "text/plain"


def test_direct_has_no_search():
    with pytest.raises(SearchUnsupported):
        DirectWeb().search("anything")


def test_direct_timeout():
    def slow(request):
        raise httpx.ReadTimeout("slow", request=request)
    with pytest.raises(ResearchError, match="timeout"):
        direct({**ok_routes(), "/doc/texinfo/gmsh.html": slow}).fetch(URL)


def test_direct_transport_error():
    def down(request):
        raise httpx.ConnectError("refused", request=request)
    with pytest.raises(ResearchError, match="transport"):
        direct({**ok_routes(), "/doc/texinfo/gmsh.html": down}).fetch(URL)


@pytest.mark.parametrize("status,exc", [(403, AccessRestricted), (401, AccessRestricted), (429, AccessRestricted),
                                        (451, AccessRestricted), (404, ResearchError), (500, ResearchError)])
def test_direct_http_errors(status, exc):
    with pytest.raises(exc):
        direct({**ok_routes(), "/doc/texinfo/gmsh.html": httpx.Response(status)}).fetch(URL)


@pytest.mark.parametrize("body,ctype", [("%PDF-1.7 ...", "application/pdf"), ("%PDF-1.7", "application/octet-stream"),
                                        ("\x89PNG", "image/png"), ("{}", "application/json")])
def test_direct_unsupported_content(body, ctype):
    with pytest.raises(UnsupportedContent):
        direct(ok_routes(body, ctype)).fetch(URL)


def test_direct_size_cap():
    with pytest.raises(UnsupportedContent, match="exceeds"):
        direct(ok_routes("<html>" + "a" * 5000 + "</html>"), max_bytes=1000).fetch(URL)


@pytest.mark.parametrize("url", ["ftp://gmsh.info/x", "file:///etc/passwd", "gmsh.info/doc"])
def test_direct_rejects_non_http(url):
    with pytest.raises(UnsupportedContent):
        DirectWeb().fetch(url)


def test_robots_disallow_is_respected():
    routes = {**ok_routes(), "/robots.txt": httpx.Response(200, text="User-agent: *\nDisallow: /doc/\n")}
    with pytest.raises(AccessRestricted, match="robots"):
        direct(routes).fetch(URL)


def test_robots_unreachable_disallows_and_missing_allows():
    with pytest.raises(AccessRestricted):
        direct({**ok_routes(), "/robots.txt": httpx.Response(503)}).fetch(URL)
    assert direct({**ok_routes(), "/robots.txt": httpx.Response(404)}).fetch(URL)["status"] == "ok"


def test_redirect_final_url_recorded():
    routes = {**ok_routes(), "/old": httpx.Response(301, headers={"location": URL})}
    p = direct(routes).fetch("https://gmsh.info/old")
    assert p["url"] == "https://gmsh.info/old" and p["final_url"] == URL


# --- Bright Data (mocked; the real service is never called) ----------------------------

def test_brightdata_search_and_fetch_normalized():
    bd = brightdata(bd_handler)
    hits = bd.search("gmsh sicn")
    assert hits == [{"title": "Gmsh manual", "url": URL, "snippet": "SICN quality measure", "provider": "brightdata"}]
    p = bd.fetch(URL)
    assert set(p) == PAGE_KEYS and p["provider"] == "brightdata" and EXCERPT in p["text"]


def test_brightdata_errors():
    with pytest.raises(BrightDataError, match="401"):
        brightdata(lambda r: httpx.Response(401, text="bad token")).search("q")
    with pytest.raises(BrightDataError, match="not parsed JSON"):
        brightdata(lambda r: httpx.Response(200, text="<html>")).search("q")
    with pytest.raises(UnsupportedContent):
        brightdata(lambda r: httpx.Response(200, text="%PDF-1.4")).fetch(URL)


# --- parity: the gate sees the same structure from either provider ---------------------

def gate_for(p):
    return ev.gate(claim_kind="metric_definition", url=p["final_url"], excerpt=EXCERPT, page=p, snippet=None,
                   metric=None, threshold=None, existing={}, supports=[], contradicts=[])


def test_gate_identical_across_providers():
    a, b = direct(ok_routes()).fetch(URL), brightdata(bd_handler).fetch(URL)
    strip = lambda p: {k: v for k, v in p.items() if k not in ("provider", "retrieved_at")}
    assert strip(a) == strip(b)
    assert gate_for(a) == gate_for(b)
    assert gate_for(a)["admissibility"] == "admissible"


# --- selection -------------------------------------------------------------------------

def test_provider_selection():
    full = {"BRIGHTDATA_API_TOKEN": "t", "BRIGHTDATA_SERP_ZONE": "s", "BRIGHTDATA_UNLOCKER_ZONE": "u"}
    assert isinstance(provider_from_env({"RESEARCH_PROVIDER": "direct", **full}), DirectWeb)  # no silent switch
    assert isinstance(provider_from_env({"RESEARCH_PROVIDER": "brightdata", **full}), BrightData)
    assert isinstance(provider_from_env(full), BrightData)  # default
    assert provider_from_env({"RESEARCH_PROVIDER": "none"}) is None
    with pytest.raises(ValueError):
        provider_from_env({"RESEARCH_PROVIDER": "google"})


def test_missing_brightdata_config_is_not_configured():
    p = provider_from_env({"RESEARCH_PROVIDER": "brightdata", "BRIGHTDATA_API_TOKEN": "t"})
    assert isinstance(p, Unconfigured)
    with pytest.raises(NotConfigured, match="BRIGHTDATA_SERP_ZONE"):
        p.search("q")
    with pytest.raises(NotConfigured):
        p.fetch(URL)


# --- through MCP -----------------------------------------------------------------------

def call_tools(tmp_path, provider, calls):
    server = build_server(Store(tmp_path), Flowlab(), 0.05, provider)

    async def go():
        async with Client(server) as c:
            cid = (await c.call_tool("create_campaign", {"question": "q", "max_experiments": 1})).structured_content["campaign_id"]
            return [(await c.call_tool(n, {"campaign_id": cid, **a})).structured_content for n, a in calls]
    return asyncio.run(go())


def test_mcp_unconfigured_brightdata(tmp_path):
    p = provider_from_env({"RESEARCH_PROVIDER": "brightdata"})
    s, f = call_tools(tmp_path, p, [("search_evidence", {"query": "q"}), ("fetch_source", {"url": URL})])
    assert s["error"]["kind"] == "not_configured" and s["error"]["provider"] == "brightdata"
    assert f["error"]["kind"] == "not_configured"


def test_mcp_direct_fetch_to_admissible_evidence(tmp_path):
    s, f, r = call_tools(tmp_path, direct(ok_routes()), [
        ("search_evidence", {"query": "q"}),
        ("fetch_source", {"url": URL, "keywords": ["SICN"]}),
        ("record_evidence", {"claim": "SICN near 1 means well-shaped", "claim_kind": "metric_definition",
                             "url": URL, "excerpt": EXCERPT, "applicability": "Flowlab reports minSICN",
                             "reasoning": "official docs"})])
    assert s["error"]["kind"] == "unsupported"
    assert f["tier"] == 1 and any("SICN" in p for p in f["passages"])
    assert r["admissibility"] == "admissible" and r["retrieved_via"] == "fetched_page"
    assert r["source"]["retrieval"]["provider"] == "direct"
    assert r["source"]["retrieval"]["discovered_via"] == "agent_supplied_url"


def test_mcp_direct_errors_are_not_evidence(tmp_path):
    routes = {**ok_routes(), "/doc/texinfo/gmsh.html": httpx.Response(403)}
    f, r = call_tools(tmp_path, direct(routes), [
        ("fetch_source", {"url": URL}),
        ("record_evidence", {"claim": "c", "claim_kind": "metric_definition", "url": URL, "excerpt": EXCERPT,
                             "applicability": "a", "reasoning": "r"})])
    assert f["error"]["kind"] == "access_restricted"
    assert r["admissibility"] == "rejected"  # nothing was retrieved


def test_redirect_to_unlisted_host_is_not_tier_one(tmp_path):
    def handler(request):
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text=ALLOW_ALL)
        if request.url.host == "doi.org":
            return httpx.Response(302, headers={"location": "https://random-site.example/x"})
        return httpx.Response(200, text=HTML, headers={"content-type": "text/html"})
    _, r = call_tools(tmp_path, DirectWeb(transport=httpx.MockTransport(handler)), [
        ("fetch_source", {"url": "https://doi.org/10.1/abc"}),
        ("record_evidence", {"claim": "c", "claim_kind": "metric_definition", "url": "https://doi.org/10.1/abc",
                             "excerpt": EXCERPT, "applicability": "a", "reasoning": "r"})])
    assert r["source"]["tier"] == 5 and r["admissibility"] == "lead_only"


# --- provider independence -------------------------------------------------------------

def imports(module):
    tree = ast.parse((PKG / f"{module}.py").read_text())
    names = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            names.add(node.module or "")
        elif isinstance(node, ast.Import):
            names.update(a.name for a in node.names)
    return names


@pytest.mark.parametrize("module", ["agent", "evidence", "mcp_server", "store", "planner", "compare", "results"])
def test_no_direct_brightdata_dependency(module):
    assert not any("brightdata" in n for n in imports(module))
