"""Provider-independent web research: a small interface, a direct HTTP provider, selection.

Every provider returns the same normalized shapes, so the evidence gate never knows which one ran:
  search(query, n) -> [{"title", "url", "snippet", "provider"}]
  fetch(url)       -> {"url", "final_url", "title", "text", "status", "content_type",
                       "provider", "retrieved_at"}
Failures raise ResearchError subclasses; nothing is returned half-valid.

Selection (RESEARCH_PROVIDER): "brightdata" (default) | "direct" | "none". No silent fallback:
a selected provider that is not configured raises NotConfigured on use.
"""

from __future__ import annotations

import os
import re
from datetime import datetime, timezone
from html.parser import HTMLParser
from typing import Protocol
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx


class ResearchError(Exception):
    """Retrieval failed (transport, HTTP status, parsing). Not evidence of anything."""


class NotConfigured(ResearchError):
    pass


class SearchUnsupported(ResearchError):
    pass


class AccessRestricted(ResearchError):
    """Robots.txt, auth, paywall, rate limit: respected, never bypassed."""


class UnsupportedContent(ResearchError):
    pass


class ResearchProvider(Protocol):
    name: str

    def search(self, query: str, n: int = 8) -> list[dict[str, str]]: ...

    def fetch(self, url: str) -> dict[str, str]: ...


# --- shared normalization --------------------------------------------------------------

class _Text(HTMLParser):
    SKIP = {"script", "style", "noscript", "nav", "header", "footer", "svg"}

    def __init__(self):
        super().__init__()
        self.parts: list[str] = []
        self.depth = 0
        self.title = ""
        self._in_title = False

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.depth += 1
        self._in_title = tag == "title"

    def handle_endtag(self, tag):
        if tag in self.SKIP and self.depth:
            self.depth -= 1
        self._in_title = False

    def handle_data(self, data):
        if self._in_title and not self.title:
            self.title = data.strip()
        elif not self.depth:
            self.parts.append(data)


def html_to_text(html: str) -> tuple[str, str]:
    """(title, visible text with whitespace collapsed)."""
    p = _Text()
    p.feed(html)
    return p.title, re.sub(r"\s+", " ", " ".join(p.parts)).strip()


def page(provider: str, url: str, final_url: str, content_type: str, body: str) -> dict[str, str]:
    """Normalize a retrieved body. HTML and plain text only; anything else is refused."""
    ctype = content_type.split(";")[0].strip().lower()
    if body.lstrip()[:5] == "%PDF-" or ctype == "application/pdf":
        raise UnsupportedContent("PDF content is not parsed")
    if ctype in ("text/html", "application/xhtml+xml") or (not ctype and "<html" in body[:2000].lower()):
        title, text = html_to_text(body)
    elif ctype == "text/plain":
        title, text = "", re.sub(r"\s+", " ", body).strip()
    else:
        raise UnsupportedContent(f"content type {ctype or 'unknown'} is not supported")
    return {"url": url, "final_url": final_url, "title": title, "text": text,
            "status": "ok" if text else "empty", "content_type": ctype or "text/html",
            "provider": provider, "retrieved_at": datetime.now(timezone.utc).isoformat()}


# --- direct HTTP -----------------------------------------------------------------------

RESTRICTED = {401, 402, 403, 407, 429, 451}


class DirectWeb:
    """Plain HTTP GET of public pages. No search engine, no browser, no access-control bypass."""

    name = "direct"
    USER_AGENT = "research-lab/0.1 (evidence retrieval; respects robots.txt)"

    def __init__(self, transport: httpx.BaseTransport | None = None, timeout: float = 20.0,
                 max_bytes: int = 3_000_000):
        self.http = httpx.Client(timeout=timeout, transport=transport, follow_redirects=True,
                                 max_redirects=5, headers={"User-Agent": self.USER_AGENT})
        self.max_bytes = max_bytes
        self._robots: dict[str, RobotFileParser | None] = {}

    def search(self, query: str, n: int = 8) -> list[dict[str, str]]:
        raise SearchUnsupported("the direct provider has no search engine; fetch a known URL instead")

    def _allowed(self, url: str) -> bool:
        """RFC 9309: robots 4xx -> allowed; 5xx or unreachable -> disallowed."""
        p = urlparse(url)
        origin = f"{p.scheme}://{p.netloc}"
        if origin not in self._robots:
            try:
                r = self.http.get(origin + "/robots.txt")
                if r.status_code >= 500:
                    rp: RobotFileParser | None = None
                else:
                    rp = RobotFileParser()
                    rp.parse(r.text.splitlines() if r.status_code < 400 else [])
            except httpx.HTTPError:
                rp = None
            self._robots[origin] = rp
        rp = self._robots[origin]
        return rp is not None and rp.can_fetch(self.USER_AGENT, url)

    def fetch(self, url: str) -> dict[str, str]:
        if urlparse(url).scheme not in ("http", "https") or not urlparse(url).netloc:
            raise UnsupportedContent(f"not an http(s) URL: {url}")
        if not self._allowed(url):
            raise AccessRestricted("disallowed by robots.txt (or robots.txt unreachable)")
        try:
            with self.http.stream("GET", url) as r:
                if r.status_code in RESTRICTED:
                    raise AccessRestricted(f"HTTP {r.status_code}: access restricted; not bypassed")
                if r.status_code >= 400:
                    raise ResearchError(f"HTTP {r.status_code}")
                raw = b""
                for chunk in r.iter_bytes():
                    raw += chunk
                    if len(raw) > self.max_bytes:
                        raise UnsupportedContent(f"page exceeds {self.max_bytes} bytes")
                body = raw.decode(r.encoding or "utf-8", errors="replace")
                return page(self.name, url, str(r.url), r.headers.get("content-type", ""), body)
        except httpx.TimeoutException as e:
            raise ResearchError(f"timeout: {e}") from e
        except httpx.HTTPError as e:
            raise ResearchError(f"transport: {e}") from e


class Unconfigured:
    """Stands in for a selected provider that lacks configuration. Fails on use, never falls back."""

    def __init__(self, name: str, reason: str):
        self.name, self.reason = name, reason

    def search(self, query: str, n: int = 8):
        raise NotConfigured(self.reason)

    def fetch(self, url: str):
        raise NotConfigured(self.reason)


def provider_from_env(env: dict[str, str] | None = None) -> ResearchProvider | None:
    env = os.environ if env is None else env
    choice = env.get("RESEARCH_PROVIDER", "brightdata").strip().lower()
    if choice == "none":
        return None
    if choice == "direct":
        return DirectWeb()
    if choice == "brightdata":
        from research_lab.brightdata import BrightData
        bd = BrightData.from_env(env)
        return bd or Unconfigured("brightdata", "Bright Data selected but BRIGHTDATA_API_TOKEN, "
                                  "BRIGHTDATA_SERP_ZONE and BRIGHTDATA_UNLOCKER_ZONE are not all set")
    raise ValueError(f"unknown RESEARCH_PROVIDER {choice!r} (brightdata | direct | none)")
