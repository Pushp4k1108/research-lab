"""Bright Data research provider: SERP search and Web Unlocker fetch. Retrieval only.

Uses POST https://api.brightdata.com/request with two zones:
  BRIGHTDATA_SERP_ZONE      SERP API zone (Google results parsed via brd_json=1)
  BRIGHTDATA_UNLOCKER_ZONE  Web Unlocker zone (raw page HTML)
  BRIGHTDATA_API_TOKEN      bearer token
Not yet verified against the live service.
"""

from __future__ import annotations

import json
import os
from urllib.parse import quote_plus

import httpx

from research_lab.research import ResearchError, page

ENDPOINT = "https://api.brightdata.com/request"


class BrightDataError(ResearchError):
    pass


class BrightData:
    name = "brightdata"

    def __init__(self, token: str, serp_zone: str, unlocker_zone: str,
                 transport: httpx.BaseTransport | None = None, timeout: float = 90.0):
        self.serp_zone, self.unlocker_zone = serp_zone, unlocker_zone
        self.http = httpx.Client(headers={"Authorization": f"Bearer {token}"}, timeout=timeout, transport=transport)

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> BrightData | None:
        env = os.environ if env is None else env
        token = env.get("BRIGHTDATA_API_TOKEN")
        serp, unlocker = env.get("BRIGHTDATA_SERP_ZONE"), env.get("BRIGHTDATA_UNLOCKER_ZONE")
        return cls(token, serp, unlocker) if token and serp and unlocker else None

    def _request(self, zone: str, url: str) -> httpx.Response:
        try:
            r = self.http.post(ENDPOINT, json={"zone": zone, "url": url, "format": "raw"})
        except httpx.HTTPError as e:
            raise BrightDataError(f"transport: {e}") from e
        if r.status_code >= 400:
            raise BrightDataError(f"HTTP {r.status_code}: {r.text[:200]}")
        return r

    def search(self, query: str, n: int = 8) -> list[dict[str, str]]:
        r = self._request(self.serp_zone, f"https://www.google.com/search?q={quote_plus(query)}&brd_json=1")
        try:
            organic = json.loads(r.text).get("organic") or []
        except (ValueError, AttributeError) as e:
            raise BrightDataError("SERP response was not parsed JSON") from e
        return [{"title": o.get("title", ""), "url": o["link"], "snippet": o.get("description", ""),
                 "provider": self.name} for o in organic[:n] if o.get("link")]

    def fetch(self, url: str) -> dict[str, str]:
        r = self._request(self.unlocker_zone, url)
        # Unlocker returns the target body; it does not forward the target's final URL.
        return page(self.name, url, url, r.headers.get("content-type", ""), r.text)
