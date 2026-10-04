"""Minimal Flowlab REST client. The only boundary between Research Lab and Flowlab."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import httpx


class FlowlabError(Exception):
    """A non-success HTTP response. ``engineering`` marks 409s: fix the model, don't retry."""

    def __init__(self, status: int, detail: Any, path: str):
        super().__init__(f"{status} {path}: {detail}")
        self.status, self.detail, self.path = status, detail, path

    @property
    def engineering(self) -> bool:
        return self.status == 409

    @property
    def infrastructure(self) -> bool:
        return self.status >= 500


class PollTimeout(Exception):
    """No Generation appeared for an input_hash in time. Unresolved, not failed."""


class Flowlab:
    def __init__(self, base_url: str = "http://localhost:8000", timeout: float = 360.0,
                 transport: httpx.BaseTransport | None = None):
        # transport: tests inject httpx.MockTransport; production uses the default.
        self.http = httpx.Client(base_url=base_url, timeout=timeout, transport=transport)

    def _call(self, method: str, path: str, **kw) -> httpx.Response:
        r = self.http.request(method, path, **kw)
        if r.status_code >= 400:
            try:
                detail = r.json().get("detail", r.text)
            except ValueError:
                detail = r.text
            raise FlowlabError(r.status_code, detail, path)
        return r

    def health(self) -> dict:
        return self._call("GET", "/health").json()

    def upload_geometry(self, path: Path) -> dict:
        with open(path, "rb") as f:
            return self._call("POST", "/api/geometry", files={"file": (path.name, f)}).json()

    def create_model(self, draft: dict) -> dict:
        return self._call("POST", "/api/models", json=draft).json()

    def get_model(self, model_id: str, version: int | None = None) -> dict:
        params = {"version": version} if version is not None else None
        return self._call("GET", f"/api/models/{model_id}", params=params).json()

    def derive_inventory(self, model_id: str) -> tuple[int, dict]:
        r = self._call("POST", f"/api/models/{model_id}/inventory")
        return r.status_code, r.json()

    def request_generation(self, model_id: str, configuration: dict, version: int | None = None) -> tuple[int, dict]:
        """200 = existing Generation (dedup hit, envelope); 202 = accepted, poll by input_hash."""
        body = {"model_id": model_id, "version": version, "configuration": configuration}
        r = self._call("POST", "/api/generations", json=body)
        return r.status_code, r.json()

    def list_generations(self, **filters) -> list[dict]:
        return self._call("GET", "/api/generations", params=filters).json()["generations"]

    def get_generation(self, generation_id: str) -> dict:
        return self._call("GET", f"/api/generations/{generation_id}").json()

    def await_generation(self, input_hash: str, interval: float = 1.5, timeout: float = 1200.0) -> tuple[dict, int]:
        """Poll until a Generation exists for input_hash. Returns (envelope, polls)."""
        deadline, polls = time.monotonic() + timeout, 0
        while True:
            polls += 1
            found = self.list_generations(input_hash=input_hash)
            if found:
                return found[0], polls
            if time.monotonic() > deadline:
                raise PollTimeout(input_hash)
            time.sleep(interval)
