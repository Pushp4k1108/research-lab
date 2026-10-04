"""Background campaign runner: launches the EXISTING agent loop without blocking the UI.

The UI thread creates the campaign record + uploads geometry synchronously
(fast, so validation errors surface immediately), then starts ONE background
thread running the existing Controller loop over an in-process MCP server,
then writes run.json + report.md exactly like the CLI. Stdlib only.
"""

from __future__ import annotations

import asyncio
import json
import os
import threading
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable

from research_lab.agent import Controller, Limits, finish
from research_lab.flowlab import Flowlab
from research_lab.llm import llm_from_env
from research_lab.mcp_server import build_server
from research_lab.research import provider_from_env
from research_lab.store import Store
from research_lab.templates import Template, demo_brief


@dataclass
class RunStatus:
    campaign_id: str
    running: bool
    started_at: str
    error: str | None = None


@dataclass
class Runner:
    """Owns background agent threads for one UI process."""

    store: Store
    flowlab_base_url: str
    poll_timeout: float = 1200.0
    # Factories so tests can inject fakes without touching env/network.
    llm_factory: Callable[[], Any] | None = None
    research_factory: Callable[[], Any] | None = None
    flowlab_factory: Callable[[], Any] | None = None
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _threads: dict[str, threading.Thread] = field(default_factory=dict, repr=False)
    _errors: dict[str, str] = field(default_factory=dict, repr=False)

    def running_campaigns(self) -> list[str]:
        with self._lock:
            return [cid for cid, t in self._threads.items() if t.is_alive()]

    def is_running(self, campaign_id: str) -> bool:
        with self._lock:
            t = self._threads.get(campaign_id)
            return t.is_alive() if t else False

    def last_error(self, campaign_id: str) -> str | None:
        with self._lock:
            return self._errors.get(campaign_id)

    def start(self, campaign_id: str, geometry_path: str, template: Template,
              setup: dict, brief: str | None = None, max_llm_turns: int = 30) -> RunStatus:
        """Start the agent loop for an already-created campaign in the background."""
        with self._lock:
            t = self._threads.get(campaign_id)
            if t is not None and t.is_alive():
                raise ValueError(f"campaign {campaign_id} is already running")
            thread = threading.Thread(target=self._run, kwargs={
                "campaign_id": campaign_id, "geometry_path": geometry_path,
                "template": template, "setup": setup, "brief": brief,
                "max_llm_turns": max_llm_turns,
            }, daemon=True, name=f"research-{campaign_id}")
            self._threads[campaign_id] = thread
            self._errors.pop(campaign_id, None)
            started = datetime.now(timezone.utc).isoformat()
            thread.start()
            return RunStatus(campaign_id=campaign_id, running=True, started_at=started)

    def _research(self, campaign_id: str, geometry_path: str, template: Template,
                        setup: dict, brief: str | None, max_llm_turns: int) -> None:
        from mcp.client import Client  # local import: mcp only needed when running

        campaign = self.store.campaign(campaign_id)
        llm = self.llm_factory() if self.llm_factory else llm_from_env()
        research = self.research_factory() if self.research_factory else provider_from_env()
        flowlab = self.flowlab_factory() if self.flowlab_factory else Flowlab(self.flowlab_base_url)
        server = build_server(self.store, flowlab, self.poll_timeout, research)

        def log(event: dict) -> None:
            self._log(event.get("campaign_id") or campaign_id, event)

        async def go():
            async with Client(server) as session:
                ctl = Controller(session, llm, Limits(max_llm_turns, 60, campaign.max_experiments),
                                 on_event=log)
                run = await ctl.research_existing(campaign_id, campaign.question, setup, brief or demo_brief(
                    template.default_demo_objective, template.default_search_lower,
                    template.default_search_upper, template.default_rel_tolerance))
                if run.campaign_id is None:
                    return
                report, _ = await finish(ctl)
            root = self.store.root / run.campaign_id
            (root / "run.json").write_text(json.dumps(
                {"stop_reason": run.stop_reason, "llm_turns": run.llm_turns,
                 "tool_calls": run.tool_calls, "experiments_run": run.experiments_run,
                 "llm": f"{llm.name}/{llm.model}", "geometry_source": Path(geometry_path).name,
                 "template": template.id}, indent=2))
            (root / "report.md").write_text(report)

        asyncio.run(go())

    def _run(self, campaign_id: str, geometry_path: str, template: Template,
             setup: dict, brief: str | None, max_llm_turns: int) -> None:
        try:
            self._research(campaign_id, geometry_path, template, setup, brief, max_llm_turns)
        except Exception as e:  # never kill the UI thread; record for the workspace
            with self._lock:
                self._errors[campaign_id] = f"{type(e).__name__}: {e}"[:500]
            self._log(campaign_id, {"kind": "runner_error", "error": str(e)[:500]})

    def _log(self, campaign_id: str, event: dict) -> None:
        try:
            with open(self.store.root / campaign_id / "activity.jsonl", "a") as f:
                f.write(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                                    **event}, default=str) + "\n")
        except OSError:
            pass


def default_runner(store: Store, flowlab_base_url: str | None = None,
                   poll_timeout: float | None = None) -> Runner:
    base = flowlab_base_url or os.environ.get("FLOWLAB_BASE_URL", "http://localhost:8000")
    try:
        timeout = poll_timeout if poll_timeout is not None else float(
            os.environ.get("RESEARCH_LAB_POLL_TIMEOUT", "1200"))
    except ValueError:
        timeout = 1200.0
    return Runner(store=store, flowlab_base_url=base, poll_timeout=timeout)
