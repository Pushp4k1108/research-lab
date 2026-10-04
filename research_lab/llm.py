"""LLM providers behind one small interface. The agent loop never sees a vendor format.

Neutral history the agent keeps (providers translate it to their wire format):
  {"role": "user", "text": str}
  {"role": "assistant", "turn": Turn}
  {"role": "tool_results", "results": [ToolResult, ...]}
A provider returns a Turn: text, parsed tool calls, and a normalized stop:
  "tool_calls" | "end" | "max_tokens" | "refusal" | "other"

Selection: RESEARCH_LLM_PROVIDER = "groq" (default) | "anthropic".
  groq:      GROQ_API_KEY (required), GROQ_MODEL, GROQ_BASE_URL, GROQ_REASONING_EFFORT
  anthropic: SDK credential resolution (ANTHROPIC_API_KEY or `ant auth login`), ANTHROPIC_MODEL
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Protocol

import httpx

GROQ_BASE_URL = "https://api.groq.com/openai/v1"
GROQ_MODEL = "openai/gpt-oss-120b"
ANTHROPIC_MODEL = "claude-opus-5-5"


class LLMError(Exception):
    """The model call failed (HTTP, transport, malformed response). Not an agent decision."""


class LLMNotConfigured(LLMError):
    pass


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict[str, Any] | None  # None when the model emitted unparseable arguments
    parse_error: str | None = None


@dataclass
class Turn:
    text: str
    tool_calls: list[ToolCall]
    stop: str
    raw: Any = None  # provider-native assistant message, replayed verbatim by that provider


@dataclass
class ToolResult:
    call_id: str
    content: str  # JSON text
    is_error: bool = False
    name: str = ""  # tool name, for history elision summaries


class LLMProvider(Protocol):
    name: str
    model: str
    usage: list[dict[str, int]]

    def step(self, system: str, tools: list[dict], history: list[dict]) -> Turn: ...


# --- OpenAI-compatible chat completions (Groq) -----------------------------------------

class OpenAICompatible:
    """POST {base_url}/chat/completions with function tools. Only our own tools are sent:
    no provider-hosted browser/search/code tools are enabled."""

    RETRY_STATUS = {429, 500, 502, 503, 504}

    def __init__(self, api_key: str, model: str, base_url: str, name: str = "openai_compatible",
                 extra: dict[str, Any] | None = None, transport: httpx.BaseTransport | None = None,
                 timeout: float = 120.0, max_retries: int = 5, max_tokens: int = 512):
        self.name, self.model, self.extra, self.max_retries = name, model, extra or {}, max_retries
        self.max_tokens = max_tokens
        self.http = httpx.Client(base_url=base_url.rstrip("/"), timeout=timeout, transport=transport,
                                 headers={"Authorization": f"Bearer {api_key}"})
        self.usage: list[dict[str, int]] = []

    @staticmethod
    def tools(tools: list[dict]) -> list[dict]:
        return [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                  "parameters": t["input_schema"]}} for t in tools]

    @staticmethod
    def messages(system: str, history: list[dict]) -> list[dict]:
        out: list[dict] = [{"role": "system", "content": system}]
        for h in history:
            if h["role"] == "user":
                out.append({"role": "user", "content": h["text"]})
            elif h["role"] == "assistant":
                t: Turn = h["turn"]
                msg: dict[str, Any] = {"role": "assistant", "content": t.text or None}
                if t.tool_calls:
                    msg["tool_calls"] = [{"id": c.id, "type": "function", "function": {
                        "name": c.name, "arguments": json.dumps(c.arguments if c.arguments is not None else {})}}
                        for c in t.tool_calls]
                out.append(msg)
            else:
                out += [{"role": "tool", "tool_call_id": r.call_id, "content": r.content} for r in h["results"]]
        return out

    @staticmethod
    def parse(body: dict) -> Turn:
        try:
            choice = body["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError) as e:
            raise LLMError(f"malformed completion: {str(body)[:200]}") from e
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function") or {}
            raw = fn.get("arguments") or "{}"
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                err = None if isinstance(args, dict) else "arguments are not a JSON object"
            except ValueError as e:
                args, err = None, f"invalid JSON arguments: {e}"
            calls.append(ToolCall(id=c.get("id", ""), name=fn.get("name", ""),
                                  arguments=args if err is None else None, parse_error=err))
        reason = choice.get("finish_reason")
        stop = ("tool_calls" if calls else {"stop": "end", "length": "max_tokens",
                                            "content_filter": "refusal"}.get(reason, "other"))
        # Reasoning models may inline <think>...</think>; it is not part of the answer or the history.
        content = re.sub(r"<think>.*?(</think>|$)", "", msg.get("content") or "", flags=re.S).strip()
        return Turn(text=content, tool_calls=calls, stop=stop, raw=msg)

    def step(self, system: str, tools: list[dict], history: list[dict]) -> Turn:
        payload = {"model": self.model, "messages": self.messages(system, history), "tools": self.tools(tools),
                   "tool_choice": "auto", "max_completion_tokens": self.max_tokens, **self.extra}
        for attempt in range(self.max_retries + 1):
            try:
                t0 = time.monotonic()
                print(
                    f"[LLM] request start: payload_chars={len(json.dumps(payload))} "
                    f"attempt={attempt + 1}",
                    flush=True,
                )
                r = self.http.post("/chat/completions", json=payload)
                print(
                    f"[LLM] response after {time.monotonic() - t0:.1f}s "
                    f"status={r.status_code}",
                    flush=True,
                )
            except httpx.HTTPError as e:
                if attempt == self.max_retries:
                    raise LLMError(f"transport: {e}") from e
                time.sleep(2 ** attempt)
                continue
            if r.status_code in self.RETRY_STATUS and attempt < self.max_retries:
                # Per-minute token limits: wait out the window the server names.
                time.sleep(min(float(r.headers.get("retry-after", 2 ** (attempt + 2))), 65))
                continue
            if r.status_code >= 400:
                raise LLMError(f"HTTP {r.status_code}: {r.text[:300]}")
            body = r.json()
            u = body.get("usage") or {}
            self.usage.append({"input": u.get("prompt_tokens", 0), "output": u.get("completion_tokens", 0)})
            return self.parse(body)
        raise LLMError("unreachable")


# --- Anthropic Messages API ------------------------------------------------------------

class Anthropic:
    """Claude via the official SDK, with prompt caching and server-side refusal fallback."""

    name = "anthropic"

    def __init__(self, model: str = ANTHROPIC_MODEL, effort: str = "medium", client: Any = None):
        if client is None:
            import anthropic
            client = anthropic.Anthropic()
        self.client, self.model, self.effort = client, model, effort
        self.usage: list[dict[str, int]] = []

    @staticmethod
    def messages(history: list[dict]) -> list[dict]:
        out = []
        for h in history:
            if h["role"] == "user":
                out.append({"role": "user", "content": h["text"]})
            elif h["role"] == "assistant":
                out.append({"role": "assistant", "content": h["turn"].raw})  # keeps thinking blocks intact
            else:
                out.append({"role": "user", "content": [
                    {"type": "tool_result", "tool_use_id": r.call_id, "content": r.content,
                     **({"is_error": True} if r.is_error else {})} for r in h["results"]]})
        return out

    @staticmethod
    def parse(resp: Any) -> Turn:
        blocks = resp.content
        calls = [ToolCall(id=b.id, name=b.name, arguments=dict(b.input or {})) for b in blocks if b.type == "tool_use"]
        text = "\n".join(b.text for b in blocks if b.type == "text")
        stop = "tool_calls" if calls else {"end_turn": "end", "max_tokens": "max_tokens",
                                           "refusal": "refusal"}.get(resp.stop_reason, "other")
        return Turn(text=text, tool_calls=calls, stop=stop, raw=blocks)

    def step(self, system: str, tools: list[dict], history: list[dict]) -> Turn:
        import anthropic
        try:
            r = self.client.beta.messages.create(
                model=self.model, max_tokens=16000,
                system=[{"type": "text", "text": system}], tools=tools, messages=self.messages(history),
                output_config={"effort": self.effort},
                cache_control={"type": "ephemeral"},
                betas=["server-side-fallback-2026-07-01"], fallbacks="default",
            )
        except anthropic.APIError as e:
            raise LLMError(f"anthropic: {e}") from e
        u = r.usage
        self.usage.append({"input": u.input_tokens, "output": u.output_tokens,
                           "cache_read": u.cache_read_input_tokens or 0,
                           "cache_write": u.cache_creation_input_tokens or 0})
        return self.parse(r)


# --- selection -------------------------------------------------------------------------

def llm_from_env(env: dict[str, str] | None = None, model: str | None = None) -> LLMProvider:
    env = os.environ if env is None else env
    choice = env.get("RESEARCH_LLM_PROVIDER", "groq").strip().lower()
    if choice == "groq":
        key = env.get("GROQ_API_KEY")
        if not key:
            raise LLMNotConfigured("RESEARCH_LLM_PROVIDER=groq requires GROQ_API_KEY")
        return OpenAICompatible(key, model or env.get("GROQ_MODEL", GROQ_MODEL),
                                env.get("GROQ_BASE_URL", GROQ_BASE_URL), name="groq",
                                extra={"reasoning_effort": env.get("GROQ_REASONING_EFFORT", "medium")})
    if choice == "anthropic":
        return Anthropic(model or env.get("ANTHROPIC_MODEL", ANTHROPIC_MODEL), env.get("ANTHROPIC_EFFORT", "medium"))
    raise LLMNotConfigured(f"unknown RESEARCH_LLM_PROVIDER {choice!r} (groq | anthropic)")
