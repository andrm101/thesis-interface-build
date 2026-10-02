"""
runner.py — the assistant's conversation loop.

One `Assistant` holds one conversation (append-only, so the cached prefix
stays valid turn to turn). `ask()` sends the question, runs every tool the
model calls (read-only analysis functions), feeds the results back, and
repeats until the model answers. The answer is then checked: every number
in it must appear in a tool result (llm.grounding), and those that do not
are reported as unverified.
"""
from __future__ import annotations

import json
import threading
from collections.abc import Callable
from dataclasses import dataclass, field

from agent import prompts, tools
from llm import client as L
from llm import grounding

MAX_TOOL_CALLS = 8


@dataclass
class ToolCall:
    result_id: str
    name: str
    input: dict
    error: str | None = None


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    grounding: dict = field(default_factory=dict)
    stop_reason: str = ""


class Assistant:
    def __init__(self, client=None, max_tool_calls: int = MAX_TOOL_CALLS):
        self.client = client
        self.max_tool_calls = max_tool_calls
        self.messages: list[dict] = []
        self.results: dict[str, dict] = {}
        self.lock = threading.Lock()       # one turn at a time per conversation

    def _client(self):
        if self.client is None:
            self.client = L.get_client()
        return self.client

    def _run_tool(self, block) -> tuple[dict, ToolCall]:
        rid = f"r{len(self.results) + 1}"
        try:
            out = tools.execute(block.name, dict(block.input or {}))
            self.results[rid] = out
            call = ToolCall(rid, block.name, dict(block.input or {}))
            content = json.dumps({"result_id": rid, **out}, default=str)
            res = {"type": "tool_result", "tool_use_id": block.id, "content": content}
        except Exception as exc:  # noqa: BLE001 — reported to the model, never raised
            call = ToolCall(rid, block.name, dict(block.input or {}), str(exc))
            res = {"type": "tool_result", "tool_use_id": block.id,
                   "content": f"Error: {exc}", "is_error": True}
        return res, call

    def ask(self, question: str,
            on_event: Callable[[str, dict], None] | None = None) -> Reply:
        client = self._client()
        start = len(self.messages)
        try:
            return self._ask(client, question, on_event or (lambda *_: None))
        except Exception:
            del self.messages[start:]      # drop the failed turn, keep the history valid
            raise

    def _ask(self, client, question: str, emit) -> Reply:
        self.messages.append({"role": "user", "content": question})
        calls: list[ToolCall] = []
        system = [{"type": "text", "text": prompts.system_prompt(),
                   "cache_control": {"type": "ephemeral"}}]
        defs = tools.definitions()
        resp = None
        for _ in range(self.max_tool_calls + 2):
            budget_left = len(calls) < self.max_tool_calls
            resp = L.create(client, system=system, tools=defs,
                            tool_choice={"type": "auto" if budget_left else "none"},
                            messages=self.messages,
                            cache_control={"type": "ephemeral"})  # cache the history
            self.messages.append({"role": "assistant", "content": resp.content})
            if resp.stop_reason != "tool_use":
                break
            results = []
            for b in resp.content:
                if getattr(b, "type", "") == "tool_use":
                    emit("tool", {"name": b.name, "input": b.input})
                    res, call = self._run_tool(b)
                    results.append(res)
                    calls.append(call)
            self.messages.append({"role": "user", "content": results})
        text = "".join(getattr(b, "text", "") for b in resp.content
                       if getattr(b, "type", "") == "text").strip()
        if resp.stop_reason == "max_tokens":
            text += "\n\n(The answer was cut off at the length limit.)"
        # every result of the conversation counts: follow-ups may reuse them
        sources = list(self.results.values()) + [question]
        return Reply(text, calls, grounding.check(text, sources), resp.stop_reason)
