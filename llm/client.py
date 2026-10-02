"""
client.py — the one place the project talks to the Claude API.

Optional: the app, reproduce.py and CI never need it. Everything here
imports `anthropic` lazily and reports *why* the LLM features are off
(package missing, no credentials) instead of failing.

Requests go through the beta Messages endpoint so a request the model's
safety classifiers decline is retried server-side on Anthropic's
recommended fallback model (`fallbacks="default"`).
"""
from __future__ import annotations

import os
from pathlib import Path

MODEL = os.environ.get("THESIS_LLM_MODEL", "claude-opus-5")
FALLBACK_BETA = "server-side-fallback-2026-07-01"
MAX_TOKENS = 16000


class LLMUnavailable(RuntimeError):
    """Raised when the anthropic package or credentials are missing."""


class LLMRefusal(RuntimeError):
    """The model (and its fallback) declined the request."""


def availability() -> tuple[bool, str]:
    """(available, reason) — cheap, no network call."""
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False, ("the anthropic package is not installed "
                       "(pip install -r requirements-llm.txt)")
    if os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN"):
        return True, f"using {MODEL}"
    if (Path.home() / ".config" / "anthropic").exists():
        return True, f"using {MODEL} (ant auth profile)"
    return False, ("no Claude API credentials: set ANTHROPIC_API_KEY "
                   "(or run ant auth login)")


def get_client():
    ok, why = availability()
    if not ok:
        raise LLMUnavailable(why)
    import anthropic
    return anthropic.Anthropic()


def _kwargs(kw: dict) -> dict:
    kw.setdefault("model", MODEL)
    kw.setdefault("max_tokens", MAX_TOKENS)
    kw.setdefault("betas", [FALLBACK_BETA])
    kw.setdefault("fallbacks", "default")
    return kw


def create(client, **kw):
    """client.beta.messages.create with the project defaults."""
    resp = client.beta.messages.create(**_kwargs(kw))
    check(resp)
    return resp


def parse(client, **kw):
    """client.beta.messages.parse (structured output) with the defaults."""
    resp = client.beta.messages.parse(**_kwargs(kw))
    check(resp)
    return resp


def check(resp) -> None:
    """Raise on a refusal before anyone reads `content`."""
    if getattr(resp, "stop_reason", None) == "refusal":
        det = getattr(resp, "stop_details", None)
        cat = getattr(det, "category", None) if det else None
        raise LLMRefusal("the model declined this request"
                         + (f" (category: {cat})" if cat else ""))
