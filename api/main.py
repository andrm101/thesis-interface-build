"""
FastAPI back end for the Angular dashboard (web/).

    uvicorn api.main:app --port 8000          # from the repository root

Serves the JSON API under /api, the thesis figures under /figures and, when
the Angular app has been built (web/dist/web/browser), the app itself at /.
"""
from __future__ import annotations

import os
import sys
import uuid
from collections import OrderedDict
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from api import service as S

WEB_DIST = os.environ.get("WEB_DIST",
                          os.path.join(ROOT, "web", "dist", "web", "browser"))


@asynccontextmanager
async def lifespan(_app: FastAPI):
    S.warm()
    yield


app = FastAPI(title="EU Innovation Panel API", version="1.0",
              lifespan=lifespan)
app.add_middleware(  # the Angular dev server runs on :4200
    CORSMiddleware, allow_origins=["http://localhost:4200"],
    allow_methods=["GET", "POST"], allow_headers=["*"])


def _guard(fn, *a, **k):
    try:
        return fn(*a, **k)
    except KeyError as e:
        raise HTTPException(404, str(e).strip("'\"")) from e
    except ValueError as e:
        raise HTTPException(422, str(e)) from e


# ── request bodies ─────────────────────────────────────────────────────────
class LPRequest(BaseModel):
    outcome: str
    shock: str
    horizons: int = Field(5, ge=1, le=10)
    lags: int = Field(1, ge=0, le=3)
    levels: bool = False
    split: bool = False


class EventStudyRequest(BaseModel):
    outcome: str
    event_set: str = "R&D tax reforms"
    custom: str = ""
    control: Literal["Never treated", "Not yet treated"] = "Not yet treated"
    detrend: bool = False
    pre: int = Field(3, ge=2, le=6)
    post: int = Field(5, ge=1, le=8)
    n_boot: int = Field(99, ge=19, le=499)


# ── endpoints ──────────────────────────────────────────────────────────────
@app.get("/api/health")
def health():
    return {"status": "ok"}


@app.get("/api/meta")
def meta():
    return S.meta()


@app.get("/api/series")
def series(variable: str, countries: str | None = Query(None)):
    names = [c for c in (countries or "").split(",") if c] or None
    return _guard(S.series, variable, names)


@app.get("/api/snapshot")
def snapshot(variable: str, year: int):
    return _guard(S.snapshot, variable, year)


@app.get("/api/typology")
def typology():
    return S.typology()


@app.get("/api/clubs")
def clubs(variable: str = "Y_per_worker"):
    return _guard(S.convergence_clubs, variable)


@app.post("/api/local-projections")
def local_projections(req: LPRequest):
    return _guard(S.local_projection, **req.model_dump())


@app.post("/api/event-study")
def event_study(req: EventStudyRequest):
    return _guard(S.event_study, **req.model_dump())


@app.get("/api/coverage")
def coverage():
    return S.coverage()


@app.get("/api/results")
def results_index():
    return S.results_index()


@app.get("/api/results/{section_id}")
def result(section_id: str):
    return _guard(S.result, section_id)


@app.get("/api/figures")
def figures():
    return S.figures()


# ── research assistant (optional: needs the anthropic package + a key) ────
class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    conversation_id: str | None = None


_CONVERSATIONS: OrderedDict[str, object] = OrderedDict()
_MAX_CONVERSATIONS = 50


@app.get("/api/assistant/status")
def assistant_status():
    from llm import client as L
    ok, why = L.availability()
    return {"available": ok, "model": L.MODEL, "detail": why}


@app.post("/api/assistant")
def assistant(req: AskRequest):
    from agent.runner import Assistant
    from llm import client as L
    ok, why = L.availability()
    if not ok:
        raise HTTPException(503, why)
    cid = req.conversation_id or uuid.uuid4().hex
    conv = _CONVERSATIONS.pop(cid, None) or Assistant()
    _CONVERSATIONS[cid] = conv
    while len(_CONVERSATIONS) > _MAX_CONVERSATIONS:
        _CONVERSATIONS.popitem(last=False)
    try:
        with conv.lock:
            reply = conv.ask(req.question)
    except L.LLMRefusal as e:
        raise HTTPException(422, str(e)) from e
    except Exception as e:                      # API errors: report, keep serving
        raise HTTPException(502, f"assistant failed: {e}") from e
    return {"conversation_id": cid, "answer": reply.text,
            "tool_calls": [vars(c) for c in reply.tool_calls],
            "grounding": reply.grounding, "stop_reason": reply.stop_reason}


if os.path.isdir(S.FIG_DIR):
    app.mount("/figures", StaticFiles(directory=S.FIG_DIR), name="figures")
if os.path.isdir(WEB_DIST):   # the built Angular app (hash routing → no rewrites)
    app.mount("/", StaticFiles(directory=WEB_DIST, html=True), name="web")
