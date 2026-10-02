"""agent/: the tool loop with a scripted fake Claude client (no network),
the real analysis tools, the grounding check and the API endpoints."""
from types import SimpleNamespace as NS

import pytest

from agent import tools
from agent.runner import Assistant
from llm import client as L
from llm import grounding


def _tool(id_, name, **inp):
    return NS(type="tool_use", id=id_, name=name, input=inp)


def _text(t):
    return NS(type="text", text=t)


class ScriptedClient:
    """Returns the scripted responses in order and records every request."""
    def __init__(self, *responses):
        self.responses, self.calls = list(responses), []
        outer = self

        class _Msgs:
            def create(self, **kw):
                outer.calls.append(kw)
                stop, content = outer.responses.pop(0)
                return NS(stop_reason=stop, content=content, stop_details=None)
        self.beta = NS(messages=_Msgs())


def test_tool_definitions_are_strict_and_use_real_names():
    defs = {d["name"]: d for d in tools.definitions()}
    assert set(defs) == set(tools.IMPL)
    for d in defs.values():
        sch = d["input_schema"]
        assert d["strict"] and sch["additionalProperties"] is False
        assert sorted(sch["required"]) == sorted(sch["properties"])
    assert "RD_pct_GDP" in defs["local_projection"]["input_schema"]["properties"]["outcome"]["enum"]
    assert "D5" in defs["result_section"]["input_schema"]["properties"]["section_id"]["enum"]


def test_loop_runs_tools_and_grounds_the_answer():
    lp = {"outcome": "RD_pct_GDP", "shock": "GBARD_pct_GDP", "horizons": 2, "levels": True,
              "split_by_group": False}
    beta0 = tools.local_projection(**lp)["rows"][0]["beta"]
    client = ScriptedClient(
        ("tool_use", [_text("Let me check."), _tool("t1", "local_projection", **lp),
                      _tool("t2", "result_section", section_id="D5")]),
        ("end_turn", [_text(f"A 1 pp rise in GBARD raises R&D intensity by "
                            f"{beta0:.2f} pp on impact [r1]; the multiplier is 12.34 "
                            f"in my memory.")]),
    )
    a = Assistant(client=client)
    events = []
    reply = a.ask("Does public R&D crowd in private R&D?", lambda k, d: events.append(d["name"]))
    assert [c.name for c in reply.tool_calls] == ["local_projection", "result_section"]
    assert [c.result_id for c in reply.tool_calls] == ["r1", "r2"] and events
    assert f"{beta0:.2f}" in reply.grounding["verified"]
    assert reply.grounding["unverified"] == ["12.34"]         # not in any tool result
    first = client.calls[0]
    assert first["fallbacks"] == "default" and first["model"] == L.MODEL
    assert first["system"][0]["cache_control"] == {"type": "ephemeral"}
    # tool results went back in ONE user message, tagged with their ids
    tool_msg = a.messages[2]
    assert tool_msg["role"] == "user" and len(tool_msg["content"]) == 2
    assert '"result_id": "r1"' in tool_msg["content"][0]["content"]


def test_tool_errors_are_returned_not_raised():
    client = ScriptedClient(
        ("tool_use", [_tool("t1", "event_study", outcome="RD_pct_GDP", event_set="Custom",
                            custom_events="Atlantis:2010", control="Not yet treated",
                            detrend=False)]),
        ("end_turn", [_text("Those events are not in the panel.")]))
    a = Assistant(client=client)
    reply = a.ask("Effect of reforms in Atlantis?")
    assert reply.tool_calls[0].error
    assert a.messages[2]["content"][0]["is_error"] is True


def test_tool_budget_forces_an_answer():
    many = [("tool_use", [_tool(f"t{i}", "typology")]) for i in range(3)]
    client = ScriptedClient(*many, ("end_turn", [_text("Done.")]))
    reply = Assistant(client=client, max_tool_calls=2).ask("Loop forever")
    assert reply.text == "Done."
    assert client.calls[2]["tool_choice"] == {"type": "none"}


def test_failed_turn_is_rolled_back():
    client = ScriptedClient(("refusal", []))
    a = Assistant(client=client)
    with pytest.raises(L.LLMRefusal):
        a.ask("anything")
    assert a.messages == []


def test_grounding_rules():
    g = grounding.check("Output rose 0.75 % (p = 0.003) in 2009 at h = 4; share 43 %; "
                        "R&D intensity 3.49 in Sweden; −0.42 pp.",
                        [{"beta": 0.7512, "p": 0.0031, "share": 0.43, "v": 3.4896,
                          "x": 0.4236}])
    assert g["unverified"] == [] and len(g["verified"]) == 5


def test_api_assistant_unavailable_without_key(monkeypatch, tmp_path):
    pytest.importorskip("fastapi")
    from fastapi.testclient import TestClient

    from api.main import app
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_AUTH_TOKEN", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    with TestClient(app) as c:
        st = c.get("/api/assistant/status").json()
        assert st["available"] is False and "ANTHROPIC_API_KEY" in st["detail"]
        assert c.post("/api/assistant", json={"question": "hi"}).status_code == 503
