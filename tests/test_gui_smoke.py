"""Headless smoke test: build every tab, load the bundled data, and visit
each tab so its refresh hook runs. Skipped when Tk or a display is absent
(CI runs it under xvfb)."""
import pytest

from conftest import DATA_PATH

tk = pytest.importorskip("tkinter")


@pytest.fixture
def app(monkeypatch):
    try:
        root = tk.Tk()
    except tk.TclError as exc:
        pytest.skip(f"no display: {exc}")
    root.withdraw()

    from tkinter import messagebox
    errors = []
    monkeypatch.setattr(messagebox, "showerror", lambda *a, **k: errors.append(a))
    monkeypatch.setattr(messagebox, "showwarning", lambda *a, **k: errors.append(a))

    from app import ThesisApp
    a = ThesisApp(root)
    a._errors = errors
    yield a
    root.destroy()


def test_all_tabs_build_and_refresh(app):
    assert app.nb.index("end") == 15
    assert app.load_path(DATA_PATH)
    # Luxembourg is excluded by default: 22 countries x 24 years.
    assert app.df is not None and len(app.df) == 528
    assert "Luxembourg" not in set(app.df["Country"])
    for i in range(app.nb.index("end")):
        app.nb.select(i)
        app.root.update()
    assert app._errors == []


def test_theme_toggle_roundtrip(app):
    app._toggle_theme()
    app._toggle_theme()
    app.root.update()
    assert app._errors == []


def test_assistant_tab_shows_answer_and_grounding(app):
    from agent.runner import Reply
    app._as_busy = True
    app._as_show(Reply("R&D intensity rose by 0.42 pp [r1].", [],
                       {"verified": [], "unverified": ["0.42"]}, "end_turn"))
    text = app.as_txt.get("1.0", "end")
    assert "0.42 pp [r1]" in text and "not found in any tool result: 0.42" in text
    assert app._as_busy is False
