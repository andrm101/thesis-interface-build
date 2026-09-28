"""End-to-end smoke test: reproduce.py --quick runs every analysis."""
import reproduce


def test_reproduce_quick(tmp_path):
    out = tmp_path / "RESULTS.md"
    reproduce.main(["--quick", "--figures", "-o", str(out)])
    text = out.read_text(encoding="utf-8")
    for sec in ("A1", "A2", "A3b", "A4", "A5", "A6",
                "B1", "B2", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12",
                "B13", "C1", "D1", "D2", "D3", "D4", "D5"):
        assert f"**{sec} ·" in text, sec
    assert "agreement with the thesis typology 100%" in text   # B2
    figs = sorted(p.name for p in (tmp_path / "figures").glob("*.pdf"))
    assert len(figs) == 8, figs
    tex = (tmp_path / "tables" / "tab_gmm_grid.tex").read_text()
    assert r"\toprule" in tex and "$^{*}^{*}" not in tex

