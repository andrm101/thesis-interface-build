"""End-to-end smoke test: reproduce.py --quick runs every analysis."""
import reproduce


def test_reproduce_quick(tmp_path):
    out = tmp_path / "RESULTS.md"
    reproduce.main(["--quick", "-o", str(out)])
    text = out.read_text(encoding="utf-8")
    for sec in ("A1", "A2", "A3b", "A4", "A5", "A6",
                "B1", "B2", "B5", "B6", "B7", "B8", "B9", "B10", "B11", "B12",
                "B13"):
        assert f"**{sec} ·" in text, sec
    assert "agreement with the thesis typology 100%" in text   # B2
