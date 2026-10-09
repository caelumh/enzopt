"""AC-17: the non-weakly-reversible control is trained through the same flow and compared with the WR model on the
held-out sets (table + figure ③(b)). Long tier."""

import csv
import json

import pytest

pytestmark = pytest.mark.long


def test_control_model_compared(full_run):
    d = full_run / "3_fit"
    f = json.loads((d / "fit_summary.json").read_text(encoding="utf-8"))
    c = f["comparisons"]["control"]
    assert c["one_way_steps"] == [5]
    assert len(c["fits"]) >= 1
    assert all(len(x["kr"]) == 5 and x["kr"][4] == 0.0 for x in c["fits"])
    with open(d / "holdout_comparison.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for exp in ("C", "Dp"):
        assert [r for r in rows if r["model"] == "nonWR" and r["experiment"] == exp]
        assert [r for r in rows if r["model"] == "PINN (WR)" and r["experiment"] == exp]
    with open(d / "fig3b_holdout.csv", encoding="utf-8") as fh:
        fig = list(csv.DictReader(fh))
    labels = {r["model"] for r in fig}
    assert any(m.startswith("nonWR") for m in labels) and any(m.startswith("WR") for m in labels)
    assert (d / "fig3b_holdout.png").exists() and (d / "fig3b_holdout.pdf").exists()
    # the control network is judged not weakly reversible in stage ①
    v = json.loads((full_run / "1_verdict" / "verdict.json").read_text(encoding="utf-8"))
    assert v["networks"]["control"]["weakly_reversible"] is False
