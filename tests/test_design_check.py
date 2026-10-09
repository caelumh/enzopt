"""AC-18: design-condition prediction curves (scipy over all samples + Hopsan representative) in one figure, the
maximum difference recorded, confirmation-experiment points overlaid. Long tier."""

import csv
import json

import pytest

pytestmark = pytest.mark.long


def test_design_check_curves(full_run):
    d = full_run / "5_design"
    s = json.loads((d / "design_summary.json").read_text(encoding="utf-8"))
    dc = s["design_check"]
    assert dc["hopsan"] is not None, dc.get("hopsan_error")
    assert dc["hopsan"]["max_abs_diff_mM"] is not None
    with open(d / "fig5c_design_check.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    models = {r["model"] for r in rows}
    seeds = {r["id"] for r in rows if r["model"] == "scipy robust"}
    assert len(seeds) == len(s["samples"])  # every sample
    assert "hopsan robust" in models
    assert {"experiment D1", "experiment D0"} <= models  # confirmation experiments as points
    for ext in ("png", "pdf"):
        assert (d / f"fig5c_design_check.{ext}").exists()
