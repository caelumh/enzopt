"""AC-16: classical fitting baseline (same ODE, multi-start least squares) vs PINN samples — one table (CSV + figure)
of held-out extrapolation errors on the realistic synthetic problem. Long tier."""

import csv
import json

import pytest

pytestmark = pytest.mark.long


def test_baseline_vs_pinn_table(full_run):
    d = full_run / "3_fit"
    f = json.loads((d / "fit_summary.json").read_text(encoding="utf-8"))
    cl = f["comparisons"]["classical"]
    assert len(cl["starts"]) >= 2  # multi-start
    fits = [x for x in cl["fits"] if "kf" in x]
    assert len(fits) >= 2
    with open(d / "holdout_comparison.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    models = {r["model"] for r in rows}
    assert {"classical", "PINN (WR)"} <= models
    for m in ("classical", "PINN (WR)"):
        for exp in ("C", "Dp"):
            vals = [float(r["rmse_A340"]) for r in rows if r["model"] == m and r["experiment"] == exp]
            assert vals and all(v == v and v >= 0 for v in vals)
    for ext in ("csv", "pdf", "png"):
        assert (d / f"fig3d_holdout_table.{ext}").exists()
    with open(d / "fig3d_holdout_table.csv", encoding="utf-8") as fh:
        t = list(csv.DictReader(fh))
    assert {r["model"] for r in t} >= {"classical", "PINN (WR)"}
