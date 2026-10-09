"""AC-12: the standard figure set — every figure as PDF + PNG (300 dpi) + CSV, CSV values equal to the result files."""

import csv
import json
import struct

import numpy as np
import pytest

STANDARD = {
    "1_verdict": ["fig1a_network", "fig1b_verdict_table"],
    "2_data": ["fig2_data"],
    "3_fit": ["fig3a_train_fit", "fig3b_holdout", "fig3c_rate_constants"],
    "4_sim": ["fig4a_classes", "fig4b_validation", "fig4c_hopsan_vs_scipy"],
    "5_design": ["fig5a_u_curve", "fig5b_conversion", "fig5c_design_check", "fig5d_equilibrium_limit"],
}


def png_dpi(path):
    data = path.read_bytes()
    assert data[:8] == b"\x89PNG\r\n\x1a\n"
    i = 8
    while i < len(data):
        n = struct.unpack(">I", data[i : i + 4])[0]
        kind = data[i + 4 : i + 8]
        if kind == b"pHYs":
            ppx, ppy, unit = struct.unpack(">IIB", data[i + 8 : i + 17])
            assert unit == 1
            return ppx * 0.0254, ppy * 0.0254
        i += 12 + n
    return None


def rows(path):
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


@pytest.mark.parametrize("stage", list(STANDARD))
def test_standard_figures_exist(quick_run, stage):
    d = quick_run / stage
    listed = {f["name"] for f in json.loads((d / "figures.json").read_text(encoding="utf-8"))}
    for name in STANDARD[stage]:
        assert name in listed
        for ext in ("pdf", "png", "csv"):
            assert (d / f"{name}.{ext}").stat().st_size > 0, f"{stage}/{name}.{ext}"
        assert (d / f"{name}.pdf").read_bytes()[:4] == b"%PDF"
        dpi = png_dpi(d / f"{name}.png")
        assert dpi is not None and abs(dpi[0] - 300) < 1 and abs(dpi[1] - 300) < 1
        assert rows(d / f"{name}.csv"), f"{name}.csv is empty"


def test_csv_values_match_results(quick_run):
    d = quick_run / "5_design"
    u = rows(d / "fig5a_u_curve.csv")
    t = rows(d / "design_table.csv")
    assert len(u) == len(t)
    for a, b in zip(u, t, strict=True):
        assert float(a["rho"]) == pytest.approx(float(b["rho"]), rel=1e-9)
        assert float(a["Et_robust_nM"]) == pytest.approx(float(b["Et_robust_nM"]), rel=1e-9)
        assert float(a["Et_nominal_nM"]) == pytest.approx(float(b["Et_nominal_nM"]), rel=1e-9)
    conv = rows(d / "fig5b_conversion.csv")
    chk = rows(d / "design_check.csv")
    assert [float(r["conversion"]) for r in conv] == pytest.approx([float(r["conversion"]) for r in chk], rel=1e-9)
    eq = rows(d / "fig5d_equilibrium_limit.csv")
    lim = rows(d / "equilibrium_limit.csv")
    assert [float(r["conversion_eq"]) for r in eq] == pytest.approx([float(r["conversion_eq"]) for r in lim], rel=1e-9)
    # ② data figure = the checked data
    fd = rows(quick_run / "2_data" / "fig2_data.csv")
    a = [float(r["value"]) for r in fd if r["experiment"] == "A" and r["quantity"] == "A340"]
    exp = np.loadtxt(quick_run / "2_data" / "exp_A.csv", delimiter=",", skiprows=1)
    assert np.allclose(a, exp[:, 1], rtol=1e-9)
    # ③ rate constants = the sample files
    rc = rows(quick_run / "3_fit" / "fig3c_rate_constants.csv")
    s0 = json.loads((quick_run / "3_fit" / "samples" / "sample_seed0.json").read_text(encoding="utf-8"))
    got = [float(r["log10_k"]) for r in rc if r["model"] == "seed0"]
    assert got == pytest.approx(list(np.log10(s0["kf"] + s0["kr"])), rel=1e-9)
    # ④ validation table = validation.csv
    vt = rows(quick_run / "4_sim" / "fig4b_validation.csv")
    vv = rows(quick_run / "4_sim" / "validation.csv")
    assert sum(int(r["pass"]) for r in vt) == sum(r["ok"] == "True" for r in vv)
