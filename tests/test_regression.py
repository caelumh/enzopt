"""AC-01 (149 class batch) and AC-02 (synthetic-sample design) against the CADKIMM baselines (evals/baselines.json)."""

import pytest

from enzopt.core.regression import batch149_scipy, design_regression


def _check_batch(res, ref):
    tol = ref["final_tolerance"]
    assert res["n_runs"] == 13
    assert res["n_pass"] == 13, {k: v for k, v in res["per_run"].items() if not v["ok"]}
    assert res["same_class_final_NADH"] == pytest.approx(ref["same_class_final"]["NADH"], abs=tol)
    assert res["same_class_spread"] < ref["same_class_final_spread_max"]
    assert res["other_class_final_NADH"] == pytest.approx(ref["other_class_final"]["NADH"], abs=tol)
    assert res["n_classes"] == 2


@pytest.mark.parametrize(
    "backend",
    ["scipy", pytest.param("hopsan", marks=pytest.mark.long)],
)
def test_batch149(backend, base, tmp_path):
    ref = base["batch149"]
    if backend == "scipy":
        res = batch149_scipy()
    else:
        from enzopt.hopsan.batch import batch149_hopsan

        res = batch149_hopsan(tmp_path / "batch149")
    _check_batch(res, ref)


@pytest.mark.parametrize("mode", ["quick", pytest.param("full", marks=pytest.mark.long)])
def test_design(mode, base):
    ref = base["design"][mode]
    res = design_regression(quick=(mode == "quick"))
    rel = ref["tolerance_rel"]
    assert res["rho_min"] == pytest.approx(base["design"]["rho_min"], rel=1e-9)
    assert res["rho_star"] == pytest.approx(ref["rho_star"], rel=1e-9)
    assert res["robust_Et_star_nM"] == pytest.approx(ref["robust_Et_star_nM"], rel=rel)
    assert res["nominal_Et_at_rho_star_nM"] == pytest.approx(ref["nominal_Et_at_rho_star_nM"], rel=rel)
    assert res["nominal_fail"] == ref["nominal_fail"]
    assert res["nominal_worst_conversion"] == pytest.approx(
        ref["nominal_worst_conversion"], abs=ref["tolerance_worst_conversion"]
    )
    assert res["robust_fail"] == ref["robust_fail"] == 0
    for got, want in zip(res["table"], ref["table"], strict=True):
        assert got["rho"] == pytest.approx(want["rho"], rel=1e-6)
        assert got["Et_robust_nM"] == pytest.approx(want["Et_robust_nM"], rel=rel)
        assert got["Et_nominal_nM"] == pytest.approx(want["Et_nominal_nM"], rel=rel)
    # equilibrium limit table (CRNT algebra, independent of rate constants)
    for got, want in zip(res["equilibrium_limit"], base["design"]["equilibrium_limit"], strict=True):
        assert got["conversion_eq"] == pytest.approx(want["conversion_eq"], rel=1e-9)
