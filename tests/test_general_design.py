"""F-13 ⑤ design on general networks (EVAL.md AC-27..AC-30), on rate-constant samples around the appendix C truths.

AC-27 assay with one, two and three substrates + the numerical equilibrium limit; AC-28 yield; AC-29 selectivity
(competing substrates); AC-30 time to target. "Exhaustive search" = a fine grid around the answer (spacing < 1 %).
"""

from concurrent.futures import ProcessPoolExecutor

import numpy as np
import pytest
from conftest import EVALS, ROOT

from enzopt.core.design import (
    AssayDesign,
    equilibrium_conversion,
    equilibrium_conversion_of,
    min_Et,
    rho_min_equilibrium,
    rho_min_of,
    run_design,
    run_time_design,
)
from enzopt.core.yamlio import load_yaml

GENERAL = ROOT / "projects" / "general"


def truth_samples(key: str, n: int = 4, spread: float = 0.15, seed: int = 11):
    """Network and n rate vectors: the truth and n−1 with the free log10 constants shaken by N(0, spread)."""
    from enzopt.synth import general_truth

    tr = general_truth(load_yaml(EVALS / "synthetic_general.yaml"), key)
    lay, net = tr["layout"], tr["network"]
    Kp = [k or 1.0 for k in tr["Kp"]]
    th0 = lay.theta(tr["kf"], tr["kr"], Kp)
    rng = np.random.default_rng(seed)
    out = []
    for i in range(n):
        th = th0 + (rng.normal(0.0, spread, len(th0)) if i else 0.0)
        n_, nf = lay.n_steps, lay.n_free_kr
        th[n_ + nf :] = th0[n_ + nf :]  # K′ kept: the design uses the known K′
        kf, kr = lay.rates(th)
        out.append(net.rate_vector(kf, kr))
    return net, tr, out


def _ok(job):
    design, Et, rho, samples = job
    return all(design.meets(k, Et, rho) for k in samples)


def exhaustive_min_Et(design, rho, samples, around: float, factor: float = 1.25, n: int = 61) -> float:
    """Smallest Et on a log grid of n points within around/factor … around·factor that meets the goal for all."""
    grid = np.geomspace(around / factor, around * factor, n)
    with ProcessPoolExecutor(max_workers=8) as ex:
        ok = list(ex.map(_ok, [(design, float(e), rho, samples) for e in grid]))
    assert ok[-1] and not ok[0], "the answer is not inside the search window"
    return float(grid[ok.index(True)])


# ------------------------------------------------------------------------------------------------ AC-27
def test_assay_equilibrium_limit_matches_ldh_formula():
    from enzopt.core.network import Network
    from enzopt.core.regression import DESIGN_ENZYME, DESIGN_SPECIES, DESIGN_STEPS

    net = Network.from_strings("d", DESIGN_SPECIES, DESIGN_STEPS, DESIGN_ENZYME)
    d = AssayDesign(net, overall="NADH + Pyr <=> Lac + NAD", Kp=23.0, conversion=0.99)
    assert rho_min_of(d) == pytest.approx(5.251, rel=1e-3)
    assert rho_min_of(d) == pytest.approx(rho_min_equilibrium(0.99, 23.0), rel=1e-3)
    for r in (1.0, 1.5, 3.0, 8.0):
        assert equilibrium_conversion_of(d, r) == pytest.approx(equilibrium_conversion(r, 23.0), rel=1e-3)


@pytest.mark.parametrize(
    "key, partner, fixed, conv",
    [("uni", None, {}, 0.75), ("ping_pong", "B", {}, 0.99), ("ordered_terter", "B", {"C": 5.0}, 0.99)],
)
def test_assay_one_two_three_substrates(key, partner, fixed, conv):
    from enzopt.pipeline.design import pipette_table

    net, tr, samples = truth_samples(key)
    analyte = "S" if key == "uni" else "A"
    d = AssayDesign(net, analyte, partner, "E", 0.1, conv, 300.0, tr["Kp"][0], fixed=fixed, overall=tr["overall"][0])
    rmin = rho_min_of(d)
    if partner is None:
        assert rmin is None
        assert equilibrium_conversion_of(d, 1.0) == pytest.approx(0.8, rel=1e-6)  # S ⇌ P, K′ = 4
        rhos = [1.0]
    else:
        assert np.isfinite(rmin) and equilibrium_conversion_of(d, rmin) == pytest.approx(conv, abs=1e-9)
        rhos = [rmin * 1.1, rmin * 2, rmin * 4]
    res = run_design(d, samples, rhos, iters=14, Et_bounds=(1e-7, 1e-2), workers=4)
    s = res.summary()
    assert np.isfinite(res.Et_star) and s["robust_fail"] == 0
    stocks = {analyte: 2.0, **({partner: 50.0} if partner else {}), **{k: 50.0 for k in fixed}}
    des = {
        "pipette": {"volume_uL": 1000.0, "enzyme_uL": 10.0, "stocks_mM": stocks, "limit_test_rhos": [1.5]},
        "analyte0_mM": 0.1,
        "time_s": 300.0,
    }
    des["pipette"]["limit_test_enzyme_multiples"] = [[10]]
    pip = pipette_table(des, res, 1e-5, d, use_limit=True)
    sets = {r["set"] for r in pip}
    assert {"D1", "D0", "Z"} <= sets and all(r["buffer_uL"] > 0 for r in pip)
    if partner:
        assert "L1" in sets and any(k.startswith(f"{partner}_stock") for k in pip[0])
    for sp in fixed:
        assert pip[0][f"{sp}_stock_uL"] > 0


def test_assay_no_limit_when_deficiency_positive():
    from enzopt.core.crnt import verdict
    from enzopt.pipeline.design import limit_reason

    net, tr, _ = truth_samples("random_bibi", n=1)
    vd = verdict(net.reactions(), "random_bibi")
    why = limit_reason(vd, 23.0, tr["overall"][0], 1)
    assert why and "결핍 2" in why
    net, tr, _ = truth_samples("ping_pong", n=1)
    assert limit_reason(verdict(net.reactions(), "pp"), 23.0, tr["overall"][0], 1) is None


# ------------------------------------------------------------------------------------------------ AC-28
def test_yield_goal_matches_exhaustive_search():
    net, tr, samples = truth_samples("ping_pong")
    d = AssayDesign(
        net, "A", "B", "E", 0.1, 0.99, 300.0, 23.0, goal="yield", product="P", yield_min=0.95, overall=tr["overall"][0]
    )
    rho = 10.0
    Et = min_Et(d, rho, samples, 1e-7, 1e-2, iters=24)
    assert np.isfinite(Et)
    assert all(d.evaluate(k, Et, rho)["yield"] >= 0.95 for k in samples)
    assert Et == pytest.approx(exhaustive_min_Et(d, rho, samples, Et), rel=0.02)
    res = run_design(d, samples, [5.0, 10.0, 20.0], iters=14, Et_bounds=(1e-7, 1e-2), workers=3)
    assert res.summary()["robust_fail"] == 0 and all("yield" in c for c in res.check)


# ------------------------------------------------------------------------------------------------ AC-29
def test_selectivity_goal_matches_exhaustive_search():
    net, tr, samples = truth_samples("ldh_competing")
    d = AssayDesign(
        net,
        "NADH",
        "Pyr",
        "E",
        0.1,
        0.99,
        300.0,
        23.0,
        fixed={"Oxb": 1.0},
        goal="selectivity",
        product="Lac",
        side_product="Hbt",
        yield_min=0.6,
        selectivity_min=0.8,
        overall=tr["overall"][0],
    )
    # at a low pyruvate ratio the side product wins whatever the enzyme amount; at a high one the goal is met
    assert np.isnan(min_Et(d, 0.5, samples, 1e-7, 1e-2, iters=12))
    rho = 8.0
    Et = min_Et(d, rho, samples, 1e-7, 1e-2, iters=24)
    assert np.isfinite(Et)
    ev = [d.evaluate(k, Et, rho) for k in samples]
    assert all(v["selectivity"] >= 0.8 and v["yield"] >= 0.6 for v in ev)
    assert Et == pytest.approx(exhaustive_min_Et(d, rho, samples, Et), rel=0.02)
    res = run_design(d, samples, [0.5, 4.0, 8.0, 16.0], iters=14, Et_bounds=(1e-7, 1e-2), workers=4)
    assert res.rho_star >= 4.0 and res.summary()["robust_fail"] == 0


# ------------------------------------------------------------------------------------------------ AC-30
def _times(job):
    d, k, Et, rho, t_max = job
    t = np.linspace(0.0, t_max, 40001)
    y = d.trajectory(k, Et, rho, t)[:, d.network.index()[d.analyte]]
    hit = np.nonzero(y <= d.target)[0]
    return float(t[hit[0]]) if len(hit) else float("inf")


def test_time_goal_matches_exhaustive_search():
    net, tr, samples = truth_samples("ping_pong")
    d = AssayDesign(net, "A", "B", "E", 0.1, 0.99, 300.0, 23.0, overall=tr["overall"][0])
    rhos = [6.0, 9.0, 13.0, 20.0, 30.0]
    Et, t_max = 2e-5, 3600.0
    res = run_time_design(d, samples, rhos, Et, t_max, workers=5)
    jobs = [(d, k, Et, r, t_max) for r in rhos for k in samples]
    with ProcessPoolExecutor(max_workers=8) as ex:
        ts = list(ex.map(_times, jobs))
    worst = [max(ts[i * len(samples) : (i + 1) * len(samples)]) for i in range(len(rhos))]
    assert res.rho_star == rhos[int(np.argmin(worst))]
    assert res.time_star == pytest.approx(min(worst), rel=0.01)
