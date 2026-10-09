"""AC-07: every simulation passes the CRNT checks; a failing run is flagged "수치 오류 의심" and re-run once
at Δt/10."""

import csv

import numpy as np
import pytest
from conftest import LDH_ENZYME, LDH_SPECIES, LDH_STEPS

from enzopt.core.massaction import MassAction, haldane_kr
from enzopt.core.network import Network, x0_vector
from enzopt.core.regression import one_step_network, sample_class
from enzopt.core.simval import SUSPECT, explicit_solver, run_validated, scipy_solver
from enzopt.core.validate import TOL_CONSERVATION, TOL_EQUILIBRIUM_LOG, class_consistency, conserved_values

Et = 24.23e-6


def rate_sets(n=4, seed=11):
    """Truth constants and Haldane-consistent perturbations (±0.5 decades)."""
    kf0 = np.array([1e4, 5e3, 300.0, 1000.0, 200.0])
    kr0 = np.array([50.0, 500.0, 1000.0, 200.0, 1.0])
    rng = np.random.default_rng(seed)
    out = [(kf0, haldane_kr(kf0, kr0, 23.0, 5))]
    for _ in range(n - 1):
        kf = kf0 * 10 ** rng.uniform(-0.5, 0.5, 5)
        kr = kr0 * 10 ** rng.uniform(-0.5, 0.5, 5)
        out.append((kf, haldane_kr(kf, kr, 23.0, 5)))
    return out


def test_all_full_mechanism_runs_pass_crnt_checks():
    net = Network.from_strings("ldh", LDH_SPECIES, LDH_STEPS, LDH_ENZYME)
    rxns = net.reactions()
    model = MassAction.from_network(net)
    t = np.linspace(0, 30000, 601)  # long enough for the slowest perturbed set to reach equilibrium
    rng = np.random.default_rng(5)
    states = sample_class(rng, 12) + [{"NADH": 0.2, "NAD": 0.0, "Pyr": 0.5, "Lac": 9.5}]
    n = 0
    for kf, kr in rate_sets():
        k = net.rate_vector(kf, kr)
        finals, cons = [], []
        for i, st in enumerate(states):
            x0 = x0_vector(net, st, Et)
            vr = run_validated(f"run{i}", scipy_solver(net, k, x0, t, Et, model=model), net.species, rxns, k, 1.0)
            r = vr.report
            assert vr.ok, r
            assert r["conservation_max_rel_drift"] < TOL_CONSERVATION
            assert r["min_concentration"] > 0
            assert r["equilibrium_max_log_error"] < TOL_EQUILIBRIUM_LOG
            assert r["lyapunov_ok"] is True
            finals.append(vr.X[-1])
            cons.append(conserved_values(vr.X, rxns, net.species))
            n += 1
        cc = class_consistency(finals, cons)
        assert len(cc) == 2 and max(cc.values()) < 1e-6  # same class → same final state
    assert n == 4 * 13


def test_failing_run_is_flagged_and_rerun_at_dt_over_10():
    one = one_step_network()
    k = one.rate_vector([2.0], [2.0 / 23.0])
    model = MassAction.from_network(one)
    x0 = np.array([0.2, 0.0, 0.2, 9.8])
    with np.errstate(all="ignore"):
        vr = run_validated(
            "coarse",
            explicit_solver(model, k, x0, t_end=60.0, n_samples=400),
            one.species,
            one.reactions(),
            k,
            step=1.3,
        )
    assert vr.retried is True
    assert SUSPECT in vr.notes and SUSPECT == "수치 오류 의심"
    assert vr.first_report["ok"] is False
    assert vr.step == pytest.approx(0.13)
    assert vr.ok, vr.report
    row = vr.row()
    assert row["suspect_numerical_error"] and row["retried_with_step_div10"] and row["first_ok"] is False


def test_validator_detects_violations():
    from enzopt.core.validate import validate_closed

    one = one_step_network()
    rx = one.reactions()
    k = one.rate_vector([2.0], [2.0 / 23.0])
    t = np.linspace(0, 10, 11)
    base = {"NADH": np.full(11, 0.15), "NAD": np.full(11, 0.05), "Pyr": np.full(11, 0.15), "Lac": np.full(11, 9.85)}
    bad = dict(base, NADH=base["NADH"] * np.linspace(1, 1.01, 11))  # total drifts
    assert validate_closed(t, bad, rx, k)["conservation_ok"] is False
    neg = {s: v.copy() for s, v in base.items()}
    neg["Pyr"][5] = -1e-6
    assert validate_closed(t, neg, rx, k)["positivity_ok"] is False


def test_pipeline_validation_table(quick_run):
    with open(quick_run / "4_sim" / "validation.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows
    for r in rows:
        assert r["ok"] == "True"
        assert float(r["conservation_max_rel_drift"]) < TOL_CONSERVATION
        assert float(r["min_concentration"]) > 0
        assert float(r["equilibrium_max_log_error"]) < TOL_EQUILIBRIUM_LOG
        assert r["lyapunov_ok"] == "True"
        assert "suspect_numerical_error" in r
