"""Validation of networks without the 149 guarantee (not weakly reversible): only conservation and non-negativity
decide; positivity, the equilibrium relation and the Lyapunov decrease are reported but do not fail the run."""

import numpy as np

from enzopt.core.network import Network, x0_vector
from enzopt.core.simval import run_validated, scipy_solver
from enzopt.core.validate import validate_closed


def uni_irreversible() -> Network:
    return Network.from_strings("uni_irr", ["E", "S", "ES", "P"], ["E + S <=> ES", "ES -> E + P"], ["E", "ES"])


def test_extinction_passes_without_guarantee_and_fails_with_it():
    """S is used up: positivity and the equilibrium relation of step 1 cannot hold, yet the run is sound."""
    net = uni_irreversible()
    k = net.rate_vector([100.0, 10.0], [50.0, 0.0])
    t = np.linspace(0.0, 3000.0, 301)
    vr_with = run_validated(
        "x", scipy_solver(net, k, x0_vector(net, {"S": 1.0}, 1e-3), t, 1e-3), net.species, net.reactions(), k, 1.0
    )
    vr_without = run_validated(
        "x",
        scipy_solver(net, k, x0_vector(net, {"S": 1.0}, 1e-3), t, 1e-3),
        net.species,
        net.reactions(),
        k,
        1.0,
        guarantee=False,
    )
    assert vr_with.report["conservation_ok"] and vr_without.report["conservation_ok"]
    assert not vr_with.ok and vr_with.retried  # the 149 checks flag it (and re-run it) although nothing is wrong
    assert vr_without.ok and not vr_without.retried
    row = vr_without.row()
    assert row["guarantee_checks"] is False and row["equilibrium_ok"] is False  # still reported


def test_real_faults_still_fail_without_guarantee():
    net = uni_irreversible()
    rx = net.reactions()
    k = net.rate_vector([100.0, 10.0], [50.0, 0.0])
    t = np.linspace(0.0, 10.0, 11)

    def traj(s_end: float) -> dict:
        """S falls from 1 to s_end, P = 1 − S (S + ES + P conserved), enzyme free."""
        S = np.linspace(1.0, s_end, 11)
        return {"E": np.full(11, 1e-3), "S": S, "ES": np.zeros(11), "P": 1.0 - S}

    assert validate_closed(t, traj(0.5), rx, k, guarantee=False)["ok"]
    drift = dict(traj(0.5), P=traj(0.5)["P"] * 1.01)  # S + ES + P not conserved
    assert not validate_closed(t, drift, rx, k, guarantee=False)["ok"]
    r = validate_closed(t, traj(-1e-3), rx, k, guarantee=False)  # conserved but clearly negative
    assert r["conservation_ok"] and r["nonnegative_ok"] is False and not r["ok"]
    r = validate_closed(t, traj(-1e-20), rx, k, guarantee=False)  # round-off at extinction
    assert r["positivity_ok"] is False and r["ok"]
