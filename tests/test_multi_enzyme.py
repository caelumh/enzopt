"""Networks with several enzymes (core.qss, pinn.general): each connected graph of enzyme forms is one enzyme with its
own amount (Et for the first, the free form's initial value for the others); the quasi-steady distribution is solved per
graph and weighted by each enzyme's share."""

import numpy as np
import pytest

from enzopt.core.network import Network
from enzopt.core.qss import QssError, qss_fractions, structure

STEPS = ["E + A <=> EA", "EA <=> E + B", "F + B <=> FB", "FB <=> F + C"]
KF = np.array([100.0, 20.0, 50.0, 5.0])
KR = np.array([30.0, 1.0, 10.0, 2.0])


def cascade():
    return Network.from_strings("cascade", ["E", "A", "EA", "B", "F", "FB", "C"], STEPS, ["E", "EA", "F", "FB"])


def two_state(kon, koff, kcat, krev, s, p):
    """Free and bound fraction of a uni-uni enzyme E + S ⇌ ES ⇌ E + P at fixed S, P."""
    to_bound, to_free = kon * s + krev * p, koff + kcat
    return np.array([to_free, to_bound]) / (to_bound + to_free)


def test_two_enzymes_are_two_graphs_with_their_own_amounts():
    qs = structure(cascade())
    assert qs.n_enzymes == 2 and qs.refs == [0, 2] and qs.comp == [0, 0, 1, 1] and qs.d == 2
    tot = qs.totals({"F": 2e-3}, 1e-3)
    assert tot.tolist() == [1e-3, 2e-3]
    f = tot / tot.sum()
    x = {"A": 1.0, "B": 0.4, "C": 0.2}
    phi = qss_fractions(qs, np.array([x[s] for s in qs.small]), KF, KR, f)
    ref = np.r_[
        f[0] * two_state(KF[0], KR[0], KF[1], KR[1], x["A"], x["B"]),
        f[1] * two_state(KF[2], KR[2], KF[3], KR[3], x["B"], x["C"]),
    ]
    assert phi == pytest.approx(ref, rel=1e-12) and phi.sum() == pytest.approx(1.0)


def test_one_enzyme_is_unchanged():
    net = Network.from_strings("one", ["E", "A", "EA", "B"], STEPS[:2], ["E", "EA"])
    qs = structure(net)
    assert qs.n_enzymes == 1 and qs.refs is None and qs.totals({}, 2e-3).tolist() == [2e-3]


def test_inactivation_with_two_enzymes_is_refused():
    net = Network.from_strings(
        "dead", ["E", "A", "EA", "B", "F", "FB", "C", "ED"], [*STEPS, "EA -> ED"], ["E", "EA", "F", "FB", "ED"]
    )
    with pytest.raises(QssError):
        structure(net)


def test_pinn_distribution_matches_the_numpy_reference():
    torch = pytest.importorskip("torch")
    from enzopt.core.thermo import layout
    from enzopt.pinn.general import GeneralPinn, PinnConfig, PinnExperiment

    net = cascade()
    t = np.linspace(0.0, 600.0, 11)
    exps = [
        PinnExperiment("x0", {"A": 1.0, "F": 2e-3}, 1e-3, t, np.zeros_like(t)),
        PinnExperiment("x1", {"C": 1.5, "F": 1e-3}, 3e-3, t, np.zeros_like(t)),
    ]
    lay = layout(net, ["A <=> C"])
    prior = np.zeros(lay.n_theta)
    cfg = PinnConfig(seeds=[0], sigma=0.01, network=net, observable={"A": 1.0}, overall=["A <=> C"], device="cpu")
    g = GeneralPinn(exps, prior, cfg)
    kf = torch.tensor(KF, dtype=torch.float64).expand(g.B, -1).reshape(g.B, 1, -1)
    kr = torch.tensor(KR, dtype=torch.float64).expand(g.B, -1).reshape(g.B, 1, -1)
    x = np.array([0.7, 0.3, 0.2])  # A, B, C
    phi = g.qss_fractions(torch.tensor(np.tile(x, (g.B, 1)).reshape(g.B, 1, -1), dtype=torch.float64), kf, kr)
    for b, e in enumerate(exps):
        tot = g.qs.totals(e.initial, e.Et)
        ref = qss_fractions(g.qs, x, KF, KR, tot / tot.sum())
        assert phi[b, 0].numpy() == pytest.approx(ref, rel=1e-10)
        e0 = np.zeros(4)
        e0[[0, 2]] = tot / tot.sum()  # all of each enzyme free at t = 0
        assert g.e0[b, 0].numpy() == pytest.approx(e0)
        assert float(g.Et[b, 0, 0]) == pytest.approx(tot.sum())
