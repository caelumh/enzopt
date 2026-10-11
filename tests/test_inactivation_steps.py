"""Inactivation as a step of the network (X -> ED): quasi-steady structure and the PINN's active fraction a(t)."""

import numpy as np
import pytest

from enzopt.core.massaction import MassAction, simulate
from enzopt.core.network import Network, x0_vector
from enzopt.core.qss import QssError, qss_fractions, structure

KF = np.array([100.0, 10.0, 0.01])  # E + S → ES 1/(mM·s), ES → E + P 1/s, ES → ED 1/s
KR = np.array([50.0, 0.0, 0.0])
ET = 1e-3  # mM


def uni_dead(enzyme=("E", "ES", "ED"), extra=("ES -> ED",)) -> Network:
    steps = ["E + S <=> ES", "ES -> E + P", *extra]
    return Network.from_strings("uni_dead", ["E", "S", "ES", "P", "ED"], steps, list(enzyme))


def test_structure_puts_dead_form_last_and_out_of_the_graph():
    qs = structure(uni_dead(enzyme=("E", "ED", "ES")), ["S <=> P"])
    assert qs.enzyme == ["E", "ES", "ED"]
    assert (qs.n_dead, qs.n_active) == (1, 2)
    assert [e.step for e in qs.inact_edges] == [2]
    assert all(e.a < 2 and e.b < 2 for e in qs.edges)
    # S = 1 mM: ES/E = kf1·S / (kr1 + kf2) = 100 / 60
    phi = qss_fractions(qs, np.array([1.0, 0.0]), KF, KR)
    assert phi == pytest.approx([60 / 160, 100 / 160, 0.0])


def test_two_dead_forms_are_refused():
    net = Network.from_strings(
        "two_dead",
        ["E", "S", "ES", "P", "ED", "ED2"],
        ["E + S <=> ES", "ES -> E + P", "ES -> ED", "E -> ED2"],
        ["E", "ES", "ED", "ED2"],
    )
    with pytest.raises(QssError, match="실활 형태는 하나만"):
        structure(net, ["S <=> P"])


def test_inactivation_step_must_not_change_small_molecules():
    with pytest.raises(QssError, match="작은 분자"):
        structure(uni_dead(extra=("ES -> ED + P",)), ["S <=> P"])


def test_pinn_learns_the_inactivation_step():
    """Synthetic data from the full ODE (40–70 % of the enzyme dead by 300 s): the PINN sample re-solved as an ODE fits
    the data at noise level and puts the inactivation constant near the truth."""
    from enzopt.pinn.general import GeneralPinn, PinnConfig, PinnExperiment

    net = uni_dead()
    model = MassAction.from_network(net)
    k = net.rate_vector(KF, KR)
    t = np.linspace(5.0, 300.0, 40)
    rng = np.random.default_rng(0)
    sigma = 0.002
    exps, data = [], []
    for name, s0 in (("a", 0.5), ("b", 1.0), ("c", 2.0)):
        X = simulate(model, k, x0_vector(net, {"S": s0}, ET), t)
        idx = net.index()
        y = X[:, idx["S"]] + X[:, idx["ES"]] + rng.normal(0.0, sigma, len(t))
        exps.append(PinnExperiment(name, {"S": s0}, ET, t, y, extra=[]))
        data.append(({"S": s0}, y))
        assert X[-1, idx["ED"]] / ET > 0.4  # the enzyme really dies in the data
    cfg = PinnConfig(
        seeds=[0],
        adam_iters=2000,
        lbfgs_iters=300,
        sigma=sigma,
        device="cpu",
        dtype="float64",
        cuda_graphs=False,
        network=net,
        observable={"S": 1.0, "ES": 1.0},
        overall=["S <=> P"],
    )
    m = GeneralPinn(exps, np.array([2.0, 1.0, -2.0, 2.0]), cfg)
    assert m.mech and m.ne_act == 2
    r = m.train()[0]
    kr_ = net.rate_vector(r.kf, r.kr)
    res = []
    for initial, y in data:
        X = simulate(model, kr_, x0_vector(net, initial, ET), t)
        res.append(X[:, net.index()["S"]] + X[:, net.index()["ES"]] - y)
    rmse = float(np.sqrt(np.mean(np.concatenate(res) ** 2)))
    assert rmse < 2 * sigma
    assert abs(np.log10(r.kf[2] / KF[2])) < 0.5
