"""PINN option `inactivation: shared`: the learned enzyme-activity curve a(t) (scripts/inactivation_demo.py)."""

import numpy as np

from enzopt.pinn.ldh import activity_curve, activity_rate


def _act(seed=0, K=12, T=900.0):
    w = np.random.default_rng(seed).uniform(0.0, 0.8, K)
    return {"T": T, "c": np.linspace(0.0, 1.0, K).tolist(), "s": 0.6 / (K - 1), "w": w.tolist()}


def test_activity_starts_at_one_and_never_rises():
    act = _act()
    t = np.linspace(0.0, 900.0, 400)
    a = activity_curve(act, t)
    assert a[0] == 1.0
    assert np.all(np.diff(a) <= 1e-12) and np.all(a > 0)
    assert np.all(activity_curve(None, t) == 1.0) and np.all(activity_rate(None, t) == 0.0)


def test_activity_rate_is_minus_dlog_a():
    act = _act(seed=1)
    t = np.linspace(5.0, 895.0, 60)
    h = 1e-3
    num = -(np.log(activity_curve(act, t + h)) - np.log(activity_curve(act, t - h))) / (2 * h)
    assert np.allclose(activity_rate(act, t), num, rtol=1e-5, atol=1e-10)


def test_pinn_with_inactivation_trains_and_reports_activity():
    """A few iterations on the CPU: the option runs end to end and returns one a(t) per seed."""
    from conftest import EXAMPLE

    from enzopt.core.dataio import read_series
    from enzopt.core.project import load_project
    from enzopt.pinn.ldh import PinnConfig, PinnExperiment, train_pinn
    from enzopt.pipeline.fit import _prior

    p = load_project(EXAMPLE)
    exps = []
    for e in p.by_role("train"):
        s = read_series(e.file)
        exps.append(PinnExperiment(e.name, e.initial, e.Et, s.t, s.y))
    cfg = PinnConfig(seeds=[0, 1], adam_iters=10, lbfgs_iters=2, inactivation="shared", inact_k=6, device="cpu")
    res = train_pinn(exps, _prior(p), cfg)
    assert len(res) == 2
    for r in res:
        assert r.activity is not None and len(r.activity["w"]) == 6
        assert len(r.theta) == 10 + 6 and np.isfinite(r.kf).all()
        a = activity_curve(r.activity, [0.0, 450.0, 895.0])
        assert a[0] == 1.0 and a[2] <= a[1] <= 1.0
    plain = train_pinn(exps, _prior(p), PinnConfig(seeds=[0], adam_iters=10, lbfgs_iters=2, device="cpu"))
    assert plain[0].activity is None and len(plain[0].theta) == 10
