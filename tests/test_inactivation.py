"""Enzyme inactivation a(t): the activity descriptions (core.activity), the ODE with enzyme loss, the PINN option
`identification.pinn.inactivation: shared`, and `enzopt run` with it (projects/example_ldh_inactivation)."""

import json

import numpy as np
import pytest
from conftest import EXAMPLE, ROOT, manifest, run_cli, run_dir_from

from enzopt.core.activity import activity_curve, activity_rate, enzyme_loss

INACT = ROOT / "projects" / "example_ldh_inactivation"


def _steps(seed=0, K=12, T=900.0):
    w = np.random.default_rng(seed).uniform(0.0, 0.8, K)
    return {"kind": "steps", "T": T, "c": np.linspace(0.0, 1.0, K).tolist(), "s": 0.6 / (K - 1), "w": w.tolist()}


def test_activity_starts_at_one_and_never_rises():
    t = np.linspace(0.0, 900.0, 400)
    for act in (_steps(), {"kind": "hill", "t50": 300.0, "n": 3.0}, {"kind": "exp", "ki": 1e-3}):
        a = activity_curve(act, t)
        assert a[0] == pytest.approx(1.0) and np.all(np.diff(a) <= 1e-12) and np.all(a > 0)
    assert np.all(activity_curve(None, t) == 1.0) and np.all(activity_rate(None, t) == 0.0)
    legacy = {k: v for k, v in _steps().items() if k != "kind"}  # PINN dicts without "kind" are steps
    assert np.allclose(activity_curve(legacy, t), activity_curve(_steps(), t))


@pytest.mark.parametrize("act", [_steps(1), {"kind": "hill", "t50": 300.0, "n": 3.0}, {"kind": "exp", "ki": 2e-3}])
def test_activity_rate_is_minus_dlog_a(act):
    t = np.linspace(5.0, 895.0, 60)
    h = 1e-3
    num = -(np.log(activity_curve(act, t + h)) - np.log(activity_curve(act, t - h))) / (2 * h)
    assert np.allclose(activity_rate(act, t), num, rtol=1e-5, atol=1e-10)


def test_enzyme_total_follows_activity():
    """With the loss term the enzyme forms add up to Et·a(t) while the reaction runs."""
    from enzopt.core.massaction import MassAction, simulate
    from enzopt.core.network import x0_vector
    from enzopt.core.project import load_project

    p = load_project(EXAMPLE)
    net = p.network
    k = net.rate_vector([1e4, 5e3, 300.0, 1000.0, 200.0], [50.0, 500.0, 1000.0, 200.0, 1e3])
    act = {"kind": "hill", "t50": 300.0, "n": 3.0}
    Et = 24.23e-6
    t = np.linspace(1.0, 900.0, 50)
    loss = enzyme_loss(net.species, net.enzyme_species, act)
    X = simulate(MassAction.from_network(net), k, x0_vector(net, p.experiments["A"].initial, Et), t, 1e-15, loss=loss)
    idx = net.index()
    total = sum(X[:, idx[s]] for s in net.enzyme_species)
    assert np.allclose(total, Et * activity_curve(act, t), rtol=1e-4)


def test_pinn_with_inactivation_trains_and_reports_activity():
    """A few iterations on the CPU: the option runs end to end and returns one a(t) per seed."""
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
        assert r.activity is not None and r.activity["kind"] == "steps" and len(r.activity["w"]) == 6
        assert len(r.theta) == 10 + 6 and np.isfinite(r.kf).all()
        a = activity_curve(r.activity, [0.0, 450.0, 895.0])
        assert a[0] == 1.0 and a[2] <= a[1] <= 1.0
    plain = train_pinn(exps, _prior(p), PinnConfig(seeds=[0], adam_iters=10, lbfgs_iters=2, device="cpu"))
    assert plain[0].activity is None and len(plain[0].theta) == 10


def test_run_with_inactivation_option(tmp_path):
    """`enzopt run` on data with enzyme inactivation: with the option the samples carry a(t) (truth a(300 s) = 0.5),
    the activity figure is drawn and the robust design holds on every sample."""
    r = run_cli("run", INACT, "--quick", "--no-cache", "--runs-root", tmp_path / "runs")
    rd = run_dir_from(r.stdout)
    m = manifest(rd)
    assert m["status"] == "completed", m.get("error")
    fs = json.loads((rd / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    assert fs["inactivation"] == "shared" and len(fs["kept"]) >= 2
    for seed in fs["kept"]:
        s = json.loads((rd / "3_fit" / "samples" / f"sample_seed{seed}.json").read_text(encoding="utf-8"))
        assert s["activity"]["kind"] == "steps"
        assert 0.3 < float(activity_curve(s["activity"], 300.0)) < 0.7
    assert (rd / "3_fit" / "fig3e_activity.png").exists()
    d = json.loads((rd / "5_design" / "design_summary.json").read_text(encoding="utf-8"))
    assert d["robust_fail"] == 0


def test_same_data_without_the_option_stops_before_design(tmp_path):
    """Ignoring the inactivation, no sample reproduces the data within 3σ: identification blocks the design."""
    import shutil

    proj = tmp_path / "p"
    shutil.copytree(INACT, proj)
    y = (proj / "project.yaml").read_text(encoding="utf-8").replace("inactivation: shared", "inactivation: none")
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    r = run_cli("run", proj, "--quick", "--no-cache", "--runs-root", tmp_path / "runs", "--stop-after", "3_fit")
    rd = run_dir_from(r.stdout)
    fs = json.loads((rd / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    assert fs["inactivation"] == "none" and fs["kept"] == [] and fs["design_blocked"]
