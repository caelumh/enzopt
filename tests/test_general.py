"""F-13 general networks (EVAL.md AC-21..AC-26, AC-31, AC-32; appendix C networks in projects/general/).

Quick: AC-21 renamed LDH end to end, AC-22 several signals, AC-23 thermodynamic constraints.
"""

import json
import re
import shutil

import numpy as np
import pytest
from conftest import EVALS, EXAMPLE, ROOT, manifest, run_cli, run_dir_from

from enzopt.core.yamlio import load_yaml

GENERAL = ROOT / "projects" / "general"
RENAME = {"NADH": "A", "NAD": "Q", "Pyr": "B", "Lac": "P", "EN": "EA", "ENP": "EAB", "EQL": "EQP", "EQP": "EQB"}


# ------------------------------------------------------------------------------------------------ AC-21
def renamed_ldh(tmp_path):
    """The example project with every LDH name replaced (NADH → A, Pyr → B, ...) and the observable called 'Abs'."""
    proj = tmp_path / "projects" / "ldh_renamed"
    shutil.copytree(EXAMPLE, proj)
    pat = re.compile(r"\b(" + "|".join(sorted(RENAME, key=len, reverse=True)) + r")\b")
    y = (proj / "project.yaml").read_text(encoding="utf-8")
    y = pat.sub(lambda m: RENAME[m.group(1)], y)
    y = y.replace("name: A340", "name: Abs").replace("name: example_ldh_synth", "name: ldh_renamed")
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    for f in (proj / "data").glob("set*.csv"):
        if not f.name.endswith("_temp.csv"):
            f.write_text(f.read_text(encoding="utf-8").replace("time_s,A340", "time_s,Abs", 1), encoding="utf-8")
    return proj


def test_renamed_ldh_runs_like_ldh(quick_run, tmp_path):
    proj = renamed_ldh(tmp_path)
    r = run_cli("run", proj, "--quick", "--runs-root", tmp_path / "runs")
    rd = run_dir_from(r.stdout)
    m = manifest(rd)
    assert m["status"] == "completed", m.get("error")
    # ③ the same samples: training RMSE per seed within 1 %
    a = json.loads((quick_run / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    b = json.loads((rd / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    assert a["seeds"] == b["seeds"] and a["kept"] == b["kept"]
    for s in a["seeds"]:
        sa = json.loads((quick_run / "3_fit" / "samples" / f"sample_seed{s}.json").read_text(encoding="utf-8"))
        sb = json.loads((rd / "3_fit" / "samples" / f"sample_seed{s}.json").read_text(encoding="utf-8"))
        assert sb["rmse_ode_train"] == pytest.approx(sa["rmse_ode_train"], rel=0.01)
    # ⑤ the same design
    da = json.loads((quick_run / "5_design" / "design_summary.json").read_text(encoding="utf-8"))
    db = json.loads((rd / "5_design" / "design_summary.json").read_text(encoding="utf-8"))
    assert db["rho_star"] == pytest.approx(da["rho_star"], rel=0.01)
    assert db["robust_Et_star_nM"] == pytest.approx(da["robust_Et_star_nM"], rel=0.01)
    assert "Abs" in (rd / "2_data" / "data.json").read_text(encoding="utf-8")


# ------------------------------------------------------------------------------------------------ AC-22
def two_signal_project(tmp_path):
    """The competing-substrate project (A340 every 10 s + Hbt by HPLC every 60 s in its own file), plus a third
    signal 'NADHf' written as an extra column of the main CSV (same times)."""
    proj = tmp_path / "projects" / "two_signals"
    shutil.copytree(GENERAL / "ldh_competing", proj)
    y = (proj / "project.yaml").read_text(encoding="utf-8")
    y = y.replace(
        "    - {name: Hbt_mM, coefficients: {Hbt: 1}, sigma: 0.005}\n",
        "    - {name: Hbt_mM, coefficients: {Hbt: 1}, sigma: 0.005}\n"
        "    - {name: NADHf, coefficients: {NADH: 1}, sigma: 0.004}\n",
    )
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    for f in (proj / "data").glob("set??.csv"):
        lines = f.read_text(encoding="utf-8").splitlines()
        out = [lines[0] + ",NADHf"] + [ln + f",{float(ln.split(',')[1]) / 6.22:.6f}" for ln in lines[1:]]
        f.write_text("\n".join(out) + "\n", encoding="utf-8")
    return proj


def test_multi_signal_read_and_errors(tmp_path):
    proj = two_signal_project(tmp_path)
    r = run_cli("check", proj)
    assert r.returncode == 0, r.stdout[-2000:]
    from enzopt.core.project import load_project

    p = load_project(proj)
    assert [s["name"] for s in p.all_signals()] == ["A340", "Hbt_mM", "NADHf"]
    # a missing column, an empty value and time going backwards are named with the file and line
    f = proj / "data" / "setT1_Hbt_mM.csv"
    good = f.read_text(encoding="utf-8")
    bad = good.splitlines()
    bad[3] = bad[3].split(",")[0] + ","
    f.write_text("\n".join(bad) + "\n", encoding="utf-8")
    r = run_cli("check", proj)
    assert r.returncode != 0 and "setT1_Hbt_mM.csv" in r.stdout + r.stderr and "4번째 줄" in r.stdout + r.stderr
    bad = good.splitlines()
    bad[3], bad[4] = bad[4], bad[3]
    f.write_text("\n".join(bad) + "\n", encoding="utf-8")
    r = run_cli("check", proj)
    assert "시간 역행" in r.stdout + r.stderr and "setT1_Hbt_mM.csv" in r.stdout + r.stderr
    f.write_text(good, encoding="utf-8")
    m = proj / "data" / "setT2.csv"
    m.write_text(m.read_text(encoding="utf-8").replace(",NADHf", ",other", 1), encoding="utf-8")
    r = run_cli("check", proj)
    assert r.returncode != 0 and "setT2.csv" in r.stdout + r.stderr and "NADHf" in r.stdout + r.stderr
    # a signal_files entry must name a declared signal
    y = (
        (proj / "project.yaml")
        .read_text(encoding="utf-8")
        .replace("signal_files: {Hbt_mM:", "signal_files: {Lac_mM:", 1)
    )
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    with pytest.raises(ValueError, match="signal_files"):
        load_project(proj)


def test_multi_signal_loss_uses_each_sigma(tmp_path):
    """The PINN data loss is the mean squared σ-normalised residual over every point of every signal."""
    import torch

    from enzopt.core.project import load_project
    from enzopt.pinn.general import GeneralPinn, PinnExperiment
    from enzopt.pipeline.fit import pinn_config

    proj = two_signal_project(tmp_path)
    p = load_project(proj)
    exps = []
    for name, e in p.experiments.items():
        if e.role != "train":
            continue
        d = np.loadtxt(e.file, delimiter=",", skiprows=1)
        extra = []
        for sg in p.signals:
            f = e.signal_files.get(sg["name"])
            if f is not None:
                ds = np.loadtxt(f, delimiter=",", skiprows=1)
                extra.append((sg["name"], ds[:, 0], ds[:, 1], sg["sigma"], sg["coefficients"]))
            else:
                extra.append((sg["name"], d[:, 0], d[:, 2], sg["sigma"], sg["coefficients"]))
        exps.append(PinnExperiment(name, e.initial, e.Et, d[:, 0], d[:, 1], extra=extra))
    cfg = pinn_config(p, p.identification, [0], "cpu", "float64", 0.003)
    m = GeneralPinn(exps, p.prior_vector(), cfg)
    n_pts = [len(x.t_obs) + sum(len(s[1]) for s in x.extra) for x in exps]
    assert m.n_obs.reshape(-1).tolist() == n_pts
    sig = sorted({float(v) for v in m.s_obs[m.mask > 0].reshape(-1).tolist()})
    assert sig == pytest.approx([0.003, 0.004, 0.005])
    with torch.no_grad():
        kf, kr = m.rates(m.theta)
        n = m.n_steps
        kfB = kf.unsqueeze(1).expand(m.S, m.E, n).reshape(m.B, 1, n)
        krB = kr.unsqueeze(1).expand(m.S, m.E, n).reshape(m.B, 1, n)
        X = m.states(m.tau_obs, kfB, krB)[0]
        pred = (X * m.coef_obs).sum(-1)
        manual = []
        for b in range(m.B):
            k = int(m.n_obs[b].item())
            r = (m.a_obs[b, :k, 0] - pred[b, :k]) / m.s_obs[b, :k, 0]
            manual.append(float((r**2).mean()))
        _, parts = m.loss(1.0)
    assert float(parts["data"][0]) == pytest.approx(float(np.mean(manual)), rel=1e-9)


# ------------------------------------------------------------------------------------------------ AC-23
def _general_networks():
    syn = load_yaml(EVALS / "synthetic_general.yaml")
    return list(syn["networks"])


@pytest.mark.parametrize("key", _general_networks())
def test_cycle_constraints(key):
    """The rate constants built by the tool satisfy every cycle condition (overall cycles: product = K′, internal
    cycles: 1, relative error 1e-9); the truths of appendix C are reproduced from their free constants."""
    from enzopt.core.project import load_project
    from enzopt.synth import general_truth

    syn = load_yaml(EVALS / "synthetic_general.yaml")
    tr = general_truth(syn, key)
    lay = tr["layout"]
    Kp = [k or 1.0 for k in tr["Kp"]]
    kf, kr = lay.rates(lay.theta(tr["kf"], tr["kr"], Kp))
    assert np.allclose(kr, tr["kr"], rtol=1e-9, atol=0)
    assert np.all(np.abs(lay.residuals(kf, kr, Kp)) < 1e-9)
    # random θ: the dependent k₋ always close every cycle
    rng = np.random.default_rng(3)
    for _ in range(5):
        th = lay.theta(tr["kf"], tr["kr"], Kp) + rng.normal(0, 1.0, lay.n_theta)
        kf2, kr2 = lay.rates(th)
        Kp2 = list(Kp)
        n, nf = lay.n_steps, lay.n_free_kr
        for c, o in enumerate(lay.overall_used):
            Kp2[o] = 10 ** th[n + nf + c]
        assert np.all(np.abs(lay.residuals(kf2, kr2, Kp2)) < 1e-9)
    # the generated project uses the same layout
    p = load_project(GENERAL / key)
    assert p.layout.dependent == lay.dependent and p.layout.overall_used == lay.overall_used


def test_cycle_constraints_ldh_matches_haldane_step():
    from enzopt.core.massaction import haldane_kr
    from enzopt.core.project import load_project

    p = load_project(EXAMPLE)
    assert [i + 1 for i in p.layout.dependent] == [5]
    syn = load_yaml(EVALS / "synthetic.yaml")["truth"]
    kf = np.array(syn["kf"], float)
    kr = np.array(syn["kr"] + [1.0], float)
    kr_h = haldane_kr(kf, kr, 23.0, 5)
    kf2, kr2 = p.layout.rates(p.layout.theta(kf, kr_h, [23.0]))
    assert np.allclose(kr2, kr_h, rtol=1e-12)
