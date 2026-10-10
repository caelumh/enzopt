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


# ------------------------------------------------------------------------------------------------ long tier
RING = ["uni_irreversible", "uni", "uni_iso", "uni_bi_ordered", "theorell_chance", "ping_pong", "ordered_terter"]
BRANCHED = ["random_bibi", "ldh_dead_end"]
MULTI = ["ldh_competing", "ldh_background"]


@pytest.fixture(scope="session")
def general_run(tmp_path_factory):
    """Full-mode run through ③ (seeds 4, restarts 8, comparison models) of one appendix C network, once per session."""
    root = tmp_path_factory.mktemp("runs_general")
    cache: dict = {}

    def get(key: str):
        if key not in cache:
            r = run_cli("run", GENERAL / key, "--no-cache", "--stop-after", "3_fit", "--runs-root", root, timeout=3600)
            rd = run_dir_from(r.stdout)
            m = manifest(rd)
            assert m["stages"]["3_fit"]["status"] == "done", (key, m.get("error"))
            cache[key] = rd
        return cache[key]

    return get


def _signals(p, e) -> list[tuple]:
    """[(signal, t, y, sigma, coefficients)] of one experiment, primary first, read from the project files."""
    from enzopt.core.dataio import read_series

    sig0 = float(p.identification["noise_sigma"])
    out = []
    for i, sg in enumerate(p.all_signals(sig0)):
        f = e.signal_files.get(sg["name"], e.file) if i else e.file
        s = read_series(f, value_column=sg["name"], max_value=None)
        out.append((sg["name"], s.t, s.y, float(sg["sigma"]), sg["coefficients"]))
    return out


def oracle_rmse(key: str) -> dict:
    """Holdout RMSE of the reference model per (experiment, signal): the synthetic truth (mechanism and rate
    constants) with one enzyme-amount scale fitted to the training sets by least squares in units of σ."""
    from scipy.optimize import minimize_scalar

    from enzopt.core.project import load_project
    from enzopt.pipeline.common import curve, rmse

    crit = load_yaml(EVALS / "synthetic_general.yaml")["common"]["holdout_criterion"]
    assert crit["oracle"] == "fitted_enzyme_scale"
    p = load_project(GENERAL / key)
    truth = json.loads((GENERAL / key / "data" / "truth.json").read_text(encoding="utf-8"))
    k = p.network.rate_vector(truth["kf"], truth["kr"])
    sig = {n: _signals(p, e) for n, e in p.experiments.items()}

    def errs(name, scale):
        e = p.experiments[name]
        return {s: rmse(curve(p.network, k, e.initial, e.Et * scale, t, cf), y) / sg for s, t, y, sg, cf in sig[name]}

    train = [n for n, e in p.experiments.items() if e.role == "train"]
    lo, hi = crit["oracle_scale_bounds"]
    best = minimize_scalar(
        lambda x: sum(v**2 for n in train for v in errs(n, x).values()), bounds=(lo, hi), method="bounded"
    ).x
    out = {}
    for n, e in p.experiments.items():
        if e.role == "holdout":
            sg = {s: g for s, _t, _y, g, _c in sig[n]}
            out.update({(n, s): v * sg[s] for s, v in errs(n, best).items()})
    return out


def identification_report(key: str, rd) -> dict:
    """Training reproduction and the holdout AC-05 frame (EVAL.md, F-13) of one run."""
    from enzopt.core.project import load_project

    crit = load_yaml(EVALS / "synthetic_general.yaml")["common"]["holdout_criterion"]
    p = load_project(GENERAL / key)
    prim = p.observable["name"]
    sigma = {s["name"]: float(s["sigma"]) for s in p.all_signals()}
    samples = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((rd / "3_fit" / "samples").glob("*.json"))]

    def rm(s, exp):  # {signal: RMSE} of one sample and experiment
        if "rmse_ode_signals" in s:
            return s["rmse_ode_signals"][exp]
        return {prim: s["rmse_ode"][exp]}

    train = [n for n, e in p.experiments.items() if e.role == "train"]
    ok_train = []
    for s in samples:
        vals = [v / sigma[sg] if v is not None else np.inf for n in train for sg, v in rm(s, n).items()]
        ok_train.append(bool(vals) and max(vals) <= float(crit["train_sigma"]))
    oracle = oracle_rmse(key)
    kept = [s for s in samples if s["kept"]]
    hold = {}
    for (exp, sg), o in oracle.items():
        limit = max(float(crit["rmse_sigma"]) * sigma[sg], float(crit["oracle_factor"]) * o)
        vals = [rm(s, exp)[sg] for s in kept]
        vals = [np.inf if v is None else v for v in vals]
        hold[f"{exp}/{sg}"] = {
            "oracle": o,
            "limit": limit,
            "mean": float(np.mean(vals)) if vals else np.inf,
            "worst": float(np.max(vals)) if vals else np.inf,
        }
    rep = {
        "train_ok_fraction": float(np.mean(ok_train)) if ok_train else 0.0,
        "n_samples": len(samples),
        "n_kept": len(kept),
        "holdout": hold,
    }
    print(key, json.dumps(rep, ensure_ascii=False))
    return rep


def check_identification(key: str, rd) -> None:
    crit = load_yaml(EVALS / "synthetic_general.yaml")["common"]["holdout_criterion"]
    rep = identification_report(key, rd)
    assert rep["train_ok_fraction"] >= float(crit["train_fraction"]), (key, "training", rep)
    assert rep["n_kept"] > 0, (key, "no kept samples", rep)
    for name, h in rep["holdout"].items():
        assert h["mean"] <= h["limit"], (key, name, "mean", h)
        assert h["worst"] <= h["limit"], (key, name, "worst", h)


@pytest.mark.long
@pytest.mark.parametrize("key", RING)
def test_ring_identification(general_run, key):
    check_identification(key, general_run(key))


@pytest.mark.long
@pytest.mark.parametrize("key", BRANCHED)
def test_branched_identification(general_run, key):
    check_identification(key, general_run(key))


@pytest.mark.long
@pytest.mark.parametrize("key", MULTI)
def test_multi_reaction_identification(general_run, key):
    check_identification(key, general_run(key))


@pytest.mark.long
@pytest.mark.parametrize("key", RING + BRANCHED + MULTI)
def test_comparisons(general_run, key):
    """AC-26: classical fit and the non-WR control through the same flow, with the AC-16/17 table and figures."""
    import csv

    from enzopt.core.project import load_project

    rd = general_run(key)
    p = load_project(GENERAL / key)
    d = rd / "3_fit"
    f = json.loads((d / "fit_summary.json").read_text(encoding="utf-8"))
    cl = f["comparisons"]["classical"]
    assert len(cl["starts"]) >= 2 and len([x for x in cl["fits"] if "kf" in x]) >= 2
    with open(d / "holdout_comparison.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    holdouts = [n for n, e in p.experiments.items() if e.role == "holdout"]
    models = ["classical", "PINN (WR)"] + (["nonWR"] if p.control else [])
    for m in models:
        for exp in holdouts:
            vals = [float(r["rmse_A340"]) for r in rows if r["model"] == m and r["experiment"] == exp]
            assert vals and all(v == v and v >= 0 for v in vals), (key, m, exp)
    for ext in ("csv", "pdf", "png"):
        assert (d / f"fig3d_holdout_table.{ext}").exists()
    if p.control:
        c = f["comparisons"]["control"]
        assert c["one_way_steps"] == p.control["one_way_steps"] and len(c["fits"]) >= 1, (key, c)
        assert all(x["kr"][s - 1] == 0.0 for x in c["fits"] for s in c["one_way_steps"])
        with open(d / "fig3b_holdout.csv", encoding="utf-8") as fh:
            labels = {r["model"] for r in csv.DictReader(fh)}
        assert any(m.startswith("nonWR") for m in labels) and any(m.startswith("WR") for m in labels)
        v = json.loads((rd / "1_verdict" / "verdict.json").read_text(encoding="utf-8"))
        assert v["networks"]["control"]["weakly_reversible"] is False


@pytest.mark.long
def test_e2e_all(tmp_path):
    """AC-32: every appendix C network and the renamed LDH run ①–⑤ in short mode; non-WR ones carry the band."""
    from enzopt.core.crnt import verdict
    from enzopt.core.project import load_project

    keys = RING + BRANCHED + MULTI
    projects = [GENERAL / k for k in keys] + [renamed_ldh(tmp_path)]
    failed = []
    for proj in projects:
        r = run_cli("run", proj, "--quick", "--runs-root", tmp_path / "runs")
        rd = run_dir_from(r.stdout)
        m = manifest(rd)
        if m["status"] != "completed":
            failed.append((proj.name, m.get("error")))
            continue
        for st in ("1_verdict", "2_data", "3_fit", "4_sim", "5_design"):
            assert m["stages"][st]["status"] in ("done", "cached"), (proj.name, st)
        p = load_project(proj)
        if not verdict(p.network.reactions(), "m").weakly_reversible:
            rep = (rd / "1_verdict" / "report.txt").read_text(encoding="utf-8")
            assert "149 보장 없음" in rep, proj.name
    assert not failed, failed
