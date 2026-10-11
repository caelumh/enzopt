"""`enzopt suggest` for any network (pipeline.suggest_general): candidates from the structure, least squares."""

import json

import numpy as np
import pytest
from conftest import ROOT

from enzopt.core.massaction import MassAction, simulate
from enzopt.core.network import Network, x0_vector

TRUTH_STEPS = ["E + S <=> ES", "ES -> E + P", "E + P <=> E_P"]  # product inhibition, K_i = 0.05 mM
KF = [100.0, 10.0, 1000.0]
KR = [50.0, 0.0, 50.0]
ET = 1e-3  # mM
SIGMA = 0.002


def write_project(root, holdout=True, kp=None):
    """A Michaelis–Menten project whose data come from the network with the product-inhibition step."""
    truth = Network.from_strings("truth", ["E", "S", "ES", "P", "E_P"], TRUTH_STEPS, ["E", "ES", "E_P"])
    model = MassAction.from_network(truth)
    k = truth.rate_vector(KF, KR)
    t = np.linspace(10.0, 600.0, 60)
    rng = np.random.default_rng(1)
    (root / "data").mkdir(parents=True)
    exps = {"T1": {"S": 0.2}, "T2": {"S": 0.5}, "T3": {"S": 1.0}}
    if holdout:
        exps["H1"] = {"S": 0.5, "P": 0.5}
    lines = []
    for name, init in exps.items():
        X = simulate(model, k, x0_vector(truth, init, ET), t)
        idx = truth.index()
        y = X[:, idx["S"]] + X[:, idx["ES"]] + rng.normal(0.0, SIGMA, len(t))
        rows = "\n".join(f"{a:g},{b:.6f}" for a, b in zip(t, y, strict=True))
        (root / "data" / f"{name}.csv").write_text("time_s,S_mM\n" + rows + "\n", encoding="utf-8")
        ini = ", ".join(f"{s}: {v}" for s, v in init.items())
        role = "holdout" if name.startswith("H") else "train"
        lines.append(f"  {name}: {{file: data/{name}.csv, {ini}, Et_nM: 1000, role: {role}}}")
    (root / "project.yaml").write_text(
        "name: gen_suggest\n"
        "network:\n"
        "  species: [E, S, ES, P]\n"
        "  enzyme_species: [E, ES]\n"
        "  steps: [E + S <=> ES, ES -> E + P]\n"
        "  observable: {name: S_mM, coefficients: {S: 1, ES: 1}}\n"
        "  thermodynamics:\n"
        "    reactions:\n"
        + (f"      - {{overall: S <=> P, Kp: {kp:g}}}\n" if kp else "      - {overall: S <=> P}\n")
        + "  prior: {kf_log10: [2, 1], kr_log10: [2], sd_decades: 1.5}\n"
        "experiments:\n" + "\n".join(lines) + "\n"
        f"identification: {{noise_sigma: {SIGMA}, baseline_fit: {{starts: 2, max_nfev: 60}}}}\n"
        "simulation: {hopsan: {enabled: false}}\n"
        "design: {enabled: false}\n",
        encoding="utf-8",
    )
    return root


def test_candidates_from_the_structure(tmp_path):
    from enzopt.core.project import load_project
    from enzopt.pipeline.suggest_general import candidates

    p = load_project(write_project(tmp_path / "proj"))
    steps = {c.kind: [] for c in candidates(p)}
    for c in candidates(p):
        steps[c.kind].append(c.step)
    # E already binds S (step 1); E + P only appears as a product complex of a one-way step, so binding P is a candidate
    assert steps["dead_end"] == ["E + P <=> E_P", "ES + S <=> ES_S", "ES + P <=> ES_P"]
    assert steps["inactivation"] == ["E -> ED", "ES -> ED"]
    assert steps["background"] == ["S -> P"]
    # S <=> P has no K′: making ES -> E + P reversible would close a cycle without it, so no "reverse" candidate
    assert "reverse" not in steps and "order" not in steps


def test_reverse_and_order_candidates(tmp_path):
    """With K′ a one-way step may become reversible (its k₋ then follows from K′); two reversible bindings in a row
    give the other binding order (LDH: pyruvate before NADH, lactate before NAD⁺)."""
    from enzopt.core.project import load_project
    from enzopt.pipeline.suggest_general import candidates, setup

    p = load_project(write_project(tmp_path / "proj", kp=10))
    rev = [c for c in candidates(p) if c.kind == "reverse"]
    assert [c.step for c in rev] == ["ES <=> E + P"] and rev[0].replace == {1: "ES <=> E + P"}
    _net, lay, prior = setup(p, rev[0])
    assert 1 in lay.dependent and len(prior) == lay.n_theta  # k₋ of the reversed step from K′: no new parameter
    ldh = load_project(ROOT / "projects" / "example_ldh_missing_step")
    order = sorted(c.step for c in candidates(ldh, ("order",)))
    assert order == ["E + Lac <=> E_Lac ; E_Lac + NAD <=> EQL", "E + Pyr <=> E_Pyr ; E_Pyr + NADH <=> ENP"]


def test_general_suggest_finds_the_missing_step(tmp_path):
    from enzopt.pipeline.suggest import run_suggest

    proj = write_project(tmp_path / "proj")
    out = run_suggest(proj, runs_root=tmp_path / "runs", log=lambda *_: None)
    rep = json.loads((out / "suggest.json").read_text(encoding="utf-8"))
    v = rep["verdict"]
    assert rep["method"] == "ode" and rep["start"] == "fresh fit"
    assert not v["base_explains_data"]
    assert v["suggested_step"] == "E + P <=> E_P"
    assert v["suggested_holdout_improves"] is True and v["best_holdout_step"] == "E + P <=> E_P"
    top = rep["candidates"][0]
    assert top["step"] == "E + P <=> E_P" and top["train"] < 1.5
    assert top["k_new"]["kr"][0] / top["k_new"]["kf"][0] == pytest.approx(0.05, rel=0.3)  # K_i
    assert (out / "candidates.csv").exists() and (out / "candidates.png").exists()


MISSING = ROOT / "projects" / "example_ldh_missing_step"


@pytest.mark.long
def test_general_path_on_ldh_finds_the_hidden_step(tmp_path):
    """The default (general) path on the LDH example with a hidden E·NADH·Lac complex (K_i 5 mM): the same answer as
    the PINN path (EN + Lac, training 23σ → 1.0σ, held out 2.5σ → 0.9σ)."""
    from enzopt.pipeline.suggest import run_suggest

    out = run_suggest(MISSING, runs_root=tmp_path / "runs", log=lambda *_: None)
    rep = json.loads((out / "suggest.json").read_text(encoding="utf-8"))
    v = rep["verdict"]
    assert v["suggested_step"] == "EN + Lac <=> EN_Lac" and v["suggested_holdout_improves"]
    top = rep["candidates"][0]
    assert top["train"] < 1.3 and top["holdout"] < 1.3
    assert 2.5 < top["k_new"]["kr"][0] / top["k_new"]["kf"][0] < 10.0  # K_i, truth 5 mM


@pytest.mark.long
def test_general_path_on_ldh_does_not_invent_a_step(tmp_path):
    import subprocess
    import sys

    from enzopt.pipeline.suggest import run_suggest

    neg = tmp_path / "neg"
    subprocess.run([sys.executable, str(MISSING / "make_data.py"), "--without-step", str(neg)], check=True, cwd=ROOT)
    out = run_suggest(neg, runs_root=tmp_path / "runs", log=lambda *_: None)
    v = json.loads((out / "suggest.json").read_text(encoding="utf-8"))["verdict"]
    assert v["base_explains_data"] and v["suggested_step"] is None and v["holdout_halving_steps"] == []


CEPH = ROOT / "projects" / "aeh_cephalexin_lagerman2021"


def test_user_hypotheses_and_removals(tmp_path):
    """``--candidate`` text (step N replaced, new enzyme forms and small molecules, priors by step shape) and
    ``--kinds remove`` (drop an enzyme form with its steps); a removal must still simulate every signal even when a
    signal sums the removed form (it then counts as 0)."""
    from enzopt.core.project import load_project
    from enzopt.pipeline.suggest_general import (
        K_INACT_LOG10,
        KF_BIND_LOG10,
        _z,
        candidates,
        load_all,
        setup,
        user_candidate,
    )

    p = load_project(CEPH)
    c = user_candidate(p, "2: ES -> EA + MeOH; E + MeOH <=> E_MeOH")
    assert c.replace == {1: "ES -> EA + MeOH"} and c.steps == ["E + MeOH <=> E_MeOH"] and c.new_enzyme == ["E_MeOH"]
    assert c.prior_kf == [KF_BIND_LOG10] and c.step == "2: ES -> EA + MeOH ; E + MeOH <=> E_MeOH"
    net, lay, prior = setup(p, c)
    assert "E_MeOH" in net.enzyme_species and "MeOH" in net.species and "MeOH" not in net.enzyme_species
    assert len(prior) == lay.n_theta
    assert user_candidate(p, "EG -> ED").prior_kf == [K_INACT_LOG10]  # ED is left by no step: inactivation
    with pytest.raises(ValueError):
        user_candidate(p, "20: E -> ED")

    rem = candidates(p, ("remove",))
    assert [r.name for r in rem] == ["remove EAN", "remove EN", "remove EG", "remove ESS", "remove EAS"]
    assert rem[3].remove == [9, 12] and rem[3].drop_species == ["ESS", "ED"]  # ESS goes, and with it its dead form
    m0 = next(x for x, _ in load_all(p, tmp_path) if x[0] == "m0")
    for r in rem:
        net, lay, prior = setup(p, r)
        assert not set(r.drop_species) & set(net.species)
        kf, kr = lay.rates(prior)
        z = _z(net, net.rate_vector(kf, kr), m0)
        assert len(z) == len(m0[3]) and all(np.all(np.isfinite(a)) for a in z)
