"""`enzopt suggest` for any network (pipeline.suggest_general): candidates from the structure, least squares."""

import json

import numpy as np
import pytest

from enzopt.core.massaction import MassAction, simulate
from enzopt.core.network import Network, x0_vector

TRUTH_STEPS = ["E + S <=> ES", "ES -> E + P", "E + P <=> E_P"]  # product inhibition, K_i = 0.05 mM
KF = [100.0, 10.0, 1000.0]
KR = [50.0, 0.0, 50.0]
ET = 1e-3  # mM
SIGMA = 0.002


def write_project(root, holdout=True):
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
        "      - {overall: S <=> P}\n"
        "  prior: {kf_log10: [2, 1], kr_log10: [2], sd_decades: 1.5}\n"
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
    assert top["k_new"]["kr"] / top["k_new"]["kf"] == pytest.approx(0.05, rel=0.3)  # K_i
    assert (out / "candidates.csv").exists() and (out / "candidates.png").exists()
