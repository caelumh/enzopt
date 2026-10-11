"""`enzopt suggest`: missing-step search (pipeline.suggest, PINN options discrepancy rate_factor / dead_end)."""

import json
import subprocess
import sys

import numpy as np
from conftest import ROOT, run_cli

from enzopt.pipeline.suggest import CANDIDATES_LDH, _prior_ext, match_seed, summarize

MISSING = ROOT / "projects" / "example_ldh_missing_step"


def _points(Ki=5.0, n=400, seed=0):
    """Collocation-like states where the true shortfall is a dead-end EN·Lac complex."""
    rng = np.random.default_rng(seed)
    pts = {f"phi_{x}": rng.uniform(0.0, 0.5, n).tolist() for x in ("E", "EN", "ENP", "EQL", "EQ")}
    pts.update({"Lac": rng.uniform(0.0, 30.0, n).tolist(), "Pyr": rng.uniform(0.0, 0.5, n).tolist()})
    pts["g"] = (-np.log1p(np.asarray(pts["phi_EN"]) * np.asarray(pts["Lac"]) / Ki)).tolist()
    return pts


def test_shape_match_recovers_the_dead_end_step():
    res = [match_seed(_points(seed=s)) for s in range(3)]
    summ = summarize(res)
    top = summ["candidates"][0]
    assert top["name"] == "EN·Lac" and top["r2_median"] > 0.99 and top["top_in_seeds"] == "3/3"
    assert abs(top["Ki_mM_median"] - 5.0) / 5.0 < 0.05


def test_prior_of_extended_network_matches_theta_layout():
    from enzopt.core.project import load_project
    from enzopt.pipeline.compare import theta_to_rates

    p = load_project(MISSING)
    net = p.network.with_extra_steps(["EN + Lac <=> ENL"], extra_species=["ENL"])
    prior = _prior_ext(p, 5.0)
    assert len(prior) == 6 + 5 + 1
    kf, kr = theta_to_rates(prior, net, p.haldane_step)
    assert kf[-1] == 1e3 and abs(kr[-1] / kf[-1] - 5.0) < 1e-9
    assert len(_prior_ext(p, None)) == 10


def test_candidates_are_dead_end_complexes():
    for _name, step, form, lig, cplx in CANDIDATES_LDH:
        assert step == f"{form} + {lig} <=> {cplx}"


def test_suggest_finds_the_hidden_step(tmp_path):
    """Data made with EN + Lac <=> ENL (K_i 5 mM): the five steps fail and that candidate is suggested."""
    r = run_cli("suggest", MISSING, "--method", "pinn", "--quick", "--verify", "0", "--runs-root", tmp_path / "runs")
    assert r.returncode == 0, r.stdout[-3000:]
    out = next((tmp_path / "runs").rglob("suggest.json"))
    rep = json.loads(out.read_text(encoding="utf-8"))
    v = rep["verdict"]
    assert not v["base_explains_data"]
    assert v["suggested_step"] == "EN + Lac <=> ENL"
    assert rep["pinn_ranking"][0]["name"] == "EN·Lac"
    assert (out.parent / "fig_suggest.png").exists()


def test_suggest_does_not_invent_a_step(tmp_path):
    """The same conditions made with the five steps only: nothing is suggested."""
    neg = tmp_path / "neg"
    subprocess.run([sys.executable, str(MISSING / "make_data.py"), "--without-step", str(neg)], check=True, cwd=ROOT)
    r = run_cli("suggest", neg, "--method", "pinn", "--quick", "--verify", "0", "--runs-root", tmp_path / "runs")
    assert r.returncode == 0, r.stdout[-3000:]
    rep = json.loads(next((tmp_path / "runs").rglob("suggest.json")).read_text(encoding="utf-8"))
    assert rep["verdict"]["suggested_step"] is None


def test_decide_reports_the_holdout_rule():
    """Q-02 (2026-10-10): the training rule decides; the held-out rule (the candidate must also lower the held-out
    RMSE of the five steps) is reported next to it."""
    from enzopt.pipeline.suggest import decide

    s = 0.003
    step = "EN + Lac <=> ENL"
    rank = [
        {"name": "다섯 단계", "step": None, "train_rmse_median": 10 * s, "holdout_rmse_median": 8 * s},
        {"name": "EN·Lac", "step": step, "train_rmse_median": s, "holdout_rmse_median": s, "Ki_mM_median": 4.0},
    ]

    def ver(ho_top):
        return [
            {"step": None, "train_rmse": 10 * s, "holdout_rmse": 8 * s, "rmse": {"A": 10 * s, "B": 10 * s}},
            {"step": step, "train_rmse": s, "holdout_rmse": ho_top, "rmse": {"A": s, "B": s}, "Ki_mM_fit": 5.0},
        ]

    good = decide(rank, ver(s), ["A", "B"], s)
    assert good["suggested_step"] == good["suggested_by_holdout"] == step and good["holdout_improves"]
    bad = decide(rank, ver(9 * s), ["A", "B"], s)
    assert bad["suggested_step"] == step  # the training rule is unchanged
    assert bad["suggested_by_holdout"] is None and bad["holdout_improves"] is False
    assert "홀드아웃 기준으로는 제안 안 함" in bad["note"]
