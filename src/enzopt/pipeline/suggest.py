"""Missing-step suggestions (``enzopt suggest``): which step would the data like the mechanism to have?

1. **Where do the steps fall short?** The PINN is trained on the training experiments with ``discrepancy:
   rate_factor``: dNADH/dt = −r₁·exp(g(NADH, Pyr, Lac, NAD)). g is one small network per seed, shared by all
   experiments and pulled toward 0, so it takes what the written steps cannot explain (plus the PINN's own small
   misfit, so it is shown as a hint, not used as the test). g is also matched to each candidate's shape: a dead-end
   complex X + Y ⇌ XY multiplies the rate by 1/(1 + φ_X·[Y]/K_i), so exp(−g) − 1 should be ∝ φ_X·[Y].
2. **Which candidate?** Every candidate is put into the PINN in that mechanistic form (``discrepancy: dead_end``, K_i
   learned), and the five steps alone as reference. Each fit is re-solved as an ODE with the extended network (the
   same check as stage ③), and candidates are ranked by the median training RMSE over the seeds. GPU: seconds each.
3. **Check and decide.** The five steps and the best candidates are fitted again by least squares on the ODE
   (multi-start, the stage-③ prior plus K_i centred on the PINN value). The five steps explain the data when their
   training RMSE is within 2σ; otherwise a candidate is suggested when it at least halves the training RMSE. The
   held-out RMSE is reported next to it, and K_i comes from the least-squares fit (the PINN only fixes φ_X/K_i).

Only the LDH ordered bi-bi PINN exists (F-13), so the candidates are the dead-end complexes that an empty substrate
site of E, EN (holds NADH) or EQ (holds NAD⁺) can form with pyruvate or lactate.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

# (name, step, enzyme form X, ligand Y, new complex)
CANDIDATES_LDH = [
    ("EN·Lac", "EN + Lac <=> ENL", "EN", "Lac", "ENL"),
    ("EQ·Pyr", "EQ + Pyr <=> EQP", "EQ", "Pyr", "EQP"),
    ("E·Pyr", "E + Pyr <=> EPy", "E", "Pyr", "EPy"),
    ("E·Lac", "E + Lac <=> ELa", "E", "Lac", "ELa"),
]
KF_DEAD_END = 1e3  # 1/mM/s: binding rate given to a candidate complex (fast against the cycle; only K_i matters)
EVIDENCE_RMS_G = 0.05  # shown only: the PINN g also takes the PINN's own small misfit, so it is a hint, not a test
BASE_OK_SIGMA = 2.0  # the five steps explain the data when their least-squares training RMSE is within 2σ


def _y_and_features(points: dict, cands) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(points["g"], float)
    y = np.expm1(-g)
    F = np.column_stack(
        [np.asarray(points[f"phi_{x}"], float) * np.asarray(points[lig], float) for _, _, x, lig, _ in cands]
    )
    return y, F


def match_seed(points: dict, cands=CANDIDATES_LDH) -> dict:
    """Fit y = exp(−g) − 1 by the candidates' φ_X·[Y], jointly (non-negative) and one at a time."""
    y, F = _y_and_features(points, cands)
    sst = float(np.sum((y - y.mean()) ** 2)) or 1e-300
    c, _ = nnls(F, y)
    joint_r2 = 1.0 - float(np.sum((y - F @ c) ** 2)) / sst
    single = []
    for m in range(F.shape[1]):
        f = F[:, m]
        cm = max(0.0, float(f @ y) / float(f @ f)) if float(f @ f) > 0 else 0.0
        single.append({"coef": cm, "r2": 1.0 - float(np.sum((y - cm * f) ** 2)) / sst})
    g = np.asarray(points["g"], float)
    return {
        "rms_g": float(np.sqrt(np.mean(g**2))),
        "max_abs_g": float(np.max(np.abs(g))),
        "joint_coef": c.tolist(),
        "joint_r2": joint_r2,
        "single": single,
    }


def summarize(per_seed: list[dict], cands=CANDIDATES_LDH) -> dict:
    """Candidates ranked by their median single-candidate R² over the seeds."""
    rows = []
    for m, (name, step, x, lig, cplx) in enumerate(cands):
        r2 = [s["single"][m]["r2"] for s in per_seed]
        coef = [s["single"][m]["coef"] for s in per_seed]
        top = [int(np.argmax([q["r2"] for q in s["single"]]) == m) for s in per_seed]
        Ki = [1.0 / c for c in coef if c > 0]
        rows.append(
            {
                "name": name,
                "step": step,
                "form": x,
                "ligand": lig,
                "complex": cplx,
                "r2_median": float(np.median(r2)),
                "r2_min": float(np.min(r2)),
                "top_in_seeds": f"{sum(top)}/{len(top)}",
                "Ki_mM_median": float(np.median(Ki)) if Ki else None,
            }
        )
    rows.sort(key=lambda r: -r["r2_median"])
    rms = float(np.median([s["rms_g"] for s in per_seed]))
    return {
        "rms_g_median": rms,
        "evidence": rms >= EVIDENCE_RMS_G,
        "joint_r2_median": float(np.median([s["joint_r2"] for s in per_seed])),
        "candidates": rows,
    }


# ------------------------------------------------------------------------------------------------ verification
def _prior_ext(p, Ki_mM: float | None) -> np.ndarray:
    pr = p.prior
    kf = list(pr["kf_log10"])
    kr = list(pr["kr_log10"])
    tail = [pr.get("Kp_log10", np.log10(p.Kp))] if p.haldane_step else []
    if Ki_mM is None:
        return np.array(kf + kr + tail, float)
    return np.array(kf + [np.log10(KF_DEAD_END)] + kr + [np.log10(KF_DEAD_END * Ki_mM)] + tail, float)


def verify(p, train, every, cands: list[dict], starts: int, max_nfev: int, progress=None) -> list[dict]:
    """Classical multi-start fits of the base network and of base + each candidate; RMSE per experiment."""
    from ..core.design import n_workers
    from ..core.thermo import layout
    from .common import rmse, safe_curve
    from .compare import _fit_one, theta_to_rates

    sigma = float(p.identification["noise_sigma"])
    sd = float(p.prior.get("sd_decades", 1.5))
    w_prior = float(p.identification["pinn"].get("w_prior", 1e-2))
    coeff = p.observable_weights()
    nets = [("base (as written)", p.network, None)]
    for c in cands:
        nets.append((c["name"], p.network.with_extra_steps([c["step"]], extra_species=[c["complex"]]), c))
    jobs, meta, lays = [], [], []
    for i, (_label, net, c) in enumerate(nets):
        Ki = (c or {}).get("Ki_mM_median") or 5.0
        prior = _prior_ext(p, Ki if c else None)
        lay = layout(net, p.overall, [p.haldane_step] if p.haldane_step else None)
        lays.append(lay)
        args = (
            net,
            lay,
            [(e.initial, e.Et, t, y, []) for _, e, t, y in train],
            coeff,
            sigma,
            prior,
            w_prior,
            sd,
        )
        for s in range(starts):
            theta0 = prior + np.random.default_rng(100 + s).uniform(-1.0, 1.0, len(prior))
            jobs.append((s, theta0, args, max_nfev))
            meta.append(i)
    w = n_workers(len(jobs))
    with ProcessPoolExecutor(max_workers=w) as ex:
        fits = list(ex.map(_fit_one, jobs))
    out = []
    for i, (label, net, c) in enumerate(nets):
        mine = [f for f, j in zip(fits, meta, strict=True) if j == i and "theta" in f]
        if not mine:
            out.append({"network": label, "error": "모든 시작점 실패"})
            continue
        best = min(mine, key=lambda f: f["cost"])
        kf, kr = theta_to_rates(np.array(best["theta"]), lays[i])
        k = net.rate_vector(kf, kr)
        per = {}
        for name, e, t, y in every:
            a = safe_curve(net, k, e.initial, e.Et, t, coeff)
            per[name] = rmse(a, y) if a is not None else None
        tr = [per[n] for n, e, *_ in every if e.role == "train" and per[n] is not None]
        ho = [per[n] for n, e, *_ in every if e.role == "holdout" and per[n] is not None]
        rec = {
            "network": label,
            "step": c["step"] if c else None,
            "train_rmse": float(np.sqrt(np.mean(np.square(tr)))) if tr else None,
            "holdout_rmse": float(np.sqrt(np.mean(np.square(ho)))) if ho else None,
            "rmse": per,
            "seconds": float(sum(f["seconds"] for f in mine)),
        }
        if c:
            rec["Ki_mM_fit"] = float(kr[-1] / kf[-1])
        out.append(rec)
        if progress:
            progress(label)
    return out


def _recheck(job):
    """Worker: RMSE per experiment of one fit, re-solved as an ODE with its (possibly extended) network."""
    from .common import rmse, safe_curve

    net, kf, kr, every, coeff = job
    k = net.rate_vector(kf, kr)
    out = {}
    for name, e, t, y in every:
        a = safe_curve(net, k, e.initial, e.Et, t, coeff)
        out[name] = rmse(a, y) if a is not None else float("inf")
    return out


def pinn_candidates(p, train, every, cands, base_cfg, log=print) -> list[dict]:
    """The five steps alone and with each candidate, fitted by the PINN; ranked by the ODE-re-solved training RMSE."""
    from dataclasses import replace

    from ..core.design import n_workers
    from ..pinn.ldh import PinnExperiment, train_pinn
    from .fit import _prior

    exps = [PinnExperiment(n, e.initial, e.Et, t, y) for n, e, t, y in train]
    coeff = p.observable_weights()
    train_names = [n for n, *_ in train]
    hold_names = [n for n, e, *_ in every if e.role == "holdout"]
    variants = [("base (as written)", None)] + [(c[0], c) for c in cands]
    fits, jobs = [], []
    for label, c in variants:
        cfg = replace(
            base_cfg,
            discrepancy="dead_end" if c else "none",
            dead_end_form=c[2] if c else "EN",
            dead_end_ligand=c[3] if c else "Lac",
        )
        t0 = time.time()
        res = train_pinn(exps, _prior(p), cfg)
        secs = time.time() - t0
        net = p.network.with_extra_steps([c[1]], extra_species=[c[4]]) if c else p.network
        for r in res:
            kf, kr = list(r.kf), list(r.kr)
            if c:
                kf.append(KF_DEAD_END)
                kr.append(KF_DEAD_END * r.dead_end["Ki_mM"])
            fits.append((label, c, secs, r.seed % 1000, r.dead_end["Ki_mM"] if c else None))
            jobs.append((net, kf, kr, every, coeff))
        log(f"  PINN {label}: {len(res)}회 {secs:.1f} s")
    with ProcessPoolExecutor(max_workers=n_workers(len(jobs))) as ex:
        checks = list(ex.map(_recheck, jobs, chunksize=2))
    out = []
    for label, c in variants:
        best: dict = {}
        secs = 0.0
        for (lab, _c, sec, seed, Ki), chk in zip(fits, checks, strict=True):
            if lab != label:
                continue
            secs = sec
            tr = float(np.sqrt(np.mean([chk[n] ** 2 for n in train_names])))
            ho = float(np.sqrt(np.mean([chk[n] ** 2 for n in hold_names]))) if hold_names else None
            if seed not in best or tr < best[seed]["train_rmse"]:
                best[seed] = {"seed": seed, "train_rmse": tr, "holdout_rmse": ho, "Ki_mM": Ki}
        rows = [best[k] for k in sorted(best)]
        Kis = [r["Ki_mM"] for r in rows if r["Ki_mM"]]
        hos = [r["holdout_rmse"] for r in rows if r["holdout_rmse"] is not None]
        out.append(
            {
                "name": label,
                "step": c[1] if c else None,
                "form": c[2] if c else None,
                "ligand": c[3] if c else None,
                "complex": c[4] if c else None,
                "pinn_seconds": secs,
                "train_rmse_median": float(np.median([r["train_rmse"] for r in rows])),
                "holdout_rmse_median": float(np.median(hos)) if hos else None,
                "Ki_mM_median": float(np.median(Kis)) if Kis else None,
                "per_seed": rows,
            }
        )
    out.sort(key=lambda r: r["train_rmse_median"])
    return out


def decide(rank: list[dict], ver: list[dict], train_names: list[str], sigma: float) -> dict:
    """Do the five steps explain the training data, and if not, does a candidate?

    With the least-squares check: the five steps explain the data when the training RMSE is within 2σ overall and
    every training set is within 3σ. A candidate is suggested when it at least halves the RMSE over the sets the five
    steps miss (all training sets if none is singled out) without raising the overall training RMSE. Without the check,
    the PINN ranking stands in (five steps within 3σ = the stage-③ rule; a candidate must halve the training RMSE).
    K_i comes from least squares when available: the PINN alone only fixes φ_X/K_i."""
    base = next(r for r in rank if r["step"] is None)
    top = next((r for r in rank if r["step"]), None)
    v_base = next((v for v in ver if v.get("step") is None and "error" not in v), None)
    v_top = next((v for v in ver if top and v.get("step") == top["step"] and "error" not in v), None)
    misses = []
    if v_base:
        per = {n: v_base["rmse"][n] for n in train_names}
        worst = max(per, key=per.get)
        misses = [n for n in train_names if per[n] > 3 * sigma]
        base_ok = v_base["train_rmse"] <= BASE_OK_SIGMA * sigma and not misses
        base_txt = f"학습 RMSE {v_base['train_rmse'] / sigma:.1f}σ, 최악 {worst} {per[worst] / sigma:.1f}σ"
    else:
        base_ok = base["train_rmse_median"] <= 3 * sigma
        base_txt = f"PINN 학습 RMSE {base['train_rmse_median'] / sigma:.1f}σ"
    if v_top:
        sets = misses or train_names

        def rms(v):
            return float(np.sqrt(np.mean([v["rmse"][n] ** 2 for n in sets])))

        better = rms(v_top) <= 0.5 * rms(v_base) and v_top["train_rmse"] <= v_base["train_rmse"]
    else:
        better = bool(top and top["train_rmse_median"] < 0.5 * base["train_rmse_median"])
    suggest = (not base_ok) and better
    if base_ok:
        note = f"다섯 단계로 데이터가 설명됨({base_txt}): 빠진 단계의 근거 없음"
    elif suggest:
        note = f"다섯 단계로는 설명되지 않음({base_txt}). {top['name']}을 넣으면 설명됨"
    else:
        note = f"다섯 단계로는 설명되지 않음({base_txt}). 후보 중 맞는 단계 없음(다른 단계·실활·측정 문제를 볼 것)"
    ki_ls = v_top.get("Ki_mM_fit") if v_top else None
    return {
        "base_explains_data": base_ok,
        "base_train_rmse": v_base["train_rmse"] if v_base else base["train_rmse_median"],
        "sets_missed_by_base": misses,
        "suggested_step": top["step"] if suggest else None,
        "Ki_mM": (ki_ls if ki_ls is not None else top["Ki_mM_median"]) if suggest else None,
        "Ki_source": ("least squares" if ki_ls is not None else "PINN (φ_X/K_i only)") if suggest else None,
        "candidate_train_rmse": (v_top["train_rmse"] if v_top else top["train_rmse_median"]) if suggest else None,
        "candidate_holdout_rmse": (v_top["holdout_rmse"] if v_top else top["holdout_rmse_median"]) if suggest else None,
        "note": note,
    }


# ------------------------------------------------------------------------------------------------ command
def run_suggest(project_dir, quick: bool = False, verify_top: int = 1, runs_root=None, log=print) -> Path:
    """``enzopt suggest``: train the discrepancy PINN, match candidates, verify the best ones; write a folder."""
    import tempfile
    from dataclasses import replace

    from .. import plots
    from ..core.dataio import load_experiment
    from ..core.project import load_project
    from ..pinn.ldh import PinnConfig, PinnExperiment, matches_network, resolve_device, train_pinn
    from .fit import _prior

    p = load_project(project_dir)
    if not matches_network(p.network):
        raise SystemExit("빠진 단계 제안은 아직 LDH 순차 bi-bi 반응망만 지원(F-13)")
    ident = p.identification
    obs = p.observable["name"]
    tmp = Path(tempfile.mkdtemp(prefix="enzopt_suggest_"))
    every = []
    for name, e in p.experiments.items():
        if e.role in ("train", "holdout"):
            s, _ = load_experiment(e, converted_dir=tmp, value_column=obs)
            every.append((name, e, s.t, s.y))
    train = [x for x in every if x[1].role == "train"]
    root = Path(runs_root) if runs_root else Path(__file__).resolve().parents[3] / "runs"
    out = root / p.name / (time.strftime("%Y-%m-%d_%H%M") + "_suggest")
    out.mkdir(parents=True, exist_ok=True)
    log(f"결과 폴더: {out}")

    seeds = 2 if quick else int(ident.get("seeds", 4))
    restarts = 2 if quick else 4
    device, dtype, _note = resolve_device(ident.get("device", "auto"), ident.get("precision", "auto"))
    pc = ident["pinn"]
    cfg = PinnConfig(
        seeds=[s + 1000 * j for s in range(seeds) for j in range(restarts)],
        adam_iters=int(pc["adam_iters"]),
        lbfgs_iters=int(pc["lbfgs_iters"]),
        sigma=float(ident["noise_sigma"]),
        eps340=float(next(iter(p.observable_weights().values()))),
        prior_sd=float(p.prior.get("sd_decades", 1.5)),
        device=device,
        dtype=dtype,
        time_scale=str(pc.get("time_scale", "per_experiment")),
        discrepancy="rate_factor",
    )
    t0 = time.time()
    res = train_pinn([PinnExperiment(n, e.initial, e.Et, t, y) for n, e, t, y in train], _prior(p), cfg)
    best: dict = {}
    for r in res:
        s = r.seed % 1000
        tot = r.history[-1][1]
        if s not in best or tot < best[s][0]:
            best[s] = (tot, r)
    chosen = [best[s][1] for s in sorted(best)]
    log(f"PINN(보정 항) {len(res)}회 학습 {time.time() - t0:.1f} s")
    per_seed = [match_seed(r.discrepancy) for r in chosen]
    summ = summarize(per_seed)
    for i, r in enumerate(chosen):
        per_seed[i]["seed"] = r.seed % 1000
    log(f"보정 크기 rms g = {summ['rms_g_median']:.3f} (0이면 다섯 단계 그대로. PINN 자체의 작은 오차도 들어가 참고용)")
    shape = ", ".join(f"{c['name']} {c['r2_median']:.2f}" for c in summ["candidates"])
    log(f"보정 모양과 후보의 일치(R², 참고): {shape}")
    log("후보마다 PINN으로 맞추기(ODE 재해석으로 순위)")
    rank = pinn_candidates(p, train, every, CANDIDATES_LDH, replace(cfg, discrepancy="none"), log)
    sigma = float(ident["noise_sigma"])
    for r in rank:
        ki = f", Ki {r['Ki_mM_median']:.3g} mM" if r["Ki_mM_median"] else ""
        log(f"  {r['name']:18s} 학습 RMSE {r['train_rmse_median']:.4f} ({r['train_rmse_median'] / sigma:.1f}σ){ki}")
    best = [r for r in rank if r["step"]][:verify_top]
    ver = []
    if verify_top > 0:
        bf = ident.get("baseline_fit") or {}
        starts = 1 if quick else int(bf.get("starts", 4))
        nfev = 3 if quick else int(bf.get("max_nfev", 200))
        t0 = time.time()
        ver = verify(p, train, every, best, starts, nfev)
        log(f"확인(고전 피팅, 반응망 {len(ver)}개 × 시작점 {starts}) {time.time() - t0:.1f} s")
        for v in ver:
            log(f"  {v['network']:18s} 학습 RMSE {v.get('train_rmse')}, 홀드아웃 RMSE {v.get('holdout_rmse')}")
    verdict = decide(rank, ver, [n for n, *_ in train], sigma)
    note = verdict["note"]
    suggest = verdict["suggested_step"] is not None
    log(
        f"결론: {note}"
        + (
            f" → {verdict['suggested_step']} (Ki ≈ {verdict['Ki_mM']:.3g} mM, {verdict['Ki_source']})"
            if suggest
            else ""
        )
    )
    report = {
        "project": p.name,
        "quick": quick,
        "verdict": verdict,
        "evidence": summ,
        "per_seed": per_seed,
        "pinn_ranking": rank,
        "verification": ver,
    }
    (out / "suggest.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8")
    plots.write_rows(
        out / "candidates.csv",
        ["network", "step", "train_rmse_median", "holdout_rmse_median", "Ki_mM_median", "shape_r2_median"],
        [
            [
                r["name"],
                r["step"] or "",
                r["train_rmse_median"],
                r["holdout_rmse_median"],
                r["Ki_mM_median"],
                next((c["r2_median"] for c in summ["candidates"] if c["name"] == r["name"]), None),
            ]
            for r in rank
        ],
    )
    plots.suggest_figure(out, chosen, per_seed, summ, rank, ver, sigma, obs)
    return out
