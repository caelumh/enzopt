"""Stage ③ identification: PINN seeds → rate-constant samples (+ classical-fit baseline and non-WR control, F-9/F-10).

Every sample is re-solved with the ODE (scipy) for all experiments; the training-data RMSE decides which seeds are
kept (SC5-3: ODE training RMSE > 3σ → excluded; fewer than 2 kept → the design stage stops). With several signals
(F-13 G6) the RMSE of each signal is divided by its σ and the root mean square over experiments and signals is used;
for one signal this is the old A340 RMSE ÷ σ.
"""

from __future__ import annotations

import time

import numpy as np

from .. import plots
from ..core.massaction import MassAction
from .common import curve, rmse, safe_curve
from .runner import RunContext, StageFailed, write_json
from .stages import load_series, load_signals

MIN_SAMPLES = 2
RESTART_STRIDE = 1000  # internal PINN seed = sample seed + RESTART_STRIDE × restart


def _prior(p) -> np.ndarray:
    if not p.prior:
        raise StageFailed("network.prior(사전분포)가 없음 — PINN·고전 피팅에 필요")
    return p.prior_vector()


def _experiments(ctx: RunContext, roles: tuple[str, ...]):
    out = []
    for name, e in ctx.project.experiments.items():
        if e.role in roles:
            t, y = load_series(ctx, name)
            out.append((name, e, t, y))
    return out


def _inactivation(pc: dict) -> str:
    v = str(pc.get("inactivation", "none"))
    if v not in ("none", "shared"):
        raise StageFailed(f"identification.pinn.inactivation '{v}'는 none 또는 shared")
    return v


def signal_rmse(net, k, e, t, y, coeff, extras, model=None, activity=None) -> dict:
    """ODE re-solve RMSE of every signal of one experiment: {signal name: RMSE or None (solver failed)}."""
    out = {}
    a = safe_curve(net, k, e.initial, e.Et, t, coeff, model=model, activity=activity)
    out["__primary__"] = rmse(a, y) if a is not None else None
    for name, ts, ys, _sg, cf in extras:
        b = safe_curve(net, k, e.initial, e.Et, ts, cf, model=model, activity=activity)
        out[name] = rmse(b, ys) if b is not None else None
    return out


def normalized(vals: list[dict], sigmas: dict) -> float:
    """Root mean square of RMSE ÷ σ over experiments and signals (inf if a re-solve failed)."""
    z = []
    for v in vals:
        for name, r in v.items():
            if r is None:
                return float("inf")
            z.append(r / sigmas[name])
    return float(np.sqrt(np.mean(np.square(z)))) if z else float("inf")


def pinn_config(p, ident: dict, seeds: list[int], device: str, dtype: str, sigma: float):
    from ..pinn.general import PinnConfig

    pc = ident["pinn"]
    return PinnConfig(
        seeds=seeds,
        adam_iters=int(pc["adam_iters"]),
        lbfgs_iters=int(pc["lbfgs_iters"]),
        width=int(pc.get("width", 32)),
        depth=int(pc.get("depth", 3)),
        w_prior=float(pc.get("w_prior", 1e-2)),
        w_phys_final=float(pc.get("w_phys_final", 1e4)),
        representation=str(pc.get("representation", "qss")),
        time_scale=str(pc.get("time_scale", "per_experiment")),
        inactivation=_inactivation(pc),
        inact_k=int(pc.get("inact_k", 12)),
        w_inact=float(pc.get("w_inact", 1e-2)),
        sigma=sigma,
        eps340=float(next(iter(p.observable_weights().values()))),
        prior_sd=float(p.prior.get("sd_decades", 1.5)),
        init_jitter=float(ident.get("init_jitter_decades", 1.0)),
        device=device,
        dtype=dtype,
        network=p.network,
        observable=p.observable_weights(),
        overall=p.overall,
        prefer_dependent=[p.haldane_step] if p.haldane_step else None,
        range_margin=float(pc.get("range_margin", 0.05)),
    )


def _candidate_fit(p, r, train, extras: dict, sigma: float, model=None) -> tuple[float, float]:
    """(ODE re-solve training RMSE, PINN network training RMSE) of one PINN result (signals divided by their σ)."""
    net = p.network
    model = model or MassAction.from_network(net)
    coeff = p.observable_weights()
    sigmas = {"__primary__": sigma, **{sg["name"]: float(sg["sigma"]) for sg in p.signals}}
    kk = net.rate_vector(r.kf, r.kr)
    vals = [signal_rmse(net, kk, e, t, y, coeff, extras[n], model, r.activity) for n, e, t, y in train]
    return sigma * normalized(vals, sigmas), float(sigma * np.sqrt(max(r.loss_parts["data"], 0.0)))


def best_restarts(p, all_results: list, seeds: list[int], train, extras: dict, sigma: float, rule: str = "ode"):
    """Each sample = the best of its restarts by training-data fit (same measure as the SC5-3 exclusion).

    ``all_results`` are the PINN results of internal seeds ``s + RESTART_STRIDE × restart``; ``train`` is
    [(name, experiment, t, y)] and ``extras`` the extra signals per experiment. Returns (one result per seed with
    ``seed`` set to the sample seed, the restart log per seed)."""
    model = MassAction.from_network(p.network)
    results, restart_log = [], {}
    for s in seeds:
        cands = []
        for r in (x for x in all_results if x.seed % RESTART_STRIDE == s):
            ode_tr, pin = _candidate_fit(p, r, train, extras, sigma, model)
            cands.append((pin if rule == "pinn" else ode_tr, r.seed // RESTART_STRIDE, r, ode_tr, pin))
        cands.sort(key=lambda c: (c[0], c[1]))
        best = cands[0]
        restart_log[s] = {
            "chosen_restart": best[1],
            "by": rule,
            "restarts": [
                {"restart": c[1], "ode_train_rmse": c[3], "pinn_rmse": c[4]} for c in sorted(cands, key=lambda c: c[1])
            ],
        }
        best[2].seed = s
        results.append(best[2])
    return results, restart_log


def criteria_samples(
    p, all_results: list, train, extras: dict, sigma: float, rule: str = "ode", sel: dict | None = None
):
    """Samples chosen by criteria instead of seeds (identification.pinn.selection: criteria, core.selection).

    ``all_results`` are the candidates trained from the deterministic starts (``seed`` = candidate index). A
    candidate qualifies when its training RMSE is within the likelihood-ratio threshold of the best one
    (n_eff = number of training experiments unless ``sel['n_eff']`` is given, χ²(1) quantile ``sel['quantile']``).
    Every qualified candidate becomes a sample unless ``sel['max_samples']`` caps the number; the order is sample 0 =
    best fit, then each next one the qualified candidate farthest from those before it (log10 constants in prior
    standard deviations), so a cap keeps the most spread-out ones. Returns (results with ``seed`` = sample index,
    a restart log in the ``best_restarts`` shape, the selection log)."""
    from ..core.selection import CHI2_1_95, lr_threshold, maximin_select

    sel = sel or {}
    model = MassAction.from_network(p.network)
    cands = sorted(all_results, key=lambda r: r.seed)
    fits = [_candidate_fit(p, r, train, extras, sigma, model) for r in cands]
    score = np.array([pin if rule == "pinn" else ode for ode, pin in fits], float)
    finite = np.isfinite(score)
    if not finite.any():
        raise StageFailed("모든 PINN 후보의 학습 데이터 재해석이 실패함")
    n_eff = int(sel.get("n_eff") or len(train))
    q = float(sel.get("quantile", CHI2_1_95))
    best = float(np.min(score[finite]))
    thr = lr_threshold(best, n_eff, q)
    qualified = finite & (score <= thr)
    cap = sel.get("max_samples", "all")
    k = int(qualified.sum()) if cap in (None, "all") else int(cap)
    n_th = len(p.prior_vector())
    theta = np.array([np.asarray(r.theta, float)[:n_th] for r in cands])
    chosen = maximin_select(theta, score, qualified, k, scale=float(p.prior.get("sd_decades", 1.5)))
    order = {ci: j for j, (ci, _) in enumerate(chosen)}
    rows = [
        {
            "candidate": int(r.seed),
            "ode_train_rmse": fits[i][0],
            "pinn_rmse": fits[i][1],
            "qualified": bool(qualified[i]),
            "sample": order.get(i),
        }
        for i, r in enumerate(cands)
    ]
    log = {
        "method": "criteria",
        "starts": "sobol (unscrambled, stratum midpoints) in prior centre ± init_jitter_decades",
        "score": rule,
        "best_rmse": best,
        "n_eff": n_eff,
        "quantile": q,
        "threshold_rmse": thr,
        "n_candidates": len(cands),
        "n_qualified": int(qualified.sum()),
        "max_samples": cap,
        "distance": "Euclidean, log10 constants / prior sd",
        "chosen": [
            {"sample": j, "candidate": int(cands[ci].seed), "min_distance": d} for j, (ci, d) in enumerate(chosen)
        ],
        "candidates": rows,
    }
    results, restart_log = [], {}
    for j, (ci, _d) in enumerate(chosen):
        r = cands[ci]
        restart_log[j] = {
            "chosen_restart": int(r.seed),
            "by": "criteria",
            "restarts": [
                {"restart": x["candidate"], "ode_train_rmse": x["ode_train_rmse"], "pinn_rmse": x["pinn_rmse"]}
                for x in rows
            ],
        }
        r.seed = j
        results.append(r)
    return results, restart_log, log


def run_fit_stage(ctx: RunContext) -> None:
    p = ctx.project
    ident = ctx.settings["identification"]
    out = ctx.stage_dir()
    (out / "samples").mkdir(exist_ok=True)
    method = ident.get("method", "pinn")
    sigma = float(ident["noise_sigma"])
    train = _experiments(ctx, ("train",))
    every = _experiments(ctx, ("train", "holdout", "design-check"))
    extras = {name: load_signals(ctx, name) for name, *_ in every}
    sigmas = {"__primary__": sigma, **{sg["name"]: float(sg["sigma"]) for sg in p.signals}}
    prior = _prior(p)
    seeds = list(range(int(ident["seeds"])))
    t0 = time.time()
    if method != "pinn":
        raise StageFailed(f"identification.method '{method}'는 아직 지원하지 않음(pinn)")
    from ..pinn.general import PinnExperiment, resolve_device, supports, train_pinn

    why = supports(p.network, p.overall)
    if why:
        raise StageFailed(f"이 반응망은 PINN 준정상 표현을 만들 수 없음: {why}")
    dev_req = ctx.options.get("device") or ident.get("device", "auto")
    prec_req = ctx.options.get("precision") or ident.get("precision", "auto")
    device, dtype, note = resolve_device(dev_req, prec_req)
    ctx.manifest["device"] = {"requested": dev_req, "used": device, "precision": dtype, "fallback": note}
    ctx.manifest["seeds"] = seeds
    ctx.save_manifest()
    if note:
        ctx.warn(note)
    n_restarts = max(1, int(ident["pinn"].get("restarts", 1)))
    selection = str(ident["pinn"].get("selection", "per_seed"))
    if selection not in ("per_seed", "criteria"):
        raise StageFailed(f"identification.pinn.selection '{selection}'는 per_seed 또는 criteria")
    if selection == "criteria":
        # seeds × restarts candidates from deterministic starts; the samples are chosen by criteria afterwards
        cfg = pinn_config(p, ident, list(range(len(seeds) * n_restarts)), device, dtype, sigma)
        cfg.start_design = "sobol"
        what = f"PINN 후보 {len(seeds) * n_restarts}개(고정 출발점)"
    else:
        cfg = pinn_config(
            p, ident, [s + RESTART_STRIDE * j for s in seeds for j in range(n_restarts)], device, dtype, sigma
        )
        what = f"PINN {len(seeds)}시드 × 재시작 {n_restarts}"
    pexps = [PinnExperiment(n, e.initial, e.Et, t, y, extra=extras[n]) for n, e, t, y in train]

    def prog(frac, msg):
        ctx.progress(0.7 * frac, f"{what} · {msg}")

    all_results = train_pinn(pexps, prior, cfg, prog)
    train_s = time.time() - t0
    net = p.network
    model = MassAction.from_network(net)
    coeff = p.observable_weights()
    rule = str(ident.get("exclusion_rmse", "ode"))
    k_sig = float(ident.get("exclusion_sigma", 3.0))
    sel_log = None
    if selection == "criteria":
        ctx.progress(0.72, "기준으로 표본 고르기(동등 적합 + 최대 거리)")
        results, restart_log, sel_log = criteria_samples(
            p, all_results, train, extras, sigma, rule, ident["pinn"].get("selection_options")
        )
        seeds = [r.seed for r in results]
    else:
        ctx.progress(0.72, "재시작 중 학습 데이터에 가장 잘 맞는 해 고르기")
        results, restart_log = best_restarts(p, all_results, seeds, train, extras, sigma, rule)
    samples, preds = [], []
    for i, r in enumerate(results):
        ctx.progress(0.75 + 0.2 * i / max(1, len(results)), f"표본 {r.seed} 재해석")
        k = net.rate_vector(r.kf, r.kr)
        ode, ode_sig = {}, {}
        for name, e, t, y in every:
            a = safe_curve(net, k, e.initial, e.Et, t, coeff, model=model, activity=r.activity)
            ode[name] = rmse(a, y) if a is not None else None
            if a is not None:
                preds += [
                    [f"seed{r.seed}", name, e.role, float(tt), float(aa), float(yy)]
                    for tt, aa, yy in zip(t, a, y, strict=True)
                ]
            if extras[name]:
                sr = signal_rmse(net, k, e, t, y, coeff, extras[name], model, r.activity)
                ode_sig[name] = {(p.observable["name"] if kk == "__primary__" else kk): v for kk, v in sr.items()}
        pinn_rmse = float(sigma * np.sqrt(max(r.loss_parts["data"], 0.0)))
        if p.signals:
            vals = [
                {("__primary__" if kk == p.observable["name"] else kk): v for kk, v in ode_sig[n].items()}
                for n, *_ in train
            ]
            z = normalized(vals, sigmas)
            ode_train = None if not np.isfinite(z) else float(sigma * z)
        else:
            tr = [ode[n] for n, *_ in train if ode.get(n) is not None]
            ode_train = float(np.sqrt(np.mean(np.square(tr)))) if tr else None
        judge = pinn_rmse if rule == "pinn" else ode_train
        kept = judge is not None and judge <= k_sig * sigma
        if kept:
            reason = None
        elif judge is None:
            reason = "학습 데이터 ODE 재해석 실패"
        else:
            reason = f"학습 데이터 RMSE {judge:.4f} > {k_sig:g}σ = {k_sig * sigma:.4f} ({rule})"
        if ode_train is not None and ode_train > 2 * max(pinn_rmse, sigma):
            ctx.warn(
                f"시드 {r.seed}: ODE 재해석 RMSE {ode_train:.4f}가 PINN RMSE {pinn_rmse:.4f}의 2배 초과(물리 불일치)"
            )
        s = {
            "seed": r.seed,
            "method": "pinn",
            "kf": [float(x) for x in r.kf],
            "kr": [float(x) for x in r.kr],
            "Kp": r.Kp,
            "Kp_all": r.Kp_all,
            "theta_log10": [float(x) for x in r.theta],
            "loss_parts": r.loss_parts,
            "rmse_pinn_train": pinn_rmse,
            "rmse_ode_train": ode_train,
            "rmse_ode": ode,
            "kept": bool(kept),
            "excluded_reason": reason,
            "activity": r.activity,
            "restart": restart_log[r.seed]["chosen_restart"],
            "n_restarts": n_restarts,
            "history": [[h[0], h[1], h[2]] for h in r.history],
        }
        if ode_sig:
            s["rmse_ode_signals"] = ode_sig
        write_json(out / "samples" / f"sample_seed{r.seed}.json", s)
        samples.append(s)
    kept = [s for s in samples if s["kept"]]
    holdout_names = [n for n, e, *_ in every if e.role == "holdout"]
    per_seed = {s["seed"]: {n: s["rmse_ode"].get(n) for n in holdout_names} for s in kept}
    agg = {}
    for n in holdout_names:
        vals = [v[n] for v in per_seed.values() if v.get(n) is not None]
        agg[n] = {"mean": float(np.mean(vals)) if vals else None, "worst": float(np.max(vals)) if vals else None}
    summary = {
        "method": "pinn",
        "representation": cfg.representation,
        "inactivation": cfg.inactivation,
        "device": ctx.manifest["device"],
        "seeds": seeds,
        "kept": [s["seed"] for s in kept],
        "excluded": {s["seed"]: s["excluded_reason"] for s in samples if not s["kept"]},
        "noise_sigma": sigma,
        "exclusion_rmse": rule,
        "exclusion_sigma": k_sig,
        "train_seconds": train_s,
        "restarts": n_restarts,
        "restart_selection": restart_log,
        "sample_selection": sel_log or {"method": "per_seed"},
        "train_experiments": [n for n, *_ in train],
        "holdout_experiments": holdout_names,
        "holdout_rmse": {"per_seed": per_seed, "summary": agg},
        "design_blocked": len(kept) < MIN_SAMPLES,
        "thermodynamics": {
            "overall": p.overall,
            "dependent_steps": [i + 1 for i in p.layout.dependent],
            "constrained_overall": [p.overall[o] for o in p.layout.overall_used],
        },
    }
    if p.signals:
        summary["signals"] = [sg["name"] for sg in p.all_signals(sigma)]
        summary["holdout_rmse_signals"] = {
            s["seed"]: {n: s.get("rmse_ode_signals", {}).get(n) for n in holdout_names} for s in kept
        }
    if summary["design_blocked"]:
        where = "④ 해석부터" if not kept else "⑤ 설계에서"
        ctx.warn(f"남은 표본 {len(kept)}개 < {MIN_SAMPLES}: {where} 멈춤(SC5-3)")
    ctx.flag("n_samples", len(kept))
    plots.write_rows(
        out / "predictions.csv", ["model", "experiment", "role", "time_s", "A340_model", "A340_obs"], preds
    )
    # F-9 classical-fit baseline and F-10 non-WR control (comparison models)
    from .compare import run_comparisons

    summary["comparisons"] = run_comparisons(ctx, train, every, prior, cfg, prog_base=0.95)
    write_json(out / "fit_summary.json", summary)
    ctx.progress(0.97, "그림")
    figs = plots.fit_figures(out, p, every, samples, summary)
    write_json(out / "figures.json", figs)


def predict_all(ctx: RunContext, k, network=None, activity: dict | None = None) -> dict:
    """Observable predictions of one rate vector (and enzyme activity a(t)) for every experiment."""
    p = ctx.project
    net = network or p.network
    res = {}
    for name, e in p.experiments.items():
        t, y = load_series(ctx, name)
        res[name] = (t, curve(net, k, e.initial, e.Et, t, p.observable_weights(), activity=activity), y)
    return res
