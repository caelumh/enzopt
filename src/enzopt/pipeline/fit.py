"""Stage ③ identification: PINN seeds → rate-constant samples (+ classical-fit baseline and non-WR control, F-9/F-10).

Every sample is re-solved with the ODE (scipy) for all experiments; the training-data RMSE decides which seeds are
kept (SC5-3: ODE training RMSE > 3σ → excluded; fewer than 2 kept → the design stage stops).
"""

from __future__ import annotations

import time

import numpy as np

from .. import plots
from ..core.massaction import MassAction
from .common import curve, rmse, safe_curve
from .runner import RunContext, StageFailed, write_json
from .stages import load_series

MIN_SAMPLES = 2
RESTART_STRIDE = 1000  # internal PINN seed = sample seed + RESTART_STRIDE × restart


def _prior(p) -> np.ndarray:
    pr = p.prior
    if not pr:
        raise StageFailed("network.prior(사전분포)가 없음 — PINN·고전 피팅에 필요")
    vals = list(pr["kf_log10"]) + list(pr["kr_log10"])
    if p.haldane_step:  # K' is a free parameter only when a Haldane step ties it to the rate constants
        vals.append(pr.get("Kp_log10", np.log10(p.Kp)))
    return np.array(vals, float)


def _experiments(ctx: RunContext, roles: tuple[str, ...]):
    out = []
    for name, e in ctx.project.experiments.items():
        if e.role in roles:
            t, y = load_series(ctx, name)
            out.append((name, e, t, y))
    return out


def run_fit_stage(ctx: RunContext) -> None:
    p = ctx.project
    ident = ctx.settings["identification"]
    out = ctx.stage_dir()
    (out / "samples").mkdir(exist_ok=True)
    method = ident.get("method", "pinn")
    sigma = float(ident["noise_sigma"])
    train = _experiments(ctx, ("train",))
    every = _experiments(ctx, ("train", "holdout", "design-check"))
    prior = _prior(p)
    seeds = list(range(int(ident["seeds"])))
    t0 = time.time()
    if method != "pinn":
        raise StageFailed(f"identification.method '{method}'는 아직 지원하지 않음(pinn)")
    from ..pinn.ldh import PinnConfig, PinnExperiment, matches_network, resolve_device, train_pinn

    if not matches_network(p.network):
        raise StageFailed(
            "이 반응망의 PINN 표현이 아직 없음(현재 LDH 순차 bi-bi만). 일반화(F-13) 전에는 표현을 코드로 추가해야 함"
        )
    dev_req = ctx.options.get("device") or ident.get("device", "auto")
    prec_req = ctx.options.get("precision") or ident.get("precision", "auto")
    device, dtype, note = resolve_device(dev_req, prec_req)
    ctx.manifest["device"] = {"requested": dev_req, "used": device, "precision": dtype, "fallback": note}
    ctx.manifest["seeds"] = seeds
    ctx.save_manifest()
    if note:
        ctx.warn(note)
    pc = ident["pinn"]
    n_restarts = max(1, int(pc.get("restarts", 1)))
    cfg = PinnConfig(
        seeds=[s + RESTART_STRIDE * j for s in seeds for j in range(n_restarts)],
        adam_iters=int(pc["adam_iters"]),
        lbfgs_iters=int(pc["lbfgs_iters"]),
        width=int(pc.get("width", 32)),
        depth=int(pc.get("depth", 3)),
        w_prior=float(pc.get("w_prior", 1e-2)),
        w_phys_final=float(pc.get("w_phys_final", 1e4)),
        representation=str(pc.get("representation", "qss")),
        time_scale=str(pc.get("time_scale", "per_experiment")),
        sigma=sigma,
        eps340=float(next(iter(p.observable_weights().values()))),
        prior_sd=float(p.prior.get("sd_decades", 1.5)),
        init_jitter=float(ident.get("init_jitter_decades", 1.0)),
        device=device,
        dtype=dtype,
        irreversible_last=not p.network.steps[-1].reversible,
    )
    pexps = [PinnExperiment(n, e.initial, e.Et, t, y) for n, e, t, y in train]

    def prog(frac, msg):
        ctx.progress(0.7 * frac, f"PINN {len(seeds)}시드 × 재시작 {n_restarts} · {msg}")

    all_results = train_pinn(pexps, prior, cfg, prog)
    train_s = time.time() - t0
    net = p.network
    model = MassAction.from_network(net)
    coeff = p.observable_weights()
    rule = str(ident.get("exclusion_rmse", "ode"))
    k_sig = float(ident.get("exclusion_sigma", 3.0))
    # each sample = the best of its restarts by training-data fit (same measure as the SC5-3 exclusion)
    ctx.progress(0.72, "재시작 중 학습 데이터에 가장 잘 맞는 해 고르기")
    results, restart_log = [], {}
    for s in seeds:
        cands = []
        for r in (x for x in all_results if x.seed % RESTART_STRIDE == s):
            kk = net.rate_vector(r.kf, r.kr)
            vals = []
            for _name, e, t, y in train:
                a = safe_curve(net, kk, e.initial, e.Et, t, coeff, model=model)
                vals.append(rmse(a, y) if a is not None else np.inf)
            ode_tr = float(np.sqrt(np.mean(np.square(vals))))
            pin = float(sigma * np.sqrt(max(r.loss_parts["data"], 0.0)))
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
    samples, preds = [], []
    for i, r in enumerate(results):
        ctx.progress(0.75 + 0.2 * i / max(1, len(results)), f"표본 {r.seed} 재해석")
        k = net.rate_vector(r.kf, r.kr)
        ode = {}
        for name, e, t, y in every:
            a = safe_curve(net, k, e.initial, e.Et, t, coeff, model=model)
            ode[name] = rmse(a, y) if a is not None else None
            if a is not None:
                preds += [
                    [f"seed{r.seed}", name, e.role, float(tt), float(aa), float(yy)]
                    for tt, aa, yy in zip(t, a, y, strict=True)
                ]
        pinn_rmse = float(sigma * np.sqrt(max(r.loss_parts["data"], 0.0)))
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
            "theta_log10": [float(x) for x in r.theta],
            "loss_parts": r.loss_parts,
            "rmse_pinn_train": pinn_rmse,
            "rmse_ode_train": ode_train,
            "rmse_ode": ode,
            "kept": bool(kept),
            "excluded_reason": reason,
            "restart": restart_log[r.seed]["chosen_restart"],
            "n_restarts": n_restarts,
            "history": [[h[0], h[1], h[2]] for h in r.history],
        }
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
        "train_experiments": [n for n, *_ in train],
        "holdout_experiments": holdout_names,
        "holdout_rmse": {"per_seed": per_seed, "summary": agg},
        "design_blocked": len(kept) < MIN_SAMPLES,
    }
    if summary["design_blocked"]:
        ctx.warn(f"남은 표본 {len(kept)}개 < {MIN_SAMPLES}: 설계 단계에서 멈춤(SC5-3)")
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


def predict_all(ctx: RunContext, k, network=None) -> dict:
    """Observable predictions of one rate vector for every experiment (used by later stages)."""
    p = ctx.project
    net = network or p.network
    res = {}
    for name, e in p.experiments.items():
        t, y = load_series(ctx, name)
        res[name] = (t, curve(net, k, e.initial, e.Et, t, p.observable_weights()), y)
    return res
