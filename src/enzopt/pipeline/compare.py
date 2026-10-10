"""Comparison models for stage ③: classical fitting baseline (F-9) and the non-weakly-reversible control (F-10).

Classical fit: the same ODE (full mechanism, Haldane), the same data, prior and starting points as the PINN seeds,
solved by multi-start least squares (scipy ``least_squares``). Non-WR control: the PINN trained on the control
network (listed steps made one-way, no K').
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares

from ..core.design import n_workers
from ..core.massaction import MassAction, SimulationError, SolverSettings, atol_vector, haldane_kr, simulate
from ..core.network import x0_vector
from .common import rmse, safe_curve
from .runner import RunContext, write_json

FIT_SETTINGS = SolverSettings(method="Radau", rtol=1e-6, atol_enzyme_rel=1e-7, atol_other=1e-10)


def theta_to_rates(theta: np.ndarray, network, haldane_step: int | None):
    """θ = log10 [kf (all steps), kr (reversible steps except the Haldane step), K' (if Haldane)] → (kf, kr)."""
    n = len(network.steps)
    kf = 10.0 ** theta[:n]
    kr = np.zeros(n)
    j = n
    for i, st in enumerate(network.steps):
        if st.reversible and (i + 1) != haldane_step:
            kr[i] = 10.0 ** theta[j]
            j += 1
    if haldane_step:  # same cycle convention as core.massaction.haldane_kr (steps 1..haldane_step)
        kr = haldane_kr(kf, kr, 10.0 ** theta[j], haldane_step)
    return kf, kr


def _residuals(theta, args):
    net, exps, coeff, sigma, prior, w_prior, sd, hs = args
    kf, kr = theta_to_rates(theta, net, hs)
    k = net.rate_vector(kf, kr)
    model = MassAction.from_network(net)
    res = []
    n_exp = len(exps)
    for initial, Et, t, y in exps:
        try:
            X = simulate(
                model,
                k,
                x0_vector(net, initial, Et),
                t,
                atol=atol_vector(net.species, net.enzyme_species, Et, FIT_SETTINGS),
                settings=FIT_SETTINGS,
            )
            idx = net.index()
            a = sum(c * X[:, idx[s]] for s, c in coeff.items())
            r = (a - y) / sigma / np.sqrt(len(t) * n_exp)
        except (SimulationError, ValueError, FloatingPointError):
            r = np.full(len(t), 1e3 / np.sqrt(len(t) * n_exp))
        res.append(np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3))
    res.append(np.sqrt(w_prior / (2 * sd**2)) * (theta - prior))
    return np.concatenate(res)


def _fit_one(job):
    start, theta0, args, max_nfev = job
    t0 = time.time()
    try:
        sol = least_squares(
            _residuals, theta0, args=(args,), method="trf", x_scale=1.0, max_nfev=max_nfev, diff_step=1e-4
        )
        theta, cost, nfev, status = sol.x, float(sol.cost), int(sol.nfev), int(sol.status)
    except Exception as e:  # noqa: BLE001 — a crashed start is reported, not fatal
        return {"start": start, "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}
    return {
        "start": start,
        "theta": theta.tolist(),
        "cost": cost,
        "nfev": nfev,
        "status": status,
        "seconds": time.time() - t0,
    }


def classical_fit(ctx: RunContext, train, prior: np.ndarray, seeds: list[int], max_nfev: int, jitter: float) -> list:
    p = ctx.project
    net = p.network
    hs = p.haldane_step
    sd = float(p.prior.get("sd_decades", 1.5))
    w_prior = float(ctx.settings["identification"]["pinn"].get("w_prior", 1e-2))
    exps = [(e.initial, e.Et, t, y) for _, e, t, y in train]
    args = (
        net,
        exps,
        p.observable_weights(),
        float(ctx.settings["identification"]["noise_sigma"]),
        prior,
        w_prior,
        sd,
        hs,
    )
    jobs = []
    for s in seeds:
        theta0 = prior + np.random.default_rng(100 + int(s)).uniform(-jitter, jitter, len(prior))
        jobs.append((int(s), theta0, args, max_nfev))
    w = n_workers(len(jobs))
    if w > 1 and len(jobs) > 1:
        with ProcessPoolExecutor(max_workers=w) as ex:
            out = list(ex.map(_fit_one, jobs))
    else:
        out = [_fit_one(j) for j in jobs]
    for r in out:
        if "theta" in r:
            kf, kr = theta_to_rates(np.array(r["theta"]), net, hs)
            r["kf"], r["kr"] = kf.tolist(), kr.tolist()
            r["Kp"] = float(10 ** r["theta"][-1]) if hs else None
    return out


def run_comparisons(ctx: RunContext, train, every, prior, pinn_cfg, prog_base: float = 0.95) -> dict:
    """Run the enabled comparison models; write ``3_fit/baseline/*`` and ``3_fit/control/*``; return a summary."""
    p = ctx.project
    ident = ctx.settings["identification"]
    out = ctx.stage_dir()
    coeff = p.observable_weights()
    summary = {}
    rows = []  # model, id, experiment, role, rmse
    # ---------------------------------------------------------------- F-9 classical fit
    bf = ident.get("baseline_fit") or {}
    if bf.get("enabled", True) and int(bf.get("starts", 0)) > 0:
        ctx.progress(prog_base, "고전 피팅(다중 시작점 최소제곱)")
        t0 = time.time()
        starts = list(range(int(bf["starts"])))
        fits = classical_fit(
            ctx, train, prior, starts, int(bf.get("max_nfev", 200)), float(ident.get("init_jitter_decades", 1.0))
        )
        d = out / "baseline"
        d.mkdir(exist_ok=True)
        model = MassAction.from_network(p.network)
        for r in fits:
            if "kf" not in r:
                continue
            k = p.network.rate_vector(r["kf"], r["kr"])
            r["rmse_ode"] = {}
            for name, e, t, y in every:
                a = safe_curve(p.network, k, e.initial, e.Et, t, coeff, model=model)
                r["rmse_ode"][name] = rmse(a, y) if a is not None else None
                rows.append(["classical", f"start{r['start']}", name, e.role, r["rmse_ode"][name]])
            write_json(d / f"fit_start{r['start']}.json", r)
        summary["classical"] = {
            "inactivation": "none",  # the least-squares baseline never models enzyme inactivation
            "starts": starts,
            "seconds": time.time() - t0,
            "fits": [{k: v for k, v in r.items() if k != "theta"} for r in fits],
        }
    # ---------------------------------------------------------------- F-10 non-WR control
    cc = ident.get("control") or {}
    ctrl_net = p.control_network()
    if ctrl_net is not None and cc.get("enabled", True) and int(cc.get("seeds", 0)) > 0:
        from ..pinn.ldh import train_pinn

        one_way = [i + 1 for i, st in enumerate(ctrl_net.steps) if not st.reversible]
        if one_way == [len(ctrl_net.steps)]:
            ctx.progress(prog_base + 0.01, "비약가역 대조 PINN")
            t0 = time.time()
            from dataclasses import replace

            from ..pinn.ldh import PinnExperiment

            cfg = replace(pinn_cfg, seeds=list(range(int(cc["seeds"]))), irreversible_last=True)
            pexps = [PinnExperiment(n, e.initial, e.Et, t, y) for n, e, t, y in train]
            res = train_pinn(pexps, prior[:9], cfg)
            d = out / "control"
            d.mkdir(exist_ok=True)
            model = MassAction.from_network(ctrl_net)
            fits = []
            for r in res:
                k = ctrl_net.rate_vector(r.kf, r.kr)
                rec = {
                    "seed": r.seed,
                    "kf": r.kf.tolist(),
                    "kr": r.kr.tolist(),
                    "activity": r.activity,
                    "loss_parts": r.loss_parts,
                    "rmse_ode": {},
                }
                for name, e, t, y in every:
                    a = safe_curve(ctrl_net, k, e.initial, e.Et, t, coeff, model=model, activity=r.activity)
                    rec["rmse_ode"][name] = rmse(a, y) if a is not None else None
                    rows.append(["nonWR", f"seed{r.seed}", name, e.role, rec["rmse_ode"][name]])
                write_json(d / f"control_seed{r.seed}.json", rec)
                fits.append(rec)
            summary["control"] = {
                "network": ctrl_net.name,
                "one_way_steps": one_way,
                "seconds": time.time() - t0,
                "fits": fits,
            }
        else:
            summary["control"] = {"skipped": f"PINN 대조는 마지막 단계 한 방향만 지원(지정: {one_way})"}
    # WR PINN rows for the comparison table
    from .common import load_samples

    for s in load_samples(ctx.run_dir, kept_only=True):
        for name, e, *_ in every:
            rows.append(["PINN (WR)", f"seed{s['seed']}", name, e.role, s["rmse_ode"].get(name)])
    from .. import plots

    plots.write_rows(out / "holdout_comparison.csv", ["model", "id", "experiment", "role", "rmse_A340"], rows)
    summary["table"] = "holdout_comparison.csv"
    return summary
