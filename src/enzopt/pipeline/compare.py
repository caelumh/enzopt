"""Comparison models for stage ③: classical fitting baseline (F-9) and the non-weakly-reversible control (F-10).

Classical fit: the same ODE (full mechanism, thermodynamic constraints from ``core.thermo``), the same data (every
signal, each in units of its σ), prior and starting points as the PINN seeds, solved by multi-start least squares
(scipy ``least_squares``). Non-WR control: the PINN trained on the control network (``control.one_way_steps`` made
one-way; the constraints of the cycles they break, and their K′, drop out).
"""

from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor

import numpy as np
from scipy.optimize import least_squares

from ..core.design import n_workers
from ..core.massaction import MassAction, SimulationError, SolverSettings, atol_vector, haldane_kr, simulate
from ..core.network import x0_vector
from ..core.thermo import ThermoLayout, layout
from .common import rmse, safe_curve
from .runner import RunContext, write_json

FIT_SETTINGS = SolverSettings(method="Radau", rtol=1e-6, atol_enzyme_rel=1e-7, atol_other=1e-10)


def theta_to_rates(theta: np.ndarray, network_or_layout, haldane_step: int | None = None):
    """θ → (kf, kr). With a ``ThermoLayout`` the general layout (``core.thermo``); with a network the old LDH form
    θ = log10 [kf (all steps), kr (reversible steps except the Haldane step), K' (if Haldane)]."""
    if isinstance(network_or_layout, ThermoLayout):
        return network_or_layout.rates(theta)
    network = network_or_layout
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
    net, lay, exps, coeff, sigma, prior, w_prior, sd = args
    kf, kr = lay.rates(theta)
    k = net.rate_vector(kf, kr)
    model = MassAction.from_network(net)
    res = []
    n_exp = len(exps)
    idx = net.index()
    for initial, Et, t, y, extras in exps:
        series = [(t, y, coeff, sigma)] + [(ts, ys, cf, sg) for _n, ts, ys, sg, cf in extras]
        n_sig = len(series)
        for ts, ys, cf, sg in series:
            w = 1.0 / np.sqrt(len(ts) * n_exp * n_sig)
            try:
                X = simulate(
                    model,
                    k,
                    x0_vector(net, initial, Et),
                    ts,
                    atol=atol_vector(net.species, net.enzyme_species, Et, FIT_SETTINGS),
                    settings=FIT_SETTINGS,
                )
                a = sum(c * X[:, idx[s]] for s, c in cf.items())
                r = (a - ys) / sg * w
            except (SimulationError, ValueError, FloatingPointError):
                r = np.full(len(ts), 1e3 * w)
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


def classical_fit(
    ctx: RunContext, train, prior: np.ndarray, seeds: list[int], max_nfev: int, jitter: float, extras=None
) -> list:
    p = ctx.project
    net = p.network
    lay = p.layout
    sd = float(p.prior.get("sd_decades", 1.5))
    w_prior = float(ctx.settings["identification"]["pinn"].get("w_prior", 1e-2))
    extras = extras or {}
    exps = [(e.initial, e.Et, t, y, extras.get(n, [])) for n, e, t, y in train]
    args = (
        net,
        lay,
        exps,
        p.observable_weights(),
        float(ctx.settings["identification"]["noise_sigma"]),
        prior,
        w_prior,
        sd,
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
    n, nf = len(net.steps), lay.n_free_kr
    for r in out:
        if "theta" in r:
            kf, kr = lay.rates(np.array(r["theta"]))
            r["kf"], r["kr"] = kf.tolist(), kr.tolist()
            kp = [float(10 ** r["theta"][n + nf + c]) for c in range(len(lay.overall_used))]
            r["Kp"] = kp[0] if kp else None
            r["Kp_all"] = kp
    return out


def control_prior(p, ctrl_net, ctrl_lay) -> np.ndarray:
    """Prior centre of the control network: the same kf and k₋ (dependent ones as implied at the prior centre)."""
    n = len(ctrl_net.steps)
    th = p.prior_vector()
    kr_all = p.kr_prior_all()
    kr = [kr_all[i] for i in ctrl_lay.free_kr]
    full_kp = {o: th[n + p.layout.n_free_kr + c] for c, o in enumerate(p.layout.overall_used)}
    kp = [full_kp.get(o, np.log10(p.reactions[o]["Kp"] or 1.0)) for o in ctrl_lay.overall_used]
    return np.array(list(th[:n]) + kr + kp, float)


def run_comparisons(ctx: RunContext, train, every, prior, pinn_cfg, prog_base: float = 0.95) -> dict:
    """Run the enabled comparison models; write ``3_fit/baseline/*`` and ``3_fit/control/*``; return a summary."""
    from .fit import signal_rmse
    from .stages import load_signals

    p = ctx.project
    ident = ctx.settings["identification"]
    out = ctx.stage_dir()
    coeff = p.observable_weights()
    extras = {name: load_signals(ctx, name) for name, *_ in every}
    summary = {}
    rows = []  # model, id, experiment, role, rmse of the primary signal, RMSE ÷ σ over all signals
    sig_of = {p.observable["name"]: float(ident["noise_sigma"]), **{sg["name"]: float(sg["sigma"]) for sg in p.signals}}

    def all_sigma(name: str, prim, by_signal: dict | None):
        """Root mean square of RMSE ÷ σ over every signal of one experiment (the primary alone without signals)."""
        vals = (by_signal or {}).get(name) or {p.observable["name"]: prim}
        if any(v is None for v in vals.values()):
            return None
        return float(np.sqrt(np.mean([(v / sig_of[k]) ** 2 for k, v in vals.items()])))

    def per_signal(net, k, model, activity=None):
        res = {}
        for name, e, t, y in every:
            if not extras[name]:
                continue
            sr = signal_rmse(net, k, e, t, y, coeff, extras[name], model, activity)
            res[name] = {(p.observable["name"] if kk == "__primary__" else kk): v for kk, v in sr.items()}
        return res

    # ---------------------------------------------------------------- F-9 classical fit
    bf = ident.get("baseline_fit") or {}
    if bf.get("enabled", True) and int(bf.get("starts", 0)) > 0:
        ctx.progress(prog_base, "고전 피팅(다중 시작점 최소제곱)")
        t0 = time.time()
        starts = list(range(int(bf["starts"])))
        fits = classical_fit(
            ctx,
            train,
            prior,
            starts,
            int(bf.get("max_nfev", 200)),
            float(ident.get("init_jitter_decades", 1.0)),
            extras,
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
            if p.signals:
                r["rmse_ode_signals"] = per_signal(p.network, k, model)
            for name, e, *_ in every:
                prim = r["rmse_ode"][name]
                aa = all_sigma(name, prim, r.get("rmse_ode_signals"))
                rows.append(["classical", f"start{r['start']}", name, e.role, prim, aa])
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
        from dataclasses import replace

        from ..pinn.general import PinnExperiment, supports, train_pinn

        # steps the control makes one-way (steps already one-way in the network, e.g. a background step, are not listed)
        one_way = [
            i + 1
            for i, (a, b) in enumerate(zip(p.network.steps, ctrl_net.steps, strict=True))
            if a.reversible and not b.reversible
        ]
        why = supports(ctrl_net, p.overall)
        if why:
            summary["control"] = {"skipped": f"대조 반응망의 PINN 표현을 만들 수 없음: {why}"}
        else:
            ctx.progress(prog_base + 0.01, "대조 PINN")
            t0 = time.time()
            ctrl_lay = layout(ctrl_net, p.overall, [p.haldane_step] if p.haldane_step else None)
            cfg = replace(pinn_cfg, seeds=list(range(int(cc["seeds"]))), network=ctrl_net)
            pexps = [PinnExperiment(n, e.initial, e.Et, t, y, extra=extras[n]) for n, e, t, y in train]
            res = train_pinn(pexps, control_prior(p, ctrl_net, ctrl_lay), cfg)
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
                if p.signals:
                    rec["rmse_ode_signals"] = per_signal(ctrl_net, k, model, r.activity)
                for name, e, *_ in every:
                    prim = rec["rmse_ode"][name]
                    aa = all_sigma(name, prim, rec.get("rmse_ode_signals"))
                    rows.append(["nonWR", f"seed{r.seed}", name, e.role, prim, aa])
                write_json(d / f"control_seed{r.seed}.json", rec)
                fits.append(rec)
            summary["control"] = {
                "network": ctrl_net.name,
                "one_way_steps": one_way,
                "seconds": time.time() - t0,
                "fits": fits,
            }
    # WR PINN rows for the comparison table
    from .common import load_samples

    for s in load_samples(ctx.run_dir, kept_only=True):
        for name, e, *_ in every:
            prim = s["rmse_ode"].get(name)
            rows.append(
                ["PINN (WR)", f"seed{s['seed']}", name, e.role, prim, all_sigma(name, prim, s.get("rmse_ode_signals"))]
            )
    from .. import plots

    # model ids stay "PINN (WR)" / "nonWR" (read by tests and the dashboard); figures name them by the network (plots)
    header = ["model", "id", "experiment", "role", "rmse_A340", "rmse_all_sigma"]
    plots.write_rows(out / "holdout_comparison.csv", header, rows)
    summary["table"] = "holdout_comparison.csv"
    return summary
