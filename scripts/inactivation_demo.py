"""What only the PINN does cheaply: learn an unknown enzyme-activity curve a(t) together with the rate constants.

    uv run python scripts/inactivation_demo.py            # → docs/inactivation/ (results.json, fig_inactivation.png)

Synthetic LDH progress curves (truth = the constants of evals/synthetic.yaml, K' = 23, σ = 0.003 A340) in which the
enzyme loses activity with a lag-then-drop shape a(t) = 1/(1 + (t/t50)^n), t = time since the enzyme was added.
Training sets A, B, C; held-out set D'. Five identifications of the same data:

    pinn_none     PINN, a ≡ 1 (enzopt today)
    pinn_shared   PINN, a(t) learned without choosing its shape (12 monotone sigmoid steps, no ODE solve in training)
    cls_none      least squares on the ODE, a ≡ 1
    cls_exp       least squares, first-order loss a = exp(−k_i t) (the usual assumption)
    cls_flex      least squares with the same 12-step a(t) as the PINN (the classical way to learn the same function)

Every result is checked the same way: the ODE is solved again with the identified constants and a(t). The design
question at the end is the one enzopt answers: the enzyme dose for 99 % NADH conversion in 300 s at ρ = 16, and what
the true enzyme then does at that dose.
"""

from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.integrate import solve_ivp
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "inactivation"
PROJECT = ROOT / "projects" / "example_ldh_synth"
SIGMA = 0.003
T_OBS = np.arange(5.0, 900.0, 10.0)
TRAIN, HOLDOUT = ["A", "B", "C"], ["Dp"]
T50, HILL = 300.0, 3.0  # truth: half the activity left at 300 s, little loss before ~150 s
K_STEPS = 12
RHO, T_DESIGN, CONV = 16.0, 300.0, 0.99
W_INACT, W_PRIOR = 1e-2, 1e-2
N_EXTRA = {"none": 0, "exp": 1, "flex": K_STEPS}  # activity parameters per classical variant
LABELS = {
    "pinn_none": "PINN, no inactivation",
    "pinn_shared": "PINN, a(t) learned",
    "cls_none": "classical, no inactivation",
    "cls_exp": "classical, a = exp(−k t)",
    "cls_flex": "classical, same a(t) basis",
}
G: dict = {}


# ---------------------------------------------------------------- shared set-up (also in worker processes)
def init():
    from enzopt.core.massaction import MassAction
    from enzopt.core.project import load_project
    from enzopt.core.yamlio import load_yaml

    p = load_project(PROJECT)
    syn = load_yaml(ROOT / "evals" / "synthetic.yaml")
    tr = syn["truth"]
    net, dnet = p.network, p.design_network()
    G.update(
        p=p,
        net=net,
        dnet=dnet,
        model=MassAction.from_network(net),
        dmodel=MassAction.from_network(dnet),
        mask=np.array([1.0 if s in net.enzyme_species else 0.0 for s in net.species]),
        dmask=np.array([1.0 if s in dnet.enzyme_species else 0.0 for s in dnet.species]),
        truth_kf=np.array(tr["kf"], float),
        truth_kr=np.array(list(tr["kr"]) + [np.nan], float),
        Kp=float(tr["Kp"]),
        Et=24.23e-6,
        coeff=p.observable_weights(),
    )


def truth_act():
    return {"kind": "hill", "t50": T50, "n": HILL}


def lam_fn(act):
    """λ(t) = −d ln a/dt for an activity description."""
    from enzopt.pinn.ldh import activity_rate

    if act is None:
        return lambda t: 0.0
    if act.get("kind") == "hill":
        t50, n = act["t50"], act["n"]
        return lambda t: 0.0 if t <= 0 else n / t * (t / t50) ** n / (1.0 + (t / t50) ** n)
    if act.get("kind") == "exp":
        return lambda t: act["ki"]
    return lambda t: float(activity_rate(act, t))


def act_curve(act, t):
    from enzopt.pinn.ldh import activity_curve

    t = np.asarray(t, float)
    if act is None:
        return np.ones_like(t)
    if act.get("kind") == "hill":
        return 1.0 / (1.0 + (t / act["t50"]) ** act["n"])
    if act.get("kind") == "exp":
        return np.exp(-act["ki"] * t)
    return activity_curve(act, t)


def simulate(model, k, x0, t_eval, mask, act, Et, rtol=1e-7):
    lam = lam_fn(act)

    def f(t, x):
        return model.rhs(t, x, k) - lam(t) * mask * x

    def jac(t, x):
        return model.jac(t, x, k) - np.diag(lam(t) * mask)

    atol = np.where(mask > 0, Et * 1e-8, 1e-11)
    sol = solve_ivp(f, (0.0, float(t_eval[-1])), x0, t_eval=t_eval, method="Radau", jac=jac, rtol=rtol, atol=atol)
    if not sol.success:
        raise RuntimeError(sol.message)
    return sol.y.T


def a340(name, kf, kr, act, t=T_OBS):
    from enzopt.core.network import x0_vector

    net, e = G["net"], G["p"].experiments[name]
    X = simulate(G["model"], net.rate_vector(kf, kr), x0_vector(net, e.initial, G["Et"]), t, G["mask"], act, G["Et"])
    idx = net.index()
    return sum(c * X[:, idx[s]] for s, c in G["coeff"].items())


def make_data():
    from enzopt.core.massaction import haldane_kr

    kr = haldane_kr(G["truth_kf"], G["truth_kr"], G["Kp"], 5)
    rng = np.random.default_rng(20261010)
    data = {}
    for name in TRAIN + HOLDOUT:
        clean = a340(name, G["truth_kf"], kr, truth_act())
        data[name] = clean + rng.normal(0.0, SIGMA, len(T_OBS))
    return data


# ---------------------------------------------------------------- re-check and design (worker functions)
def recheck(job):
    kf, kr, act, data = job
    out = {}
    for name in TRAIN + HOLDOUT:
        try:
            out[name] = float(np.sqrt(np.mean((a340(name, kf, kr, act) - np.asarray(data[name])) ** 2)))
        except Exception:  # noqa: BLE001 — a failed solve counts as a miss
            out[name] = float("inf")
    out["train"] = float(np.sqrt(np.mean([out[n] ** 2 for n in TRAIN])))
    return out


def nadh_left(kf, kr, act, Et):
    from enzopt.core.network import x0_vector
    from enzopt.pipeline.common import design_rates

    dnet = G["dnet"]
    des = G["p"].design
    k = design_rates(dnet, {"kf": list(kf), "kr": list(kr)}, des, G["Kp"], 5)
    x0 = x0_vector(dnet, {"NADH": 0.1, "Pyr": RHO * 0.1}, Et)
    try:
        X = simulate(G["dmodel"], k, x0, np.array([T_DESIGN]), G["dmask"], act, Et)
    except Exception:  # noqa: BLE001
        return float("nan")
    return float(X[-1, dnet.index()["NADH"]])


def dose(job):
    """Smallest Et (mM) with NADH(300 s) ≤ 1 % of 0.1 mM for one set of constants and a(t) (log bisection)."""
    kf, kr, act = job
    lo, hi = 1e-6, 1e-2
    target = (1.0 - CONV) * 0.1
    if not nadh_left(kf, kr, act, hi) <= target:
        return float("nan")
    for _ in range(22):
        mid = np.sqrt(lo * hi)
        if nadh_left(kf, kr, act, mid) <= target:
            hi = mid
        else:
            lo = mid
    return hi


def truth_conversion(Et):
    from enzopt.core.massaction import haldane_kr

    kr = haldane_kr(G["truth_kf"], G["truth_kr"], G["Kp"], 5)
    return 1.0 - nadh_left(G["truth_kf"], kr, truth_act(), Et) / 0.1


# ---------------------------------------------------------------- classical fits
def basis_act(z):
    w = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0.0)  # softplus
    return {"T": float(T_OBS[-1]), "c": np.linspace(0.0, 1.0, K_STEPS).tolist(), "s": 0.6 / (K_STEPS - 1), "w": w}


def h_end(act):
    return -np.log(act_curve(act, T_OBS[-1]))


def cls_unpack(theta, kind):
    from enzopt.pipeline.compare import theta_to_rates

    kf, kr = theta_to_rates(theta[:10], G["net"], 5)
    if kind == "none":
        return kf, kr, None
    if kind == "exp":
        return kf, kr, {"kind": "exp", "ki": float(10.0 ** theta[10])}
    return kf, kr, basis_act(theta[10:])


def cls_residuals(theta, kind, data, prior):
    kf, kr, act = cls_unpack(theta, kind)
    res = []
    for name in TRAIN:
        try:
            r = (a340(name, kf, kr, act) - data[name]) / SIGMA / np.sqrt(len(T_OBS) * len(TRAIN))
        except Exception:  # noqa: BLE001
            r = np.full(len(T_OBS), 1e3 / np.sqrt(len(T_OBS) * len(TRAIN)))
        res.append(np.nan_to_num(r, nan=1e3, posinf=1e3, neginf=-1e3))
    res.append(np.sqrt(W_PRIOR / (2 * 1.5**2)) * (theta[:10] - prior))
    if kind == "exp":
        res.append(np.sqrt(W_PRIOR / (2 * 1.5**2)) * (theta[10:] - (-3.0)))
    if kind == "flex":
        res.append(np.array([np.sqrt(W_INACT) * h_end(act)]))
    return np.concatenate(res)


def cls_fit(job):
    kind, start, data, prior = job

    rng = np.random.default_rng(100 + start)
    theta0 = list(prior + rng.uniform(-1.0, 1.0, len(prior)))
    if kind == "exp":
        theta0.append(-3.0 + rng.uniform(-1.0, 1.0))
    if kind == "flex":
        theta0 += [-5.0] * K_STEPS
    t0 = time.time()
    sol = least_squares(
        cls_residuals, np.array(theta0), args=(kind, data, prior), method="trf", max_nfev=200, diff_step=1e-4
    )
    kf, kr, act = cls_unpack(sol.x, kind)
    return {
        "method": f"cls_{kind}",
        "start": start,
        "kf": kf.tolist(),
        "kr": kr.tolist(),
        "act": _jsonable(act),
        "seconds": time.time() - t0,
        "nfev": int(sol.nfev),
        "njev": int(sol.njev or 0),
    }


def _jsonable(act):
    if act is None:
        return None
    return {k: (np.asarray(v).tolist() if isinstance(v, np.ndarray | list) else v) for k, v in act.items()}


# ---------------------------------------------------------------- main
def main():
    from enzopt.pinn.ldh import PinnConfig, PinnExperiment, train_pinn
    from enzopt.pipeline.fit import _prior

    init()
    OUT.mkdir(parents=True, exist_ok=True)
    data = make_data()
    prior = _prior(G["p"])
    p = G["p"]
    report: dict = {"truth": {"t50_s": T50, "hill_n": HILL}, "methods": {}}
    cands = []  # (method, seed, kf, kr, act)

    # PINN: 8 seeds x 8 restarts, with and without a(t)
    exps = [PinnExperiment(n, p.experiments[n].initial, G["Et"], T_OBS, data[n]) for n in TRAIN]
    for kind in ["none", "shared"]:
        cfg = PinnConfig(
            seeds=[s + 1000 * j for s in range(8) for j in range(8)],
            sigma=SIGMA,
            device="cuda",
            dtype="float32",
            inactivation=kind,
            inact_k=K_STEPS,
            w_inact=W_INACT,
        )
        t0 = time.time()
        res = train_pinn(exps, prior, cfg)
        report["methods"][f"pinn_{kind}"] = {"train_seconds": time.time() - t0, "fits": len(res)}
        cands += [(f"pinn_{kind}", r.seed, r.kf.tolist(), r.kr.tolist(), r.activity) for r in res]
        print(f"pinn_{kind}: {time.time() - t0:.1f} s for {len(res)} fits", flush=True)

    # classical: 8 starts each
    jobs = [(kind, s, data, prior) for kind in ["none", "exp", "flex"] for s in range(8)]
    t0 = time.time()
    with ProcessPoolExecutor(22, initializer=init) as ex:
        cls = list(ex.map(cls_fit, jobs))
    wall = time.time() - t0
    for kind in ["none", "exp", "flex"]:
        rs = [r for r in cls if r["method"] == f"cls_{kind}"]
        report["methods"][f"cls_{kind}"] = {
            "fits": len(rs),
            "seconds_per_fit_median": float(np.median([r["seconds"] for r in rs])),
            "ode_solves_per_fit_median": float(np.median([r["nfev"] + r["njev"] * (10 + N_EXTRA[kind]) for r in rs])),
        }
    report["classical_wall_seconds_all_24"] = wall
    cands += [(r["method"], r["start"], r["kf"], r["kr"], r["act"]) for r in cls]
    print(f"classical: {wall:.1f} s wall for {len(cls)} fits", flush=True)

    # re-check every candidate with the ODE
    with ProcessPoolExecutor(22, initializer=init) as ex:
        checks = list(ex.map(recheck, [(c[2], c[3], c[4], data) for c in cands], chunksize=4))
    # one sample per seed/start: best restart by training re-check (PINN), then the 3σ rule
    samples: dict = {}
    for c, chk in zip(cands, checks, strict=True):
        key = (c[0], c[1] % 1000)
        if key not in samples or chk["train"] < samples[key][1]["train"]:
            samples[key] = (c, chk)
    tgrid = np.linspace(0.0, T_OBS[-1], 200)
    a_true = act_curve(truth_act(), tgrid)

    def entry(c, chk):
        a_err = float(np.sqrt(np.mean((act_curve(c[4], tgrid) - a_true) ** 2)))
        return {"seed": c[1] % 1000, "train": chk["train"], "Dp": chk["Dp"], "activity_rmse": a_err}

    kept: dict = {}
    for (m, _), (c, chk) in samples.items():
        if chk["train"] <= 3 * SIGMA:
            kept.setdefault(m, []).append((c, chk))
    for m in report["methods"]:
        ks = [entry(*k) for k in kept.get(m, [])]
        best_c, best_chk = min((v for (mm, _), v in samples.items() if mm == m), key=lambda v: v[1]["train"])
        report["methods"][m].update(
            {
                "kept": len(ks),
                "train_rmse_median": float(np.median([k["train"] for k in ks])) if ks else None,
                "holdout_Dp_rmse_median": float(np.median([k["Dp"] for k in ks])) if ks else None,
                "activity_rmse_median": float(np.median([k["activity_rmse"] for k in ks])) if ks else None,
                "best": {**entry(best_c, best_chk), "kf": best_c[2], "kr": best_c[3], "act": _jsonable(best_c[4])},
            }
        )

    # design: per-sample dose, robust dose = max over kept samples, truth conversion at that dose
    from enzopt.core.massaction import haldane_kr

    tr_kr = haldane_kr(G["truth_kf"], G["truth_kr"], G["Kp"], 5)
    with ProcessPoolExecutor(22, initializer=init) as ex:
        jobs = [(m, (k[0][2], k[0][3], k[0][4])) for m, ks in kept.items() for k in ks]
        doses = list(ex.map(dose, [j[1] for j in jobs]))
        truth_dose = list(ex.map(dose, [(G["truth_kf"].tolist(), tr_kr.tolist(), truth_act())]))[0]
        per: dict = {}
        for (m, _), d in zip(jobs, doses, strict=True):
            per.setdefault(m, []).append(d)
        robust = {m: float(np.nanmax(v)) for m, v in per.items()}
        conv = dict(zip(robust, ex.map(truth_conversion, list(robust.values())), strict=True))
        conv_each = {m: list(ex.map(truth_conversion, v)) for m, v in per.items()}
    report["design"] = {
        "rho": RHO,
        "time_s": T_DESIGN,
        "target_conversion": CONV,
        "truth_dose_nM": truth_dose * 1e6,
        "methods": {
            m: {
                "robust_dose_nM": robust[m] * 1e6,
                "truth_conversion_at_dose": conv[m],
                "per_sample_nM": [d * 1e6 for d in per[m]],
                "truth_conversion_per_sample_dose": conv_each[m],
            }
            for m in robust
        },
    }
    report["samples"] = {
        m: [{**entry(*k), "kf": k[0][2], "kr": k[0][3], "act": _jsonable(k[0][4])} for k in ks]
        for m, ks in kept.items()
    }
    report["data"] = {n: data[n].tolist() for n in data}
    (OUT / "results.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    figure(report)
    for m, r in report["methods"].items():
        d = report["design"]["methods"].get(m, {})
        print(
            f"{m:12s} kept {r['kept']:2d} train {r['train_rmse_median']} Dp {r['holdout_Dp_rmse_median']} "
            f"a-err {r['activity_rmse_median']} dose {d.get('robust_dose_nM')} conv {d.get('truth_conversion_at_dose')}"
        )
    print("truth dose nM", truth_dose * 1e6)


def figure(report):
    """Four panels from results.json: fits of sets A and B, a(t), and the enzyme dose of each method."""
    from enzopt.figures import DOUBLE, _mpl, color, new_figure

    init()
    colors = {
        "pinn_none": "#999999",
        "pinn_shared": "#222222",
        "cls_none": color(5),
        "cls_exp": color(0),
        "cls_flex": color(1),
    }
    data = {n: np.asarray(v) for n, v in report["data"].items()}
    meth = report["methods"]
    fig, ax = new_figure(DOUBLE, 4.8, ncols=2, nrows=2)
    tf = np.linspace(0.0, T_OBS[-1], 300)
    for j, name in enumerate(["A", "B"]):
        a = ax[0, j]
        a.plot(T_OBS, data[name], "o", ms=2.2, mfc="none", mew=0.5, color=color(2), label="data")
        for m, ls in [("pinn_none", "--"), ("pinn_shared", "-")]:
            b = meth[m]["best"]
            tag = "" if meth[m]["kept"] else f" (rejected, RMSE {b['train'] / SIGMA:.0f}σ)"
            a.plot(
                tf, a340(name, b["kf"], b["kr"], b["act"], tf), color=colors[m], lw=1.1, ls=ls, label=LABELS[m] + tag
            )
        a.set_title(f"({'ab'[j]}) training set {name}", loc="left")
        a.set_xlabel("time (s)")
        a.set_ylabel("A340")
        if j == 0:
            a.legend(frameon=False, fontsize=5.5, loc="upper right")
    a = ax[1, 0]
    a.plot(tf, act_curve(truth_act(), tf), color="black", lw=2.2, label="truth")
    for m, ls in [("pinn_shared", "-"), ("cls_flex", ":"), ("cls_exp", "--")]:
        for i, k in enumerate(report["samples"].get(m, [])):
            lab = LABELS[m] if i == 0 else None
            a.plot(tf, act_curve(k["act"], tf), color=colors[m], lw=0.8, ls=ls, alpha=0.85, label=lab)
    a.set_ylim(0, 1.05)
    a.set_title("(c) enzyme activity a(t), every kept sample", loc="left")
    a.set_xlabel("time since enzyme added (s)")
    a.set_ylabel("active fraction")
    a.legend(frameon=False, fontsize=5.5)
    a = ax[1, 1]
    dm = report["design"]["methods"]
    ms = [m for m in LABELS if m in dm]
    need = report["design"]["truth_dose_nM"]
    for xi, m in enumerate(ms):
        v = np.asarray(dm[m]["per_sample_nM"])
        xs = xi + np.linspace(-0.18, 0.18, len(v))
        fail = v < need
        a.plot(xs[~fail], v[~fail], "o", ms=3, color=colors[m], mfc="none", mew=0.8)
        a.plot(xs[fail], v[fail], "x", ms=4, color=color(3), mew=0.9)
        a.plot([xi - 0.3, xi + 0.3], [dm[m]["robust_dose_nM"]] * 2, color=colors[m], lw=2.0)
        a.text(xi + 0.32, dm[m]["robust_dose_nM"], f"{dm[m]['robust_dose_nM']:.0f}", va="center", fontsize=5.5)
    a.axhline(need, color=color(3), ls=":", lw=0.9, label=f"true enzyme needs {need:.0f} nM")
    a.plot([], [], "x", color=color(3), label="sample whose own dose fails on the true enzyme")
    a.plot([], [], "-", color="#555555", lw=2.0, label="robust dose (max over samples)")
    a.set_yscale("log")
    a.set_xticks(range(len(ms)), [LABELS[m].replace(", ", ",\n") for m in ms], fontsize=5.2)
    a.set_xlim(-0.6, len(ms) - 0.2)
    a.set_ylabel("enzyme for 99 % in 300 s (nM)")
    a.set_title("(d) design dose at ρ = 16 (dots: samples)", loc="left")
    a.legend(frameon=False, fontsize=5.2, loc="lower right")
    fig.savefig(OUT / "fig_inactivation.png", dpi=220)
    _mpl().close(fig)


if __name__ == "__main__":
    import sys

    if "--figure-only" in sys.argv:
        figure(json.loads((OUT / "results.json").read_text(encoding="utf-8")))
    else:
        main()
