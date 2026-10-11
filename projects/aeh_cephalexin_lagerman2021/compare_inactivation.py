"""Compare enzyme-inactivation hypotheses by least squares on the full ODE (README "실활 단계 고르기").

    uv run python projects/aeh_cephalexin_lagerman2021/compare_inactivation.py runs/aeh_cephalexin_lagerman2021/<run>

<run> is a run of the network without step 13 (its best classical fit is the warm start). For each hypothesis (one
extra one-way step X -> ED to a dead enzyme form, or none) the script fits (a) the 6 training experiments, then
predicts m4 and m7, and (b) all 8 experiments. It prints the RMSE per experiment (mean over the 4 signals, mM) and
writes ``figures/inactivation_hypotheses.json``. 14 fits in parallel, about 3 minutes on 24 cores.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from enzopt.core.massaction import MassAction, SimulationError, SolverSettings, atol_vector, simulate
from enzopt.core.network import Network, x0_vector
from enzopt.core.project import load_project
from enzopt.core.thermo import layout

HERE = Path(__file__).resolve().parent
FIT = SolverSettings(method="Radau", rtol=1e-6, atol_enzyme_rel=1e-7, atol_other=1e-10)
TRAIN = ("m0", "m1", "m2", "m3", "m5", "m6")
HYP = {
    "none": [],
    "E": ["E -> ED"],
    "ES": ["ES -> ED"],
    "ESS": ["ESS -> ED"],
    "EA": ["EA -> ED"],
    "EAS": ["EAS -> ED"],
    "EAN": ["EAN -> ED"],
}


def setup(base_run: Path, hyp: str):
    p = load_project(base_run / "inputs")  # the run's own copy: the network without step 13
    base = p.network
    extra = HYP[hyp]
    species = list(base.species) + (["ED"] if extra else [])
    enz = list(base.enzyme_species) + (["ED"] if extra else [])
    net = Network.from_strings(f"aeh_{hyp}", species, [str(s) for s in base.steps] + extra, enz)
    lay = layout(net, p.overall)
    pr = p.prior_vector()
    nb = len(base.steps)
    prior = np.concatenate([pr[:nb], [-2.0] * len(extra), pr[nb:]])
    c = json.loads((base_run / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))["comparisons"]["classical"]
    best = min(c["fits"], key=lambda f: f["cost"])
    theta0 = lay.theta(list(best["kf"]) + [1e-2] * len(extra), list(best["kr"]) + [0.0] * len(extra))
    exps = []
    for e in p.experiments.values():
        d = np.genfromtxt(e.file, delimiter=",", names=True)
        exps.append((e.name, e.initial, e.Et, d))
    return p, net, lay, prior, theta0, exps


def predict(net, lay, theta, exps, signals):
    kf, kr = lay.rates(theta)
    k = net.rate_vector(kf, kr)
    model = MassAction.from_network(net)
    idx = net.index()
    out = {}
    for name, initial, Et, d in exps:
        atol = atol_vector(net.species, net.enzyme_species, Et, FIT)
        X = simulate(model, k, x0_vector(net, initial, Et), d["time_s"], atol=atol, settings=FIT)
        out[name] = {sg["name"]: sum(c * X[:, idx[s]] for s, c in sg["coefficients"].items()) for sg in signals}
    return out


def job(args):
    base_run, hyp, mode = args
    t0 = time.time()
    p, net, lay, prior, theta0, exps = setup(base_run, hyp)
    signals = p.all_signals()
    fit_exps = [x for x in exps if x[0] in TRAIN or (mode == "all8" and x[0] == "m7")]
    n_exp, n_sig = len(fit_exps), len(signals)

    def res(theta):
        r = []
        try:
            pred = predict(net, lay, theta, fit_exps, signals)
        except (SimulationError, ValueError, FloatingPointError):
            pred = None
        for name, _i, _E, d in fit_exps:
            for sg in signals:
                w = 1.0 / np.sqrt(len(d) * n_exp * n_sig)
                if pred is None:
                    r.append(np.full(len(d), 1e3 * w))
                else:
                    r.append(np.nan_to_num((pred[name][sg["name"]] - d[sg["name"]]) / sg["sigma"] * w, nan=1e3))
        r.append(np.sqrt(1e-2 / (2 * 1.5**2)) * (theta - prior))  # the prior term of the classical fit
        return np.concatenate(r)

    sol = least_squares(res, theta0, method="trf", max_nfev=150, diff_step=1e-4)
    pred = predict(net, lay, sol.x, exps, signals)
    rm = {
        name: float(np.mean([np.sqrt(np.mean((pred[name][s["name"]] - d[s["name"]]) ** 2)) for s in signals]))
        for name, _i, _E, d in exps
    }
    kf, _ = lay.rates(sol.x)
    return {
        "hypothesis": hyp,
        "fit": mode,
        "cost": float(sol.cost),
        "rmse_mM": rm,
        "k_inact": [float(x) for x in kf[len(p.network.steps) :]],
        "seconds": time.time() - t0,
    }


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    base_run = Path(sys.argv[1]).resolve()
    jobs = [(base_run, h, m) for h in HYP for m in ("train6", "all8")]
    with ProcessPoolExecutor(max_workers=len(jobs)) as ex:
        results = list(ex.map(job, jobs))
    out = HERE / "figures" / "inactivation_hypotheses.json"
    out.write_text(json.dumps(results, indent=1), encoding="utf-8", newline="\n")
    for r in results:
        rm = r["rmse_mM"]
        tr = np.mean([rm[k] for k in TRAIN])
        k_in = ", ".join(f"{x:.3g}" for x in r["k_inact"])
        print(
            f"{r['hypothesis']:5s} {r['fit']:6s} cost {r['cost']:7.3f} train {tr:5.2f} m4 {rm['m4']:5.2f} "
            f"m6 {rm['m6']:5.2f} m7 {rm['m7']:6.2f} k_inact [{k_in}] /s"
        )


if __name__ == "__main__":
    main()
