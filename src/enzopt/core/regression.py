"""CADKIMM regression cases (T1-회귀): the 149 class batch (scipy backend) and the synthetic-sample design."""

from __future__ import annotations

import numpy as np

from .design import AssayDesign, default_rho_grid, param_samples, rho_min_equilibrium, run_design
from .massaction import MassAction, simulate
from .network import Network
from .validate import class_consistency, conservation_laws, validate_closed

ONE_STEP_SPECIES = ["NADH", "NAD", "Pyr", "Lac"]
DESIGN_STEPS = [
    "E + NADH <=> EN",
    "EN + Pyr <=> ENP",
    "ENP <=> EQL",
    "EQL <=> EQ + Lac",
    "EQ <=> E + NAD",
    "EQ + Pyr <=> EQP",
]
DESIGN_SPECIES = ["E", "NADH", "EN", "Pyr", "ENP", "EQL", "Lac", "EQ", "NAD", "EQP"]
DESIGN_ENZYME = ["E", "EN", "ENP", "EQL", "EQ", "EQP"]


def one_step_network() -> Network:
    return Network.from_strings("ldh_one_step", ONE_STEP_SPECIES, ["Pyr + NADH <=> Lac + NAD"])


def sample_class(rng, n: int, N_tot=0.2, S_tot=10.0, D_tot=0.0) -> list[dict]:
    """Positive initial states on one compatibility class (CADKIMM batch_class.sample_class)."""
    lo, hi = max(0.0, D_tot), min(N_tot, S_tot + D_tot)
    xs = rng.uniform(lo + 1e-4, hi - 1e-4, n)
    return [{"NADH": x, "NAD": N_tot - x, "Pyr": x - D_tot, "Lac": S_tot - (x - D_tot)} for x in xs]


def batch149_cases(seed: int = 7, n: int = 12) -> list[tuple[str, dict]]:
    rng = np.random.default_rng(seed)
    cases = [(f"cls_{i:02d}", x0) for i, x0 in enumerate(sample_class(rng, n))]
    cases.append(("other_class", {"NADH": 0.20, "NAD": 0.0, "Pyr": 0.50, "Lac": 9.50}))
    return cases


def summarize_batch(results: dict, rxns, rates) -> dict:
    """results: {name: (t, {species: array})} → validation per run, class consistency and the headline numbers."""
    species = ONE_STEP_SPECIES
    W = conservation_laws(MassAction(species, rxns).N)
    finals, cons, per_run = [], [], {}
    n_pass = 0
    for name, (t, S) in results.items():
        r = validate_closed(t, S, rxns, rates)
        per_run[name] = r
        n_pass += int(r["ok"])
        X = np.column_stack([S[s] for s in species])
        finals.append(X[-1])
        cons.append(X[0] @ W.T)
    cc = class_consistency(finals, cons)
    same = [results[k][1]["NADH"][-1] for k in results if k.startswith("cls_")]
    other = results["other_class"][1]["NADH"][-1] if "other_class" in results else float("nan")
    return {
        "n_runs": len(results),
        "n_pass": n_pass,
        "per_run": per_run,
        "same_class_final_NADH": float(np.mean(same)),
        "same_class_spread": float(np.ptp(same)),
        "other_class_final_NADH": float(other),
        "class_spread": {str(np.round(k, 4).tolist()): v for k, v in cc.items()},
        "n_classes": len(cc),
    }


def batch149_scipy(k1: float = 2.0, Kp: float = 23.0, t_end: float = 600.0, n_samples: int = 2048) -> dict:
    net = one_step_network()
    rxns = net.reactions()
    k = net.rate_vector([k1], [k1 / Kp])
    model = MassAction.from_network(net)
    t = np.linspace(0.0, t_end, n_samples)
    results = {}
    for name, x0 in batch149_cases():
        X = simulate(model, k, np.array([x0[s] for s in net.species]), t, atol=1e-12)
        results[name] = (t, {s: X[:, i] for i, s in enumerate(net.species)})
    out = summarize_batch(results, rxns, k)
    out["backend"] = "scipy"
    return out


def design_network() -> Network:
    return Network.from_strings("ldh_design", DESIGN_SPECIES, DESIGN_STEPS, DESIGN_ENZYME)


def design_regression(quick: bool = False, workers: int | None = None, progress=None) -> dict:
    """CADKIMM design_optimize.py (synthetic samples) with enzopt.core.design."""
    net = design_network()
    d = AssayDesign(net, Kp=23.0)
    n = 3 if quick else 9
    samples = [net.rate_vector(kf, kr) for kf, kr in param_samples(n)]
    rmin = rho_min_equilibrium(d.conversion, d.Kp)
    res = run_design(
        d, samples, default_rho_grid(rmin, quick), iters=12 if quick else 18, workers=workers, progress=progress
    )
    out = res.summary()
    out["table"] = [
        {"rho": r, "Et_nominal_nM": a * 1e6, "Et_robust_nM": b * 1e6}
        for r, a, b in zip(res.rhos, res.Et_nominal, res.Et_robust, strict=True)
    ]
    out["check"] = res.check
    out["equilibrium_limit"] = res.equilibrium_limit
    return out
