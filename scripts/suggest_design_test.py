"""Which experiments let `enzopt suggest` find a missing step? Hidden dead-end complex × experiment plan.

    uv run python scripts/suggest_design_test.py [OUT_DIR]       # default: runs/suggest_design_test

Synthetic LDH data (truth = evals/synthetic.yaml constants, K' 23, σ 0.003 A340) whose true mechanism has one extra
dead-end complex with K_i = 5 mM (k_on 10³ /mM/s). The same truth is measured under three plans:

    plan        A, B (train) and C, D' (held out): the ldh_ph10 plan, lactate ≈ 10 mM in every set
    plan+Lac    plan + R30 (NAD⁺ 0.2, lactate 30) + F0 (NADH 0.2, pyruvate 0.2, no lactate) + P16 (pyruvate 1.6)
    plan+Pyr    plan + R30 + F0 + P50 (pyruvate 5) + P100 (pyruvate 10)

`enzopt suggest` (two least-squares checks) runs on each and the verdicts are tabulated. Result 2026-10-10: the plan
alone misses both complexes; lactate-varied sets reveal EN·Lac (K_i 4.8 mM); only pyruvate ≥ 5 mM reveals EQ·Pyr
(K_i 4.9 mM) — at 1.6 mM the five steps still absorb it by shifting their rate constants.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

from enzopt.core.massaction import MassAction, haldane_kr, simulate
from enzopt.core.network import x0_vector
from enzopt.core.project import load_project
from enzopt.core.yamlio import load_yaml
from enzopt.pipeline.suggest import run_suggest

ROOT = Path(__file__).resolve().parents[1]
TRUTHS = {"EN·Lac": ("EN + Lac <=> ENL", "ENL"), "EQ·Pyr": ("EQ + Pyr <=> EQP", "EQP")}
KON, KI, SIGMA = 1e3, 5.0, 0.003
T = np.arange(5.0, 900.0, 10.0)
PLAN = {
    "A": ({"NADH": 0.2, "NAD": 0.0, "Pyr": 0.2, "Lac": 9.8}, "train"),
    "B": ({"NADH": 0.0, "NAD": 0.2, "Pyr": 0.0, "Lac": 10.0}, "train"),
    "C": ({"NADH": 0.1, "NAD": 0.1, "Pyr": 0.1, "Lac": 9.9}, "holdout"),
    "Dp": ({"NADH": 0.2, "NAD": 0.0, "Pyr": 0.5, "Lac": 9.5}, "holdout"),
}
R30 = {"R30": ({"NADH": 0.0, "NAD": 0.2, "Pyr": 0.0, "Lac": 30.0}, "train")}
F0 = {"F0": ({"NADH": 0.2, "NAD": 0.0, "Pyr": 0.2, "Lac": 0.0}, "train")}
DESIGNS = {
    "plan": PLAN,
    "plan+Lac": {**PLAN, **R30, **F0, "P16": ({"NADH": 0.1, "NAD": 0.0, "Pyr": 1.6, "Lac": 0.0}, "train")},
    "plan+Pyr": {
        **PLAN,
        **R30,
        **F0,
        "P50": ({"NADH": 0.1, "NAD": 0.0, "Pyr": 5.0, "Lac": 0.0}, "train"),
        "P100": ({"NADH": 0.1, "NAD": 0.0, "Pyr": 10.0, "Lac": 0.0}, "train"),
    },
}


def make_project(folder: Path, truth: str, sets: dict) -> Path:
    step, cplx = TRUTHS[truth]
    template = (ROOT / "projects" / "example_ldh_missing_step" / "project.yaml").read_text(encoding="utf-8")
    head, rest = template.split("experiments:", 1)
    tail = rest[rest.index("\nidentification:") :]
    lines = []
    for n, (ini, role) in sets.items():
        body = ", ".join(f"{k}: {v:g}" for k, v in ini.items())
        lines.append(f"  {n}: {{file: data/set{n}.csv, {body}, Et_nM: 24.23, role: {role}}}")
    name = f"design_{truth.replace('·', '')}_{folder.name}"
    text = head.replace("name: example_ldh_missing_step", f"name: {name}") + "experiments:\n" + "\n".join(lines) + tail
    (folder / "data").mkdir(parents=True, exist_ok=True)
    (folder / "project.yaml").write_text(text, encoding="utf-8")
    p = load_project(folder)
    tr = load_yaml(ROOT / "evals" / "synthetic.yaml")["truth"]
    kf = np.array(tr["kf"], float)
    kr = haldane_kr(kf, np.array(list(tr["kr"]) + [np.nan], float), float(tr["Kp"]), 5)
    net = p.network.with_extra_steps([step], extra_species=[cplx])
    k = net.rate_vector(list(kf) + [KON], list(kr) + [KON * KI])
    model, idx = MassAction.from_network(net), net.index()
    rng = np.random.default_rng(20261010)
    for n, e in p.experiments.items():
        atol = np.where(np.isin(net.species, net.enzyme_species), e.Et * 1e-9, 1e-12)
        X = simulate(model, k, x0_vector(net, e.initial, e.Et), T, atol=atol)
        y = sum(c * X[:, idx[s]] for s, c in p.observable_weights().items()) + rng.normal(0.0, SIGMA, len(T))
        rows = "".join(f"{t:g},{v:.6f}\n" for t, v in zip(T, y, strict=True))
        (folder / "data" / f"set{n}.csv").write_text("time_s,A340\n" + rows, encoding="utf-8")
    return folder


def main() -> None:
    out = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs" / "suggest_design_test"
    rows = []
    for truth in TRUTHS:
        for design, sets in DESIGNS.items():
            if truth == "EN·Lac" and design == "plan+Pyr":
                continue  # the lactate sets are the same as in plan+Lac
            proj = make_project(out / f"{truth.replace('·', '')}_{design}", truth, sets)
            res = run_suggest(proj, verify_top=2, runs_root=out / "runs", log=lambda *a: None)
            v = json.loads((res / "suggest.json").read_text(encoding="utf-8"))["verdict"]
            rows.append((truth, design, v["suggested_step"], v["Ki_mM"], v["note"]))
            found = v["suggested_step"] or "없음"
            print(f"{truth:7s} {design:9s} → {found}  Ki {v['Ki_mM']}  | {v['note']}", flush=True)


if __name__ == "__main__":
    main()
