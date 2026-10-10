"""Synthetic LDH progress curves whose true mechanism has one step more than project.yaml says (for `enzopt suggest`).

    uv run python projects/example_ldh_missing_step/make_data.py                  # truth = 5 steps + EN + Lac <=> ENL
    uv run python projects/example_ldh_missing_step/make_data.py --without-step D  # negative control into folder D

The extra step is an abortive complex: lactate bound to the NADH form of the enzyme (K_i = 5 mM, k_on 10³ /mM/s).
Lactate is varied (0, 10, 30 mM) because a complex that only forms with lactate is invisible when lactate is constant.
Rate constants otherwise = evals/synthetic.yaml truth, K' = 23, noise σ = 0.003 A340. data/truth.json keeps the answer.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from enzopt.core.massaction import MassAction, haldane_kr, simulate
from enzopt.core.network import x0_vector
from enzopt.core.project import load_project
from enzopt.core.yamlio import load_yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
EXTRA = "EN + Lac <=> ENL"
KON, KI = 1e3, 5.0
T_OBS = np.arange(5.0, 900.0, 10.0)
SIGMA = 0.003


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--without-step", metavar="DIR", help="write a copy of this project with data from the 5 steps only"
    )
    a = ap.parse_args()
    target = HERE
    if a.without_step:
        target = Path(a.without_step)
        if target.exists():
            shutil.rmtree(target)
        shutil.copytree(HERE, target, ignore=shutil.ignore_patterns("data"))
    p = load_project(target)
    tr = load_yaml(ROOT / "evals" / "synthetic.yaml")["truth"]
    kf = np.array(tr["kf"], float)
    kr = haldane_kr(kf, np.array(list(tr["kr"]) + [np.nan], float), float(tr["Kp"]), 5)
    if a.without_step:
        net, k = p.network, p.network.rate_vector(kf, kr)
    else:
        net = p.network.with_extra_steps([EXTRA], extra_species=["ENL"])
        k = net.rate_vector(list(kf) + [KON], list(kr) + [KON * KI])
    model = MassAction.from_network(net)
    idx = net.index()
    rng = np.random.default_rng(20261010)
    (target / "data").mkdir(exist_ok=True)
    for e in p.experiments.values():
        atol = np.where(np.isin(net.species, net.enzyme_species), e.Et * 1e-9, 1e-12)
        X = simulate(model, k, x0_vector(net, e.initial, e.Et), T_OBS, atol=atol)
        a340 = sum(c * X[:, idx[s]] for s, c in p.observable_weights().items()) + rng.normal(0.0, SIGMA, len(T_OBS))
        rows = "".join(f"{t:g},{v:.6f}\n" for t, v in zip(T_OBS, a340, strict=True))
        (target / e.file.relative_to(p.root)).write_text("time_s,A340\n" + rows, encoding="utf-8")
    truth = {
        "kf": kf.tolist(),
        "kr": kr.tolist(),
        "Kp": float(tr["Kp"]),
        "extra_step": None if a.without_step else {"step": EXTRA, "k_on": KON, "Ki_mM": KI},
        "noise_sigma": SIGMA,
    }
    (target / "data" / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print("wrote", target / "data")


if __name__ == "__main__":
    main()
