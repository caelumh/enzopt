"""Synthetic LDH progress curves with enzyme inactivation for this example (truth = evals/synthetic.yaml constants).

    uv run python projects/example_ldh_inactivation/make_data.py

The enzyme loses activity after a lag: a(t) = 1/(1 + (t/300 s)^3), t = time since the enzyme was added (same in
every cuvette). Noise σ = 0.003 A340, no other instrument effects. data/truth.json keeps the answer.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from enzopt.core.activity import enzyme_loss
from enzopt.core.massaction import MassAction, haldane_kr, simulate
from enzopt.core.network import x0_vector
from enzopt.core.project import load_project
from enzopt.core.yamlio import load_yaml

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
ACTIVITY = {"kind": "hill", "t50": 300.0, "n": 3.0}
T_OBS = np.arange(5.0, 900.0, 10.0)
SIGMA = 0.003


def main() -> None:
    p = load_project(HERE)
    tr = load_yaml(ROOT / "evals" / "synthetic.yaml")["truth"]
    kf = np.array(tr["kf"], float)
    kr = haldane_kr(kf, np.array(list(tr["kr"]) + [np.nan], float), float(tr["Kp"]), 5)
    net = p.network
    model = MassAction.from_network(net)
    k = net.rate_vector(kf, kr)
    rng = np.random.default_rng(20261010)
    (HERE / "data").mkdir(exist_ok=True)
    for e in p.experiments.values():
        Et = e.Et
        atol = np.where(np.isin(net.species, net.enzyme_species), Et * 1e-9, 1e-12)
        X = simulate(
            model,
            k,
            x0_vector(net, e.initial, Et),
            T_OBS,
            atol=atol,
            loss=enzyme_loss(net.species, net.enzyme_species, ACTIVITY),
        )
        idx = net.index()
        a340 = sum(c * X[:, idx[s]] for s, c in p.observable_weights().items()) + rng.normal(0.0, SIGMA, len(T_OBS))
        rows = "".join(f"{t:g},{v:.6f}\n" for t, v in zip(T_OBS, a340, strict=True))
        Path(e.file).write_text("time_s,A340\n" + rows, encoding="utf-8")
    truth = {"kf": kf.tolist(), "kr": kr.tolist(), "Kp": float(tr["Kp"]), "activity": ACTIVITY, "noise_sigma": SIGMA}
    (HERE / "data" / "truth.json").write_text(json.dumps(truth, indent=1), encoding="utf-8")
    print("wrote", sorted(str(Path(e.file).name) for e in p.experiments.values()))


if __name__ == "__main__":
    main()
