"""Compare the PINN enzyme representations on the realistic synthetic example (evidence for QUESTIONS Q3, 2026-10-09).

    uv run python scripts/compare_pinn_representation.py [--device cuda|cpu] [--out docs/pinn_representation]

For each representation the same seeds are trained on sets A, B; the learned rate constants are re-solved with the ODE
(scipy) and the RMSE against every experiment is tabulated:
  qss          enzyme forms at quasi-steady state (King–Altman), enzopt default
  full_gross   enzyme forms as network outputs, enzyme residuals scaled by the gross flux (CADKIMM methodology 4.3)
  full_slow    enzyme forms as network outputs, enzyme residuals scaled by the substrate rate (stricter physics)
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from enzopt.core.massaction import MassAction
from enzopt.core.project import load_project
from enzopt.pinn.ldh import PinnConfig, PinnExperiment, resolve_device, train_pinn
from enzopt.pipeline.common import rmse, safe_curve

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = {
    "qss": dict(representation="qss"),
    "full_gross": dict(representation="full", fast_norm="gross"),
    "full_slow": dict(representation="full", fast_norm="slow"),
}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="auto")
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--out", default=str(ROOT / "docs" / "pinn_representation"))
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    p = load_project(ROOT / "projects" / "example_ldh_synth")
    data = {n: np.loadtxt(e.file, delimiter=",", skiprows=1) for n, e in p.experiments.items()}
    train = [
        PinnExperiment(n, e.initial, e.Et, data[n][:, 0], data[n][:, 1])
        for n, e in p.experiments.items()
        if e.role == "train"
    ]
    pr = p.prior
    prior = np.array(pr["kf_log10"] + pr["kr_log10"] + [pr["Kp_log10"]], float)
    device, dtype, _ = resolve_device(a.device, "auto")
    model = MassAction.from_network(p.network)
    rows, lines = [], []
    for name, kw in VARIANTS.items():
        cfg = PinnConfig(seeds=list(range(a.seeds)), device=device, dtype=dtype, **kw)
        t0 = time.time()
        res = train_pinn(train, prior, cfg)
        sec = time.time() - t0
        for r in res:
            k = p.network.rate_vector(r.kf, r.kr)
            row = {
                "variant": name,
                "seed": r.seed,
                "Kp": r.Kp,
                "seconds": round(sec, 1),
                "pinn_data_rmse": float(cfg.sigma * np.sqrt(max(r.loss_parts["data"], 0.0))),
            }
            for n, e in p.experiments.items():
                if e.role in ("train", "holdout"):
                    c = safe_curve(p.network, k, e.initial, e.Et, data[n][:, 0], p.observable_weights(), model=model)
                    row[f"ode_rmse_{n}"] = rmse(c, data[n][:, 1]) if c is not None else None
            rows.append(row)
            lines.append(" ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}" for k, v in row.items()))
            print(lines[-1], flush=True)
    keys = list(rows[0].keys())
    with open(out / "comparison.csv", "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(keys) + "\n")
        for r in rows:
            fh.write(",".join("" if r[k] is None else str(r[k]) for k in keys) + "\n")
    (out / "comparison.json").write_text(
        json.dumps({"device": device, "dtype": dtype, "rows": rows}, indent=1), encoding="utf-8"
    )


if __name__ == "__main__":
    main()
