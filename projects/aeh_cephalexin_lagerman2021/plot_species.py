"""All four measured species against the identified models of one run (the stage-3 figures show only PGME).

    uv run python projects/aeh_cephalexin_lagerman2021/plot_species.py runs/aeh_cephalexin_lagerman2021/<run>

Writes ``species_fit.png`` (one row per experiment: PGME, 7-ADCA, cephalexin, phenylglycine; points = HPLC data,
coloured lines = kept PINN samples re-solved as ODEs, dash-dot = classical fits) and ``species_rmse.csv``
(RMSE in mM per experiment, species and model) into the project's ``figures/`` folder.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from enzopt.core.activity import activity_curve  # noqa: E402
from enzopt.core.massaction import MassAction  # noqa: E402
from enzopt.core.project import load_project  # noqa: E402
from enzopt.pipeline.common import curve, load_samples, rmse, sample_rates  # noqa: E402

HERE = Path(__file__).resolve().parent
LABELS = {"PGME_mM": "PGME", "ADCA_mM": "7-ADCA", "CEX_mM": "cephalexin", "PG_mM": "phenylglycine"}
COLORS = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]


def main(run: Path) -> None:
    p = load_project(run / "inputs")
    net = p.network
    model = MassAction.from_network(net)
    signals = p.all_signals()
    samples = load_samples(run)
    classical = []
    for f in sorted((run / "3_fit" / "baseline").glob("fit_start*.json")):
        c = json.loads(f.read_text(encoding="utf-8"))
        classical.append(net.rate_vector(c["kf"], c["kr"]))
    models = [
        (f"PINN sample {s['seed']}", sample_rates(net, s), COLORS[i % 8], "-", s.get("activity"))
        for i, s in enumerate(samples)
    ]
    models += [(f"classical {i}", k, "0.35", "-.", None) for i, k in enumerate(classical)]

    exps = list(p.experiments.values())
    t = np.linspace(0.0, 1800.0, 361)
    fig, axes = plt.subplots(len(exps), len(signals), figsize=(3.0 * len(signals), 2.0 * len(exps)), squeeze=False)
    rows = []
    for r, e in enumerate(exps):
        d = np.genfromtxt(e.file, delimiter=",", names=True)
        for c, sg in enumerate(signals):
            ax = axes[r, c]
            ax.plot(d["time_s"], d[sg["name"]], "o", ms=3.5, mfc="none", color="k", label="data", zorder=5)
            for name, k, col, ls, act in models:
                y = curve(net, k, e.initial, e.Et, t, sg["coefficients"], model=model, activity=act)
                ax.plot(t, y, ls, lw=1.1, color=col, label=name)
                yd = curve(net, k, e.initial, e.Et, d["time_s"], sg["coefficients"], model=model, activity=act)
                rows.append([e.name, e.role, sg["name"], name, f"{rmse(yd, d[sg['name']]):.3f}"])
            if r == 0:
                ax.set_title(LABELS.get(sg["name"], sg["name"]), fontsize=10)
            if c == 0:
                star = "*" if e.role == "train" else " (held out)"
                ax.set_ylabel(f"{e.name}{star}\nmM", fontsize=9)
            if r == len(exps) - 1:
                ax.set_xlabel("time (s)")
            ax.tick_params(labelsize=8)
    h, lab = axes[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=min(len(lab), 6), frameon=False, fontsize=9)
    fig.suptitle("Cephalexin synthesis by AEH (Lagerman 2021): data and identified models (* = training set)", y=0.995)
    fig.tight_layout(rect=(0, 0.03, 1, 0.985))
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.savefig(out / "species_fit.png", dpi=110)
    acts = [(s["seed"], s.get("activity")) for s in samples if s.get("activity")]
    if acts:  # shared enzyme-activity curves a(t) of the samples
        fa, ax = plt.subplots(figsize=(4.2, 2.8))
        for i, (seed, a) in enumerate(acts):
            ax.plot(t, activity_curve(a, t), color=COLORS[i % 8], label=f"PINN sample {seed}")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("enzyme activity a(t)")
        ax.set_ylim(0, 1.05)
        ax.legend(frameon=False, fontsize=8)
        fa.tight_layout()
        fa.savefig(out / "activity.png", dpi=110)
    with open(out / "species_rmse.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(["experiment", "role", "signal", "model", "rmse_mM"])
        w.writerows(rows)
    print(f"wrote {out / 'species_fit.png'} and species_rmse.csv ({len(samples)} samples, {len(classical)} classical)")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
