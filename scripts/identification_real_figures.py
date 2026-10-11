"""Figures for part 4 of docs/identification/README.md: the same steps on measured literature data (AEH cephalexin
synthesis, projects/aeh_cephalexin_lagerman2021).

    uv run python scripts/identification_real_figures.py --run runs/aeh_cephalexin_lagerman2021/<run> \
        --holdout-test runs/aeh_cephalexin_lagerman2021/<suggest run, 6 sets fitted> \
        --all-test runs/aeh_cephalexin_lagerman2021/<suggest run, --all-train> [--esss-run <run with ESSS, m7 trained>]

fig10: four measured species against the kept PINN samples and the classical fits (training, held out inside and
outside the training range). fig11: the hypotheses of `enzopt suggest --candidate`, fitted on six sets (held-out
error) and on all eight. fig12: the design goal ``peak`` (when to stop) against the measured set with the same ρ.
fig13 (optional): m7 learnt with the ESSS steps. Text in the figures is English, like every enzopt figure.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from enzopt.core.massaction import MassAction
from enzopt.core.project import load_project
from enzopt.figures import DOUBLE, _mpl, color, new_figure
from enzopt.pipeline.common import curve, load_samples, sample_rates

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "identification"
LABELS = {"PGME_mM": "PGME", "ADCA_mM": "7-ADCA", "CEX_mM": "cephalexin", "PG_mM": "phenylglycine"}
SHORT = {
    "as written": "as written",
    "ESS + PGME <=> ESSS ; ESSS -> ED": "ESSS -> ED",
    "ESS + PG <=> ESS_PG": "ESS + PG",
    "2: ES -> EA + MeOH ; E + MeOH <=> E_MeOH": "MeOH binds E",
    "2: ES -> EA + MeOH ; E + MeOH -> ED": "MeOH kills E",
    "EG -> ED": "EG -> ED",
}


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220)
    _mpl().close(fig)
    print("wrote", OUT / f"{name}.png", flush=True)


def _models(run: Path):
    p = load_project(run / "inputs")
    net = p.network
    samples = load_samples(run)
    out = [(f"PINN sample {s['seed']}", sample_rates(net, s), color(i), "-", 1.0) for i, s in enumerate(samples)]
    for i, f in enumerate(sorted((run / "3_fit" / "baseline").glob("fit_start*.json"))):
        c = json.loads(f.read_text(encoding="utf-8"))
        out.append(("classical fit" if i == 0 else None, net.rate_vector(c["kf"], c["kr"]), "0.25", "-.", 0.9))
    return p, net, out


def species(run: Path, rows: list[str], name: str, title: str) -> None:
    p, net, models = _models(run)
    model = MassAction.from_network(net)
    signals = p.all_signals()
    t = np.linspace(0.0, 1800.0, 361)
    fig, axes = new_figure(DOUBLE, DOUBLE * 0.2 * len(rows) + 0.6, ncols=len(signals), nrows=len(rows))
    for r, ename in enumerate(rows):
        e = p.experiments[ename]
        d = np.genfromtxt(e.file, delimiter=",", names=True)
        for c, sg in enumerate(signals):
            a = axes[r, c]
            a.plot(d["time_s"], d[sg["name"]], "o", ms=3.2, mfc="none", color="k", label="data", zorder=5)
            for label, k, col, ls, lw in models:
                y = curve(net, k, e.initial, e.Et, t, sg["coefficients"], model=model)
                a.plot(t, y, ls=ls, lw=lw, color=col, label=label)
            if r == 0:
                a.set_title(LABELS.get(sg["name"], sg["name"]))
            if c == 0:
                role = "train" if e.role == "train" else "held out"
                a.set_ylabel(f"{ename} ({role})\nmM")
            if r == len(rows) - 1:
                a.set_xlabel("time (s)")
    h, lab = axes[0, 0].get_legend_handles_labels()
    keep = [(hh, ll) for hh, ll in zip(h, lab, strict=True) if ll and not ll.startswith("_")]
    fig.legend(*zip(*keep, strict=True), loc="outside lower center", ncol=len(keep), frameon=False)
    fig.suptitle(title)
    _save(fig, name)


def _per(d: dict, key: str) -> dict:
    return d["per_experiment"][key]


def hypotheses(holdout_test: Path, all_test: Path) -> dict:
    h = json.loads((holdout_test / "suggest.json").read_text(encoding="utf-8"))
    a = json.loads((all_test / "suggest.json").read_text(encoding="utf-8"))
    order = list(SHORT)
    rows_h = {"as written": h["as_written"], **{r["step"]: r for r in h["candidates"]}}
    rows_a = {"as written": a["as_written"], **{r["step"]: r for r in a["candidates"]}}
    y = np.arange(len(order))[::-1]
    fig, axes = new_figure(DOUBLE, DOUBLE * 0.5, ncols=2)
    a0, a1 = axes[0]
    a0.barh(y + 0.18, [rows_h[k]["holdout"] for k in order], 0.36, color=color(1), label="held out (m4, m7)")
    a0.barh(y - 0.18, [_per(h, k)["m7"] for k in order], 0.36, color=color(5), label="m7 alone")
    a0.axvline(rows_h["as written"]["holdout"], color=color(1), ls="--", lw=0.8)
    a0.axvline(rows_h["as written"]["holdout"] / 2, color="0.5", ls=":", lw=0.8, label="half of as written")
    a0.set_title("(a) fitted on 6 sets, predict m4 and m7", loc="left")
    a1.barh(y + 0.18, [rows_a[k]["train"] for k in order], 0.36, color=color(0), label="all 8 sets")
    a1.barh(y - 0.18, [_per(a, k)["m7"] for k in order], 0.36, color=color(5), label="m7 alone")
    a1.axvline(2.0, color="0.5", ls=":", lw=0.8, label="2σ (explained)")
    a1.axvline(3.0, color="0.5", ls="--", lw=0.8, label="3σ (each set)")
    a1.set_title("(b) fitted on all 8 sets", loc="left")
    for ax in (a0, a1):
        ax.set_yticks(y)
        ax.set_xlabel("RMSE / σ (all four species)")
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    a0.set_yticklabels([SHORT[k] for k in order])
    a1.set_yticklabels([f"{SHORT[k]}  (ΔAIC {rows_a[k].get('d_aic', 0):+.0f})" for k in order])
    _save(fig, "fig11_real_hypotheses")
    return {
        "holdout": {SHORT[k]: round(rows_h[k]["holdout"], 2) for k in order},
        "m7_holdout_test": {SHORT[k]: round(_per(h, k)["m7"], 2) for k in order},
        "all8": {SHORT[k]: round(rows_a[k]["train"], 2) for k in order},
        "m7_all8": {SHORT[k]: round(_per(a, k)["m7"], 2) for k in order},
    }


def peak(run: Path) -> dict:
    p = load_project(run / "inputs")
    des = run / "5_design"
    summ = json.loads((des / "design_summary.json").read_text(encoding="utf-8"))
    with open(des / "fig5a_peak.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    with open(des / "peak_table.csv", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    m3 = np.genfromtxt(p.experiments["m3"].file, delimiter=",", names=True)
    a0_mM = float(summ["analyte0_mM"])
    fig, axes = new_figure(DOUBLE, DOUBLE * 0.46, ncols=2)
    a, b = axes[0]
    lo, hi = summ["stop_window_s"]
    a.axvspan(lo, hi, color="0.88", label="stop window (95 %)")
    for i, s in enumerate(sorted({r["sample"] for r in rows})):
        rs = [r for r in rows if r["sample"] == s]
        a.plot([float(r["time_s"]) for r in rs], [100 * float(r["yield"]) for r in rs], color=color(i), lw=1.0,
               label=f"PINN sample {s.removeprefix('sample')}")  # fmt: skip
    a.plot(m3["time_s"], 100 * m3["CEX_mM"] / a0_mM, "o", mfc="none", color="k", ms=3.5, label="m3 measured (ρ = 3)")
    a.axvline(summ["robust_stop_s"], color="k", ls="--", lw=0.8, label=f"robust stop {summ['robust_stop_s']:.0f} s")
    a.set_xlim(0, 1800)
    a.set_xlabel("time (s)")
    a.set_ylabel("cephalexin yield (%)")
    a.set_title(f"(a) ρ = {summ['rho_star']:g}, enzyme {summ['Et_nM']:.0f} nM, PGME {a0_mM:g} mM", loc="left")
    a.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
    rho = [float(r["rho"]) for r in table]
    worst = [100 * float(r["yield_robust"]) for r in table]
    b.plot(rho, worst, "o-", color=color(0), label="worst sample at robust stop")
    b.set_xlabel("ρ = 7-ADCA / PGME")
    b.set_ylabel("best yield (%)")
    b2 = b.twinx()
    b2.plot(rho, [float(r["stop_robust_s"]) for r in table], "s--", color=color(1), label="robust stop time")
    b2.set_ylabel("stop time (s)")
    b.set_title("(b) best yield and stop time against ρ", loc="left")
    h1, l1 = b.get_legend_handles_labels()
    h2, l2 = b2.get_legend_handles_labels()
    b.legend(h1 + h2, l1 + l2, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=2)
    _save(fig, "fig12_real_peak")
    return {"table": table, "m3_peak": [float(m3["time_s"][np.argmax(m3["CEX_mM"])]), float(m3["CEX_mM"].max())]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="a full run of projects/aeh_cephalexin_lagerman2021")
    ap.add_argument("--holdout-test", type=Path, required=True, help="enzopt suggest run with --candidate (6 sets)")
    ap.add_argument("--all-test", type=Path, required=True, help="the same hypotheses with --all-train")
    ap.add_argument("--esss-run", type=Path, help="a full run of the network with the ESSS steps and m7 trained")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    species(a.run, ["m3", "m6", "m4", "m7"], "fig10_real_fit", "Cephalexin synthesis: HPLC data and identified models")
    notes = {"hypotheses": hypotheses(a.holdout_test, a.all_test), "peak": peak(a.run)}
    if a.esss_run:
        title = "With ESS + PGME <=> ESSS, ESSS -> ED and m7 trained"
        species(a.esss_run, ["m6", "m7", "m4"], "fig13_real_esss", title)
    print(json.dumps(notes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
