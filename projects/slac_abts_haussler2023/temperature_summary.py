"""Collect the rate constants of the five temperature runs and plot their temperature dependence.

    uv run python projects/slac_abts_haussler2023/temperature_summary.py [runs root]

For each temperature the latest completed run of slac_abts_haussler2023_<T>C is read: the kept PINN samples and the
classical fits give kcat = k₂, Km = (k₋₁ + k₂)/k₁, kcat/Km and the inactivation constants k₃ (E → ED)
and k₄ (ES → ED). The authors'
values at pH 3.0 (SLAC_kinetic_characterization.ipynb, irreversible Michaelis–Menten with time-dependent inactivation of
the whole enzyme) are drawn for comparison. An Arrhenius line ln kcat = ln A − Ea/(R T) is fitted to the classical
fits. Writes figures/temperature_dependence.png and figures/temperature_table.csv.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
TEMPS = [25, 30, 35, 40, 45]
R_GAS = 8.314
# authors' pH 3.0 results (notebook cell 17): kcat 1/s, Km µM
AUTHORS = {25: (0.459962, 35.137357), 30: (0.470366, 32.114083), 35: (0.546029, 37.454496),
           40: (0.712488, 46.441229), 45: (0.815724, 47.967746)}  # fmt: skip


def latest_run(runs: Path, T: int) -> Path | None:
    d = runs / f"slac_abts_haussler2023_{T}C"
    done = []
    for r in sorted(d.glob("20*")) if d.exists() else []:
        m = r / "manifest.json"
        if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "completed":
            done.append(r)
    return done[-1] if done else None


def constants(kf, kr) -> dict:
    k1, k2, k3, km1 = kf[0], kf[1], kf[2], kr[0]
    km = (km1 + k2) / k1 * 1000.0  # µM
    k4 = kf[3] if len(kf) > 3 else float("nan")  # ES → ED (network with both inactivation steps)
    return {"kcat": k2, "Km_uM": km, "kcat_Km": k2 / km, "k_inact": k3, "k_inact_ES": k4}


def main() -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    runs = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "runs"
    rows = []
    for T in TEMPS:
        run = latest_run(runs, T)
        if run is None:
            print(f"{T} °C: no completed run")
            continue
        for f in sorted((run / "3_fit" / "samples").glob("sample_seed*.json")):
            s = json.loads(f.read_text(encoding="utf-8"))
            if s["kept"]:
                rows.append(
                    {"T_C": T, "model": f"PINN seed{s['seed']}", **constants(s["kf"], s["kr"]), "run": run.name}
                )
        for f in sorted((run / "3_fit" / "baseline").glob("fit_start*.json")):
            c = json.loads(f.read_text(encoding="utf-8"))
            rows.append({"T_C": T, "model": f"classical {f.stem[-1]}", **constants(c["kf"], c["kr"]), "run": run.name})
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    keys = ["T_C", "model", "kcat", "Km_uM", "kcat_Km", "k_inact", "k_inact_ES", "run"]
    with open(out / "temperature_table.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(keys)
        for r in rows:
            w.writerow([f"{r[k]:.6g}" if isinstance(r[k], float) else r[k] for k in keys])
    cl = [r for r in rows if r["model"].startswith("classical")]
    Ts = np.array(sorted({r["T_C"] for r in cl}), float)
    kc = np.array([np.median([r["kcat"] for r in cl if r["T_C"] == T]) for T in Ts])
    slope, icpt = np.polyfit(1.0 / (Ts + 273.15), np.log(kc), 1)
    ea = -slope * R_GAS / 1000.0
    a_t = np.array(sorted(AUTHORS))
    a_kc = np.array([AUTHORS[T][0] for T in a_t])
    a_slope = np.polyfit(1.0 / (a_t + 273.15), np.log(a_kc), 1)[0]
    print(f"Arrhenius (classical fits): Ea = {ea:.1f} kJ/mol; authors' kcat: Ea = {-a_slope * R_GAS / 1000:.1f} kJ/mol")
    for T in Ts:
        sel = [r for r in rows if r["T_C"] == T]
        print(
            f"{T:.0f} °C: kcat {min(r['kcat'] for r in sel):.3f}–{max(r['kcat'] for r in sel):.3f} /s, "
            f"Km {min(r['Km_uM'] for r in sel):.1f}–{max(r['Km_uM'] for r in sel):.1f} µM, "
            f"k_inact E {min(r['k_inact'] for r in sel):.4f}–{max(r['k_inact'] for r in sel):.4f}, "
            f"ES {min(r['k_inact_ES'] for r in sel):.4f}–{max(r['k_inact_ES'] for r in sel):.4f} /s "
            f"(authors: kcat {AUTHORS[int(T)][0]:.3f}, Km {AUTHORS[int(T)][1]:.1f})"
        )
    fig, ax = plt.subplots(1, 4, figsize=(13.5, 3.3), layout="constrained")
    panels = [("kcat", "kcat (1/s)", True), ("Km_uM", "Km (µM)", False), ("kcat_Km", "kcat/Km (1/(s·µM))", True),
              ("k_inact", "inactivation (1/s)", True)]  # fmt: skip
    for a, (key, label, logy) in zip(ax, panels, strict=True):
        for kind, mk, col in (("PINN", "o", "#0072B2"), ("classical", "s", "#E69F00")):
            sel = [r for r in rows if r["model"].startswith(kind)]
            a.plot([r["T_C"] + (-0.6 if kind == "PINN" else 0.6) for r in sel], [r[key] for r in sel], mk, mfc="none",
                   color=col, label=f"enzopt {kind}")  # fmt: skip
        if key == "k_inact":
            for kind, mk, col in (("PINN", "o", "#0072B2"), ("classical", "s", "#E69F00")):
                sel = [r for r in rows if r["model"].startswith(kind) and np.isfinite(r["k_inact_ES"])]
                a.plot([r["T_C"] + (-0.6 if kind == "PINN" else 0.6) for r in sel], [r["k_inact_ES"] for r in sel],
                       mk, color=col)  # fmt: skip
            a.plot([], [], "o", mfc="none", color="0.4", label="E → ED (open)")
            a.plot([], [], "o", color="0.4", label="ES → ED (filled)")
            a.legend(fontsize=7, frameon=False, loc="lower right")
        if key in ("kcat", "Km_uM", "kcat_Km"):
            vals = {"kcat": a_kc, "Km_uM": np.array([AUTHORS[T][1] for T in a_t]),
                    "kcat_Km": a_kc / np.array([AUTHORS[T][1] for T in a_t])}[key]  # fmt: skip
            a.plot(a_t, vals, "k^--", lw=0.8, ms=5, label="authors (whole-enzyme inactivation)")
        if key == "kcat":
            tt = np.linspace(25, 45, 50)
            a.plot(
                tt, np.exp(icpt + slope / (tt + 273.15)), color="#E69F00", lw=0.8, label=f"Arrhenius {ea:.0f} kJ/mol"
            )
        if logy:
            a.set_yscale("log")
        a.set_xlabel("temperature (°C)")
        a.set_title(label, fontsize=10)
    ax[0].legend(fontsize=7, frameon=False)
    fig.suptitle("SLAC + ABTS, pH 3.0: rate constants identified at each temperature", fontsize=11)
    fig.savefig(out / "temperature_dependence.png", dpi=150)
    print(f"wrote {out / 'temperature_dependence.png'} and temperature_table.csv ({len(rows)} rows)")


if __name__ == "__main__":
    main()
