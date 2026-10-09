"""Build this project's data/ from Holmes (1988) Tables II–III and Figures 4–5 (see README.md).

    uv run python projects/ldh_holmes1988/make_data.py

Each of the 21 progress curves is rebuilt from the published integrated rate equation (thesis eq 6), using the mean
coefficients of the six replicate fits (Table III):

    e0·t = −Cf·ln(1 − ΔP/A0) − Cs·ln(1 − ΔP/B0) + C1·ΔP + ½·C2·ΔP²      (A = NADH, B = pyruvate; e0 units/ml, t min)

ΔP runs from 0 to 90 % of the limiting substrate (the range Holmes fitted). Time is converted with one nominal
enzyme concentration E0 = 0.05 units/ml for every set (the published curves are on the e0·t axis), and the observable
is the NADH concentration NADH = A0 − ΔP (mM), sampled every 5 s. The raw points read from Figures 4 and 5 become two
extra short series (``set408fig``, ``set172fig``).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
E0_UNITS_PER_ML = 0.05
DT_S = 5.0


def rebuild(row: dict) -> tuple[np.ndarray, np.ndarray]:
    A0, B0 = float(row["NADH_mM"]), float(row["Pyr_mM"])
    Cf, Cs, C1, C2 = (float(row[k]) for k in ("Cf", "Cs", "C1", "C2"))
    dP = np.linspace(0.0, 0.9 * min(A0, B0), 2001)
    e0t = -Cf * np.log1p(-dP / A0) - Cs * np.log1p(-dP / B0) + C1 * dP + 0.5 * C2 * dP**2
    if np.any(np.diff(e0t) <= 0):
        raise ValueError(f"set {row['set']}: e0·t is not increasing in ΔP — coefficients inconsistent")
    t = e0t / E0_UNITS_PER_ML * 60.0
    grid = np.arange(DT_S, t[-1] + 1e-9, DT_S)
    return grid, A0 - np.interp(grid, t, dP)


def write(path: Path, t, y) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write("time_s,NADH_mM\n")
        for a, b in zip(t, y, strict=True):
            fh.write(f"{a:.1f},{b:.6f}\n")


def main() -> None:
    data = HERE / "data"
    data.mkdir(exist_ok=True)
    rows = list(csv.DictReader(open(HERE / "raw" / "holmes1988_tables.csv", encoding="utf-8")))
    for r in rows:
        t, y = rebuild(r)
        write(data / f"set{r['set']}.csv", t, y)
        print(f"set {r['set']}: {len(t)} points, {t[-1]:.0f} s")
    a0 = {r["set"]: float(r["NADH_mM"]) for r in rows}
    pts = list(csv.DictReader(open(HERE / "raw" / "holmes1988_figure_points.csv", encoding="utf-8")))
    for s in sorted({p["set"] for p in pts}):
        sel = [p for p in pts if p["set"] == s]
        t = np.array([float(p["e0t_units_min_per_ml"]) for p in sel]) / E0_UNITS_PER_ML * 60.0
        y = a0[s] * (1.0 - np.array([float(p["dP_over_A0"]) for p in sel]))
        write(data / f"set{s}fig.csv", t, y)
        print(f"set {s} figure points: {len(t)}")


if __name__ == "__main__":
    main()
