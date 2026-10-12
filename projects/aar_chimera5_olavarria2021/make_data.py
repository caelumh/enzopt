"""Build this project's data/ from the workbooks of Olavarria (2021) (Mendeley Data doi:10.17632/tspycdy2rm.1,
CC BY 4.0; see README.md).

    uv run --with openpyxl python -I projects/aar_chimera5_olavarria2021/make_data.py

The two workbooks of the engineered acetoacetyl-CoA reductase "Chimera 5" with NADH (raw/) hold, per plate well, the
product formed (µM) every 8 s for 1 h at 30 °C, computed by the authors from the absorbance at 360 nm
(ε(NADH) 4075 + ε(AcAcCoA) 23 M⁻¹cm⁻¹, 0.5 cm). Row 1 is the enzyme (µM), rows 2–3 the nominal AcAcCoA and NADH
(µM), row 10 the authors' estimate of the product formed before the first reading (not used, see below).

- varying NADH: wells A1…B12 (24), AcAcCoA 1600 µM, NADH 12–600 µM, enzyme 1–4 nM
- varying AcAcCoA: wells C1…C12 (12, first block of the sheet), NADH about 1000 µM, AcAcCoA 24–200 µM, enzyme 1 nM

Each well becomes data/<series>_<well>.csv (time_s, P_mM). The first reading is the initial condition: product =
first data point, each substrate = nominal − that product (printed as project.yaml lines). Values are unchanged except
µM → mM; only the authors' fitting windows are kept (DYNAFIT scripts: "maximum 600" for varying NADH, 1200 for
varying AcAcCoA, in s), every 2nd reading (16 s). Later the reactions slow down well before the substrate runs out (an
open question, README). The noise is estimated per well from second differences of all readings (σ ≈ sd(Δ²y)/√6).
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np
import openpyxl

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
THIN = 2  # every 2nd reading (16 s)
SERIES = {
    "N": ("chimera5_varying_NADH.xlsx", 24, "NADH", 600.0),  # prefix, workbook, wells, varied substrate, window (s)
    "A": ("chimera5_varying_AcAcCoA.xlsx", 12, "AcAc", 1200.0),
}


def rows_of(path: Path) -> list[tuple]:
    wb = openpyxl.load_workbook(path, data_only=True, read_only=True)
    return list(wb["data and calculations"].iter_rows(values_only=True))


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    lines, sig = [], {}
    for prefix, (book, n, varied, window) in SERIES.items():
        rows = rows_of(RAW / book)
        wells = rows[19][2 : 2 + n]
        data = [r for r in rows[20:] if isinstance(r[1], (int, float)) and r[2] is not None]
        t = np.array([float(r[1]) for r in data])
        for j, well in enumerate(wells):
            c = 2 + j
            et_uM, acac_nom, nadh_nom = (float(rows[i][c]) for i in (0, 1, 2))
            y = np.array([float(r[c]) for r in data])
            # product already formed at the first reading = the first data point (the workbook's row 10 differs from it
            # in some wells, e.g. B11 1.3 vs 4.7 µM, which would put the curve off its own first point)
            p0 = float(y[0])
            s0 = (nadh_nom if varied == "NADH" else acac_nom) - p0
            name = f"{prefix}_{well}"
            with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(["time_s", "P_mM"])
                keep = t <= window
                for tt, yy in zip(t[keep][::THIN], y[keep][::THIN], strict=True):
                    w.writerow([f"{tt:g}", f"{yy / 1000:.7g}"])
            d2 = np.diff(y, 2)
            sig[name] = float(np.std(d2) / np.sqrt(6.0))
            if varied == "NADH":
                nadh, acac = s0, acac_nom - p0
            else:
                acac, nadh = s0, nadh_nom - p0
            lines.append(
                f"  {name}: {{file: data/{name}.csv, AcAc: {acac / 1000:.6g}, NADH: {nadh / 1000:.6g}, "
                f"NAD: {p0 / 1000:.4g}, HB: {p0 / 1000:.4g}, Et_nM: {et_uM * 1000:g}, role: ROLE}}"
                f"   # {varied} {nadh_nom if varied == 'NADH' else acac_nom:g} µM nominal"
            )
        print(f"{prefix} ({varied} varied): {n} wells, {len(t)} readings to {t[-1]:g} s, kept 0–{window:g} s / {THIN}")
    s = np.array(list(sig.values()))
    print(
        f"noise (second differences, µM): median {np.median(s):.3f}, 90% {np.quantile(s, 0.9):.3f}, max {s.max():.3f}"
    )
    for k in ("N", "A"):
        v = [x for n_, x in sig.items() if n_.startswith(k)]
        print(f"  series {k}: median {np.median(v):.3f} µM")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
