"""Build the five temperature projects t25 … t45 from the SLAC/ABTS plate-reader exports of Häußler (2023)
(DaRUS doi:10.18419/DARUS-3337, CC BY 4.0; see README.md).

    uv run python -I projects/slac_abts_haussler2023/make_data.py

raw/timecourse/SLAC_pH3.0_<T>C.txt: SpectraMax kinetic export (UTF-16), 11 reads over 15 min, each read with the plate
chamber temperature and two 6 × 12 blocks at 340 nm (ABTS) and 420 nm (ABTS radical). Columns 1–10 hold ABTS 0, 5,
10, 15, 25, 50, 75, 100, 150, 200 µM; rows A–C small laccase SLAC 0.0739 µM (3 replicates), rows D–F no enzyme
(controls). raw/standards/pH3.0+3.5_<T>C_standards.txt: ABTS 0–200 µM at the same temperature, rows A–C = pH 3.0.

As in the authors' analysis (parser_functions, SLAC_kinetic_characterization.ipynb) ABTS is read at 340 nm through
the standard curve of the same temperature, and absorbances ≥ 3.2 are left out. The standard curve here is a
straight line through the standards up to 100 µM (Beer–Lambert, see standard_curve). Each enzyme well becomes
t<T>/data/S<µM>_<row>.csv (time_s, ABTS_mM, T_C), the 200 µM column is left out (its absorbance starts above 3.2).
The initial ABTS of a well is its own first reading (printed as project.yaml lines); the controls give the noise
and the drift without enzyme.
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw"
TEMPS = [25, 30, 35, 40, 45]
CONC = [0, 5, 10, 15, 25, 50, 75, 100, 150, 200]  # µM, columns 1–10 (authors' measurement_parser)
STD_CONC = [0, 5, 10, 15, 25, 50, 75, 100, 125, 150, 175, 200]  # µM, columns 1–12 (authors' notebook)
CUTOFF = 3.2  # absorbance (authors: cutoff_absorption=3.2)
LINEAR_MAX = 100  # µM: standards used for the straight line
ROWS = "ABCDEF"
USE = [1, 2, 3, 4, 5, 6, 7, 8]  # column indices 5 … 150 µM


def _lines(path: Path) -> list[list[str]]:
    text = path.read_bytes().decode("utf-16")
    return [ln.split("\t") for ln in text.replace("\r", "").split("\n")]


def _num(x: str) -> float:
    try:
        return float(x)
    except ValueError:
        return float("nan")


def read_kinetic(path: Path):
    """Times (s), chamber temperatures (°C) and absorbances (read, row, column, wavelength 0 = 340 / 1 = 420)."""
    rows = _lines(path)
    t, temp, blocks = [], [], []
    for i, r in enumerate(rows):
        if r and len(r) > 2 and r[0].count(":") == 2:
            h, m, s = (int(v) for v in r[0].split(":"))
            t.append(h * 3600 + m * 60 + s)
            temp.append(_num(r[1]))
            plate = []
            for k in range(6):
                rr = rows[i + k]
                v340 = [_num(x) for x in rr[2:14]]
                v420 = [_num(x) for x in rr[15:27]]
                plate.append([v340, v420])
            blocks.append(plate)
    A = np.array(blocks)  # (reads, 6 rows, 2 wavelengths, 12 columns)
    return np.array(t, float), np.array(temp, float), A.transpose(0, 1, 3, 2)


def read_standards(path: Path) -> np.ndarray:
    """ABTS standards at 340 nm, rows A–C (pH 3.0): (3, 12)."""
    rows = _lines(path)
    for i, r in enumerate(rows):
        if len(r) > 2 and r[0] == "" and _num(r[1]) == _num(r[1]) and len([x for x in r[2:14] if x.strip()]) == 12:
            return np.array([[_num(x) for x in rows[i + k][2:14]] for k in range(3)])
    raise ValueError(f"{path}: standards block not found")


def standard_curve(std: np.ndarray):
    """Straight line A = a + b·c (Beer–Lambert) through the replicate means up to LINEAR_MAX µM, used up to the
    cutoff. Returns (coefficients, inverse, max residual, number of points). Above 100 µM the 125 µM standard sits
    0.085 A above the line and 150 µM 0.07 A below it at every temperature (a pipetting outlier, not curvature), so
    a quadratic or cubic through all points fits the range that matters worse than the line."""
    c = np.array(STD_CONC, float)
    a = std.mean(axis=0)
    ok = c <= LINEAR_MAX
    coef = np.polyfit(c[ok], a[ok], 1)
    res = float(np.max(np.abs(np.polyval(coef, c[ok]) - a[ok])))

    def inverse(x):
        x = np.asarray(x, float)
        return np.where(x < CUTOFF, (x - coef[1]) / coef[0], np.nan)

    return coef, inverse, res, int(ok.sum())


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    for T in TEMPS:
        t, temp, A = read_kinetic(RAW / "timecourse" / f"SLAC_pH3.0_{T}C.txt")
        std = read_standards(RAW / "standards" / f"pH3.0+3.5_{T}C_standards.txt")
        coef, inv, res, n_std = standard_curve(std)
        out = HERE / f"t{T}" / "data"
        out.mkdir(parents=True, exist_ok=True)
        lines, ctrl_sd, ctrl_drift = [], [], []
        for j in USE:
            for r in range(3):
                s = inv(A[:, r, j, 0]) / 1000.0  # mM
                keep = np.isfinite(s)
                name = f"S{CONC[j]}_{ROWS[r]}"
                with open(out / f"{name}.csv", "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh, lineterminator="\n")
                    w.writerow(["time_s", "ABTS_mM", "T_C"])
                    for tt, ss, tc in zip(t[keep], s[keep], temp[keep], strict=True):
                        w.writerow([f"{tt:g}", f"{ss:.6g}", f"{tc:g}"])
                lines.append(
                    f"  {name}: {{file: data/{name}.csv, ABTS: {s[keep][0]:.5g}, P: 0, Et_nM: 73.94, role: ROLE}}"
                    f"   # {CONC[j]} µM nominal, {int(keep.sum())} points"
                )
            for r in range(3, 6):  # controls: same column without enzyme
                c_ = inv(A[:, r, j, 0])
                c_ = c_[np.isfinite(c_)]
                if len(c_) > 3:
                    i = np.arange(len(c_))
                    ctrl_sd.append(float(np.std(c_ - np.polyval(np.polyfit(i, c_, 1), i))))
                    ctrl_drift.append(float(c_[-1] - c_[0]))
        print(
            f"{T} °C: chamber {temp.min():g}–{temp.max():g} °C, {len(t)} reads 0–{t[-1]:g} s; standard curve on "
            f"{n_std} points, max residual {res:.3f} A; controls: noise {np.median(ctrl_sd):.2f} µM (median sd about "
            f"a line), "
            f"drift {np.median(ctrl_drift):+.2f} µM over 15 min (median)"
        )
        (HERE / f"t{T}" / "experiments.txt").write_text("\n".join(lines) + "\n", encoding="utf-8", newline="\n")


if __name__ == "__main__":
    main()
