"""Build this project's data/ from the EnzymeML archive of Lagerman et al. (2021) (see README.md).

    uv run python projects/aeh_cephalexin_lagerman2021/make_data.py

``raw/EnzymeML_Lagerman.omex`` (DaRUS doi:10.18419/DARUS-2468, CC BY 4.0) holds eight HPLC time courses of
α-amino ester hydrolase (AEH) synthesising cephalexin, each with four measured species in mM at 0–30 min:
PGME (phenylglycine methyl ester), 7-ADCA, cephalexin (CEX) and phenylglycine (PG). Times are converted to seconds;
concentrations stay in mM. The t = 0 row is kept as a data point and also printed, since project.yaml uses it as the
initial condition (run 7 lists 7-ADCA 50 mM in its metadata but 40 mM in its t = 0 row; the row is used).
"""

from __future__ import annotations

import csv
import io
import zipfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
OMEX = HERE / "raw" / "EnzymeML_Lagerman.omex"
COLUMNS = ["time_s", "PGME_mM", "ADCA_mM", "CEX_mM", "PG_mM"]  # omex column order: time (min), s0, s1, s2, s3


def main() -> None:
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    with zipfile.ZipFile(OMEX) as z:
        names = sorted(n for n in z.namelist() if n.startswith("data/m") and n.endswith(".csv"))
        for name in names:
            rows = list(csv.reader(io.TextIOWrapper(z.open(name), encoding="utf-8")))
            run = Path(name).stem  # m0 … m7 = "Cephalexin synthesis 1 … 8"
            with open(out / f"{run}.csv", "w", newline="", encoding="utf-8") as fh:
                w = csv.writer(fh, lineterminator="\n")
                w.writerow(COLUMNS)
                for r in rows:
                    t_min, *conc = (float(x) for x in r)
                    w.writerow([f"{t_min * 60:g}", *(f"{c:g}" for c in conc)])
            t0 = [float(x) for x in rows[0]]
            print(f"{run}: PGME {t0[1]:g}, ADCA {t0[2]:g}, CEX {t0[3]:g}, PG {t0[4]:g} mM, {len(rows)} points")


if __name__ == "__main__":
    main()
