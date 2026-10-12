"""Build this project's data/ from the EnzymeML document of Giess et al. (2026), Scenario 2 (GitHub
EnzymeML/Giess_2026, BSD-3-Clause; see README.md).

    uv run python -I projects/pgm_eno_giess2026/make_data.py

raw/pgm-eno_output.json (EnzymeML v2) holds 15 ³¹P NMR time courses of yeast lysate converting 3-phosphoglycerate (3PG)
↔ 2-phosphoglycerate (2PG) ↔ phosphoenolpyruvate (PEP) by phosphoglycerate mutase (PGM) and enolase (ENO), 25 °C, pH 7:
concentrations in mM against the internal standard triethyl phosphate (TEP, about 5 mM), time in min, lysate protein in
mg/mL as the only enzyme amount. Seven runs start from 3PG, seven from PEP and one from both.

Each run becomes data/<id>.csv (time_s, P3_mM, P2_mM, PEP_mM); minutes → seconds, values unchanged (the 1e-10
placeholders of the initial values become 0). The noise is the scatter of the internal standard about its mean in each
run (it should stay constant), printed with the project.yaml lines.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DOC = HERE / "raw" / "pgm-eno_output.json"
SPECIES = {"chem01": "P3", "chem02": "P2", "chem03": "PEP"}
ENZYME_PER_MG_NM = 1000.0  # assumed: 1 µM of each enzyme per mg/mL lysate protein (only relative amounts matter)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    doc = json.loads(DOC.read_text(encoding="utf-8"))
    out = HERE / "data"
    out.mkdir(exist_ok=True)
    lines, tep_sd = [], []
    for m in doc["measurements"]:
        sd = {s["species_id"]: s for s in m["species_data"]}
        t = np.array(sd["chem01"]["time"], float) * 60.0
        cols = {v: np.array(sd[k]["data"], float) for k, v in SPECIES.items()}
        tep = np.array(sd["chem04"]["data"], float)
        tep_sd.append(float(np.std(tep)))
        with open(out / f"{m['id']}.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh, lineterminator="\n")
            w.writerow(["time_s", "P3_mM", "P2_mM", "PEP_mM"])
            for i in range(len(t)):
                w.writerow([f"{t[i]:g}"] + [f"{cols[v][i]:.6g}" for v in SPECIES.values()])
        init = {v: float(sd[k]["initial"]) for k, v in SPECIES.items()}
        init = {k: (0.0 if v < 1e-6 else v) for k, v in init.items()}
        prot = float(sd["prot01"]["initial"])
        et = prot * ENZYME_PER_MG_NM
        start = "+".join(k for k, v in init.items() if v > 0)
        lines.append(
            f"  {m['id']}: {{file: data/{m['id']}.csv, P3: {init['P3']:.5g}, P2: {init['P2']:.5g}, "
            f"PEP: {init['PEP']:.5g}, Et_nM: {et:g}, F: {et * 1e-6:.6g}, role: ROLE}}"
            f"   # from {start}, lysate {prot:g} mg/mL, {len(t)} points to {t[-1]:.0f} s"
        )
    print(f"internal standard scatter (mM): median {np.median(tep_sd):.3f}, range {min(tep_sd):.3f}–{max(tep_sd):.3f}")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
