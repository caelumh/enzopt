"""Read HopsanCLI ``--resultsFullCSV`` output (row format), ported from CADKIMM ``hopsan/read_results.py``.

Format: each row = ``name,(empty unit),v1,v2,...``; the time row is ``Time,,s,v1,v2,...``.
Returns ``(t, {species: array})`` taken from the integrator outputs ``X_<species>#out#Value``.
"""

from __future__ import annotations

import numpy as np


def read_all(path) -> dict:
    with open(path, "rb") as fh:
        txt = fh.read().decode("utf-8", "ignore").replace("\r\n", "\n").replace("\r", "\n")
    data = {}
    for line in txt.split("\n"):
        if not line.strip():
            continue
        p = line.split(",")
        name = p[0]
        # skip leading non-numeric cells (empty unit, "s"); once numbers start, keep only numbers
        vals = []
        for v in p[1:]:
            v = v.strip()
            if not v:
                continue
            try:
                vals.append(float(v))
            except ValueError:
                if vals:
                    break
        if vals:
            data[name] = np.array(vals)
    return data


def read_species(path):
    d = read_all(path)
    t = d.get("Time")
    s = {k.split("#")[0][2:]: v for k, v in d.items() if k.startswith("X_") and k.endswith("#out#Value")}
    if t is None:
        raise ValueError(f"시간 행(Time)을 찾지 못함: {path}")
    return t, s
