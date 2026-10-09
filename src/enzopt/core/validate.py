"""CRNT-based automatic validation of simulation output, ported from CADKIMM ``analysis/crnt_validate.py``.

Closed-system checks (mass action, reactions = rxns):
  1. conservation : every conservation law w keeps w·x(t) at its initial value (relative drift < 1e-6)
  2. positivity   : every species > 0 from the second sample on (149 for weakly reversible networks)
  3. equilibrium  : at the final state, Q/K = 1 for each reversible pair (|ln(Q/K)| < 0.01)
  4. Lyapunov     : G(t) = Σ x(ln(x/c*) − 1) + c* does not increase (c* = final state)
Across runs:
  5. class consistency: runs with the same conserved quantities end in the same state

The tolerances are fixed by the spec (E7) and must not be changed.
"""

from __future__ import annotations

import numpy as np

from .crnt import build, conservation_laws, stoichiometric_matrix

TOL_CONSERVATION = 1e-6
TOL_EQUILIBRIUM_LOG = 0.01
TOL_LYAPUNOV = 1e-9


def _species(rxns):
    return build(rxns)[1]


def validate_closed(
    t, S, rxns, rates=None, tol_cons=TOL_CONSERVATION, tol_G=TOL_LYAPUNOV, tol_eq=TOL_EQUILIBRIUM_LOG
) -> dict:
    """t: time array, S: {species: array}, rxns: [(a, b)], rates: [k] aligned with rxns (for the equilibrium check)."""
    species = _species(rxns)
    X = np.column_stack([np.asarray(S[s], float) for s in species])
    N = stoichiometric_matrix(rxns, species)
    W = conservation_laws(N)
    out = {}
    scale = max(np.abs(X).max(), 1e-12)
    drift = [float(np.max(np.abs(X @ w - X[0] @ w)) / (np.abs(w).sum() * scale)) for w in W]
    out["conservation_max_rel_drift"] = max(drift) if drift else 0.0
    out["conservation_ok"] = bool(out["conservation_max_rel_drift"] < tol_cons)
    # 149 assumes a positive initial state; a boundary start becomes positive at once, so check from the 2nd sample.
    out["min_concentration"] = float(X[1:].min())
    out["positivity_ok"] = bool(out["min_concentration"] > 0.0)
    xf = X[-1]
    out["equilibrium_max_log_error"] = None
    out["equilibrium_ok"] = None
    if rates is not None:
        idx = {s: i for i, s in enumerate(species)}
        errs = []
        for j in range(0, len(rxns) - 1):
            a, b = rxns[j]
            a2, b2 = rxns[j + 1]
            if a == b2 and b == a2:  # reversible pair (j: a→b, j+1: b→a)
                with np.errstate(divide="ignore", invalid="ignore"):
                    Q = np.prod([xf[idx[s]] ** c for s, c in b.items()]) / np.prod(
                        [xf[idx[s]] ** c for s, c in a.items()]
                    )
                    K = rates[j] / rates[j + 1]
                    errs.append(float(abs(np.log(Q / K))))
        if errs:
            e = max(errs)
            out["equilibrium_max_log_error"] = e if np.isfinite(e) else float("inf")
            out["equilibrium_ok"] = bool(np.isfinite(e) and e < tol_eq)
    out["lyapunov_max_increase"] = None
    out["lyapunov_ok"] = None
    if out["positivity_ok"] and np.all(xf > 0):
        cstar = xf
        with np.errstate(divide="ignore", invalid="ignore"):
            xlogx = np.where(X > 0, X * np.log(np.where(X > 0, X, 1.0) / cstar), 0.0)
        G = np.sum(xlogx - X + cstar, axis=1)
        dG = np.diff(G)
        out["lyapunov_max_increase"] = float(max(dG.max(), 0.0)) if len(dG) else 0.0
        out["lyapunov_ok"] = bool(len(dG) == 0 or dG.max() <= tol_G * max(G[0], 1e-12) + 1e-14)
    out["ok"] = passed(out)
    return out


def passed(r: dict) -> bool:
    """Overall pass: conservation and positivity, plus equilibrium/Lyapunov when they were evaluated."""
    ok = r["conservation_ok"] and r["positivity_ok"]
    if r.get("equilibrium_ok") is not None:
        ok = ok and r["equilibrium_ok"]
    if r.get("lyapunov_ok") is not None:
        ok = ok and r["lyapunov_ok"]
    elif r["positivity_ok"]:
        ok = False  # positivity held but the final state had a zero entry: Lyapunov not defined
    return bool(ok)


def conserved_values(X: np.ndarray, rxns, species_order: list[str]) -> np.ndarray:
    """Conserved quantities (orthonormal basis) of the initial state of X (columns in ``species_order``)."""
    sp = _species(rxns)
    pos = [species_order.index(s) for s in sp]
    W = conservation_laws(stoichiometric_matrix(rxns, sp))
    return X[0, pos] @ W.T


def class_consistency(finals, conserved, decimals=4) -> dict:
    """finals: [final state vectors], conserved: [conserved quantity vectors] — spread of the final state per class.

    Conserved quantities are grouped at ``decimals`` places because simulator files carry limited digits
    (Hopsan CSV ≈ 7 significant digits).
    """
    groups = {}
    for f, c in zip(finals, conserved, strict=True):
        groups.setdefault(tuple(np.round(c, decimals)), []).append(f)
    return {k: float(np.max(np.ptp(np.array(v), axis=0))) for k, v in groups.items()}


def class_groups(conserved, decimals=4) -> list[int]:
    """Group index of each run (same conserved quantities → same group), in first-seen order."""
    keys, out = {}, []
    for c in conserved:
        k = tuple(np.round(c, decimals))
        keys.setdefault(k, len(keys))
        out.append(keys[k])
    return out
