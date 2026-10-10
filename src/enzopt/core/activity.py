"""Enzyme activity a(t) during an experiment (inactivation), t = time since the enzyme was added.

The active enzyme is Et·a(t), a(0) = 1. In the mass-action ODE every enzyme form is lost at the same first-order rate
λ(t) = −d ln a/dt, so the enzyme total follows Et·a(t) exactly and the fractions of the forms are untouched.

Descriptions (plain dicts, stored in sample files):
    {"kind": "steps", "T", "c", "s", "w"}  a = exp(−h(t/T)), h = Σ w_k [σ((u − c_k)/s) − σ(−c_k/s)], w_k ≥ 0
                                            (learned by the PINN option ``inactivation: shared``; shape not chosen)
    {"kind": "hill", "t50", "n"}            a = 1/(1 + (t/t50)^n)
    {"kind": "exp", "ki"}                   a = exp(−ki·t)
A dict without "kind" but with "w" is "steps" (as written by the PINN). ``None`` means no inactivation.

numpy only: this module is part of the KIMM-CAD compatible core.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np


def _kind(act: dict) -> str:
    return act.get("kind") or ("steps" if "w" in act else "")


def _sig(z):
    return 0.5 * (1.0 + np.tanh(0.5 * z))  # logistic without overflow warnings


def activity_curve(act: dict | None, t) -> np.ndarray:
    """a(t); 1 everywhere without inactivation."""
    t = np.asarray(t, float)
    if not act:
        return np.ones_like(t)
    k = _kind(act)
    if k == "steps":
        u = t[..., None] / float(act["T"])
        c, s, w = np.asarray(act["c"], float), float(act["s"]), np.asarray(act["w"], float)
        return np.exp(-(w * (_sig((u - c) / s) - _sig(-c / s))).sum(axis=-1))
    if k == "hill":
        return 1.0 / (1.0 + (np.maximum(t, 0.0) / float(act["t50"])) ** float(act["n"]))
    if k == "exp":
        return np.exp(-float(act["ki"]) * t)
    raise ValueError(f"알 수 없는 효소 활성 형식: {act.get('kind')}")


def activity_rate(act: dict | None, t) -> np.ndarray:
    """λ(t) = −d ln a/dt (1/s): the first-order loss rate of every enzyme form that gives a(t)."""
    t = np.asarray(t, float)
    if not act:
        return np.zeros_like(t)
    k = _kind(act)
    if k == "steps":
        T = float(act["T"])
        u = t[..., None] / T
        c, s, w = np.asarray(act["c"], float), float(act["s"]), np.asarray(act["w"], float)
        sg = _sig((u - c) / s)
        return (w * sg * (1.0 - sg) / s).sum(axis=-1) / T
    if k == "hill":
        t50, n = float(act["t50"]), float(act["n"])
        tt = np.maximum(t, 1e-300)
        q = (tt / t50) ** n
        return np.where(t > 0, n / tt * q / (1.0 + q), 0.0)
    if k == "exp":
        return np.full_like(t, float(act["ki"]))
    raise ValueError(f"알 수 없는 효소 활성 형식: {act.get('kind')}")


def enzyme_loss(species: list[str], enzyme_species: list[str], act: dict | None):
    """(mask, λ(t)) for ``core.massaction.simulate(loss=...)``; None without inactivation."""
    if not act:
        return None
    ens = set(enzyme_species)
    mask = np.array([1.0 if s in ens else 0.0 for s in species])

    def rate(t: float) -> float:
        return float(activity_rate(act, t))

    return mask, rate


def horizon(act: dict | None) -> float | None:
    """Time up to which a learned a(t) is supported by data (the longest training experiment); None otherwise."""
    if act and _kind(act) == "steps":
        return float(act["T"])
    return None


LossFn = tuple[np.ndarray, Callable[[float], float]]
