"""Extra figures for docs/GUIDE.md (concepts, pipette volumes, Holmes results) written to docs/guide/.

    uv run python scripts/guide_figures.py --example runs/example_ldh_synth/<run> --holmes runs/ldh_holmes1988/<run>

The per-stage figures (1a … 5d) are copies from the example run; this script draws only the figures that combine
several files or runs. Text in the figures is English, like every enzopt figure.
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
from collections import defaultdict
from pathlib import Path

import numpy as np

from enzopt.core.dataio import read_series
from enzopt.core.project import load_project
from enzopt.figures import DOUBLE, _mpl, color, new_figure
from enzopt.pipeline.common import curve, rmse

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "guide"
GREY = "#444444"


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220)
    _mpl().close(fig)
    print("wrote", OUT / f"{name}.png")


def _sci(x: float) -> str:
    e = int(np.floor(np.log10(abs(x))))
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{x / 10**e:.1f}×10{sup}"


def same_fit(example_run: Path) -> None:
    """One PINN sample and one classical fit: same curve on training set A, different curves on held-out D'."""
    p = load_project(ROOT / "projects" / "example_ldh_synth")
    w = p.observable_weights()
    pinn = json.loads((example_run / "3_fit" / "samples" / "sample_seed0.json").read_text(encoding="utf-8"))
    cls = json.loads((example_run / "3_fit" / "baseline" / "fit_start0.json").read_text(encoding="utf-8"))
    models = [
        (f"PINN sample (k₊₁ = {_sci(pinn['kf'][0])} /mM/s)", pinn, GREY, "-"),
        (f"classical fit (k₊₁ = {_sci(cls['kf'][0])} /mM/s)", cls, color(0), (0, (5, 1.5, 1, 1.5))),
    ]
    fig, ax = new_figure(DOUBLE, 2.35, ncols=2)
    panels = [("A", "(a) training set A: both fit"), ("Dp", "(b) held-out set D′: they part")]
    for j, (name, title) in enumerate(panels):
        e = p.experiments[name]
        s = read_series(e.file)
        a = ax[0, j]
        a.plot(s.t, s.y, "o", mfc="none", ms=2.6, mew=0.6, color=color(1), label="data")
        t = np.linspace(0.0, float(s.t.max()), 400)
        for label, m, c, ls in models:
            k = p.network.rate_vector(m["kf"], m["kr"])
            y = curve(p.network, k, e.initial, e.Et, t, w)
            r = rmse(curve(p.network, k, e.initial, e.Et, s.t, w), s.y)
            a.plot(t, y, color=c, ls=ls, lw=1.3, label=f"{label}, RMSE {r:.4f}")
        a.set_title(title, loc="left")
        a.set_xlabel("time (s)")
        a.set_ylabel("A340")
        a.legend(frameon=False, loc="upper right")
    _save(fig, "concept_same_fit")


def equilibrium_limit(Kp_values=(23.0, 2.3e4), c: float = 0.99, rho_star: float = 16.0) -> None:
    """Unconverted NADH fraction at equilibrium vs substrate ratio: set by K' alone."""
    rho = np.logspace(np.log10(0.3), 2, 600)
    fig, ax = new_figure(DOUBLE * 0.62, 2.45)
    a = ax[0, 0]
    for i, (Kp, lab) in enumerate(zip(Kp_values, ["K′ = 23 (pH 10)", "K′ = 2.3×10⁴ (pH 7)"], strict=True)):
        A = Kp - 1.0
        B = Kp * (1.0 + rho)
        f = (B - np.sqrt(B * B - 4.0 * A * Kp * rho)) / (2.0 * A)
        a.plot(rho, 1.0 - f, color=[color(0), color(1)][i], ls=["-", "--"][i], lw=1.3, label=lab)
    Kp = Kp_values[0]
    rho_min = c + c * c / ((1.0 - c) * Kp)
    a.axhline(1.0 - c, color=color(3), lw=0.8, ls=":")
    a.axvline(rho_min, color=color(3), lw=0.8, ls=":")
    a.annotate(f"ρ_min = {rho_min:.2f}", (rho_min, 0.5), xytext=(4, 0), textcoords="offset points", color=color(3))
    a.annotate("target: 99 % converted", (0.32, 1.0 - c), xytext=(0, 3), textcoords="offset points", color=color(3))
    A, B = Kp - 1.0, Kp * (1.0 + rho_star)
    f_star = (B - np.sqrt(B * B - 4.0 * A * Kp * rho_star)) / (2.0 * A)
    a.plot([rho_star], [1.0 - f_star], "*", ms=7, color=color(0))
    a.annotate(f"robust design ρ* = {rho_star:g}", (rho_star, 1.0 - f_star), xytext=(6, 5), textcoords="offset points")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(1e-6, 1.2)
    a.set_xlabel("substrate ratio ρ = Pyr₀ / NADH₀")
    a.set_ylabel("NADH left at equilibrium (fraction)")
    a.set_title("equilibrium limit: no rate constants, no enzyme amount", loc="left")
    a.legend(frameon=False, loc="lower left")
    _save(fig, "concept_equilibrium_limit")


def pipette_volumes(example_run: Path) -> None:
    """Stacked 1000 µL composition of every pipette-table row."""
    with open(example_run / "5_design" / "pipette.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    names = {"L1": "limit test", "L2": "limit test", "D1": "robust", "D0": "nominal", "Z": "no enzyme"}
    labels = [
        f"{r['set']}  {names.get(r['set'], '')}, ρ = {float(r['rho']):g}"
        + (f", ×{float(r['Et_over_E0']):.3g} E₀" if float(r["Et_over_E0"]) > 0 else "")
        for r in rows
    ]
    parts = [
        ("NADH stock", "NADH_stock_uL", color(0)),
        ("pyruvate stock", "Pyr_stock_uL", color(1)),
        ("buffer", "buffer_uL", "#d9d9d9"),
        ("enzyme (added last)", "enzyme_uL", color(2)),
    ]
    fig, ax = new_figure(DOUBLE, 0.55 + 0.28 * len(rows))
    a = ax[0, 0]
    y = np.arange(len(rows))[::-1]
    left = np.zeros(len(rows))
    for lab, key, c in parts:
        v = np.array([float(r[key]) for r in rows])
        a.barh(y, v, left=left, color=c, height=0.62, label=lab, edgecolor="white", linewidth=0.4)
        for yi, li, vi in zip(y, left, v, strict=True):
            if vi >= 40:
                a.text(li + vi / 2, yi, f"{vi:g}", ha="center", va="center", fontsize=6, color="black")
        left += v
    for yi, r in zip(y, rows, strict=True):
        a.text(1012, yi, f"Et = {float(r['Et_final_nM']):.3g} nM", va="center", fontsize=6.3)
    a.set_yticks(y, labels)
    a.set_xlim(0, 1150)
    a.set_xticks([0, 250, 500, 750, 1000])
    a.set_xlabel("volume in the 1000 µL reaction (µL)")
    a.spines[["top", "right"]].set_visible(False)
    a.legend(frameon=False, ncols=4, loc="upper center", bbox_to_anchor=(0.45, -0.2 - 0.3 / len(rows)))
    _save(fig, "pipette_volumes")


def _half_errors(holmes_run: Path, p) -> dict:
    """Error of the predicted product at the time the data reach half of the limiting substrate (%), per model."""
    with open(holmes_run / "3_fit" / "fig3b_holdout.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    out = {}
    for e in ["s172", "s408", "s422", "s465", "s251", "s457"]:
        A0, B0 = p.experiments[e].initial["NADH"], p.experiments[e].initial["Pyr"]
        half = 0.5 * min(A0, B0)
        d = [r for r in rows if r["experiment"] == e]

        def series(m, d=d, A0=A0):
            return np.array(sorted((float(r["time_s"]), A0 - float(r["NADH_mM"])) for r in d if r["model"] == m)).T

        to, po = series("data")
        th = np.interp(half, po, to)
        res = defaultdict(list)
        for m in sorted({r["model"] for r in d} - {"data"}):
            tm, pm = series(m)
            res[m.split()[0]].append((half - np.interp(th, tm, pm)) / half * 100.0)
        out[e] = res
    return out


def holmes_half_error(holmes_run: Path) -> None:
    p = load_project(ROOT / "projects" / "ldh_holmes1988")
    errs = _half_errors(holmes_run, p)
    with open(ROOT / "projects" / "ldh_holmes1988" / "raw" / "holmes1988_tables.csv", encoding="utf-8") as fh:
        table = {f"s{r['set']}": float(r["err_pct_obs_minus_phat"]) for r in csv.DictReader(fh)}
    sets = list(errs)
    x = np.arange(len(sets))
    fig, ax = new_figure(DOUBLE * 0.75, 2.3)
    a = ax[0, 0]
    wd = 0.26
    a.bar(x - wd, [table[s] for s in sets], wd, color="#bbbbbb", label="Holmes 1988 (global constants, 172 in fit)")
    wr = [errs[s]["WR"] for s in sets]
    a.bar(x, [np.mean(v) for v in wr], wd, color=GREY, label="enzopt PINN (mean of 4 samples, range)")
    a.errorbar(
        x,
        [np.mean(v) for v in wr],
        yerr=[[np.mean(v) - min(v) for v in wr], [max(v) - np.mean(v) for v in wr]],
        fmt="none",
        ecolor="black",
        elinewidth=0.7,
        capsize=1.5,
    )
    a.bar(x + wd, [np.mean(errs[s]["classical"]) for s in sets], wd, color=color(0), label="classical fit (mean)")
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(x, [s[1:] for s in sets])
    a.set_xlabel("held-out set (Holmes numbering)")
    a.set_ylabel("error at 50 % reaction (%)")
    a.set_title("prediction error on sets not used for training", loc="left")
    a.legend(frameon=False, fontsize=5.8, loc="upper right")
    _save(fig, "holmes_half_error")


def pinn_vs_classical(example_run: Path, holmes_run: Path) -> None:
    """Held-out RMSE in units of σ, PINN samples vs classical starts, on both data sets."""
    panels = [
        (example_run, 0.003, "(a) synthetic example (σ = 0.003 A340)", ["C", "Dp", "D1", "D0"]),
        (
            holmes_run,
            0.001,
            "(b) Holmes 1988 (σ = 0.001 mM)",
            ["s172", "s408", "s422", "s465", "s251", "s457", "s172fig", "s408fig"],
        ),
    ]
    fig, ax = new_figure(DOUBLE, 2.4, ncols=2, gridspec_kw={"width_ratios": [1, 1.8]})
    for j, (run, sigma, title, sets) in enumerate(panels):
        with open(run / "3_fit" / "holdout_comparison.csv", encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        a = ax[0, j]
        for k, (model, lab, c, mk) in enumerate(
            [("PINN (WR)", "PINN samples", GREY, "o"), ("classical", "classical fit starts", color(0), "+")]
        ):
            for i, e in enumerate(sets):
                v = [float(r["rmse_A340"]) / sigma for r in rows if r["model"] == model and r["experiment"] == e]
                xs = i + (k - 0.5) * 0.3 + np.linspace(-0.06, 0.06, len(v))
                ms = 3.2 if mk == "o" else 5
                a.plot(xs, v, mk, color=c, ms=ms, mfc="none", mew=0.8, label=lab if i == 0 else None)
        a.axhline(2.0, color=color(3), lw=0.7, ls=":", label="2σ")
        a.set_yscale("log")
        names = {"Dp": "D′", "D1": "D1\n(check)", "D0": "D0\n(check)"}
        a.set_xticks(range(len(sets)), [names.get(s, s.lstrip("s")) for s in sets])
        a.set_xlim(-0.6, len(sets) - 0.4)
        a.set_ylabel("held-out RMSE / σ")
        a.set_title(title, loc="left")
    h, lab = ax[0, 0].get_legend_handles_labels()
    fig.legend(h, lab, frameon=False, ncols=3, loc="outside lower center")
    _save(fig, "pinn_vs_classical")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--example", type=Path, required=True, help="full-mode run folder of projects/example_ldh_synth")
    ap.add_argument("--holmes", type=Path, required=True, help="run folder of projects/ldh_holmes1988")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    same_fit(a.example)
    equilibrium_limit()
    pipette_volumes(a.example)
    holmes_half_error(a.holmes)
    pinn_vs_classical(a.example, a.holmes)
    shutil.copyfile(a.holmes / "3_fit" / "fig3b_holdout.png", OUT / "holmes_fig3b_holdout.png")
    print("wrote", OUT / "holmes_fig3b_holdout.png")


if __name__ == "__main__":
    main()
