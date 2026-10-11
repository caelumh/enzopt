"""Figures for docs/identification/README.md: how stage ③ turns A340 progress curves and a reaction scheme into
rate-constant samples, and how the samples feed the design (synthetic LDH example).

    uv run python scripts/identification_figures.py --run runs/example_ldh_synth/<run>

The synthetic truth (projects/example_ldh_synth/data/truth.json, never read by the pipeline) gives the forward problem
and the reference marks; the run gives the PINN samples, the classical fits, the restart log and the design. The
classical-fit walk of figure 4 is recomputed here (one start at the prior centre, about a minute), and figure 9 trains
the PINN once more with the design-check set D0 added (GPU about 10 s, CPU about 2 min). Text in the figures is
English, like every enzopt figure.
"""

from __future__ import annotations

import argparse
import ast
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from enzopt.core.dataio import read_series
from enzopt.core.design import AssayDesign, min_Et
from enzopt.core.massaction import MassAction, SolverSettings, atol_vector, simulate
from enzopt.core.network import x0_vector
from enzopt.core.project import load_project
from enzopt.figures import DOUBLE, _mpl, color, new_figure
from enzopt.pipeline.common import curve, design_rates, load_samples
from enzopt.pipeline.compare import _residuals

ROOT = Path(__file__).resolve().parents[1]
PROJ = ROOT / "projects" / "example_ldh_synth"
OUT = ROOT / "docs" / "identification"
GREY = "#444444"
TRUTH_STYLE = {"color": "black", "ls": (0, (4, 2)), "lw": 1.0}
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
LONG = {"E": "free enzyme", "EN": "E·NADH", "ENP": "E·NADH·Pyr", "EQL": "E·NAD⁺·Lac", "EQ": "E·NAD⁺"}
SETS = ["A", "B", "C", "Dp", "D1", "D0"]
SET_GROUPS = [(0.5, "train"), (2.5, "held out"), (4.5, "design check")]


def _save(fig, name: str) -> None:
    fig.savefig(OUT / f"{name}.png", dpi=220)
    _mpl().close(fig)
    print("wrote", OUT / f"{name}.png", flush=True)


def _k(i: int, sign: str) -> str:
    return f"k{sign}{str(i).translate(SUB)}"


def _name(n: str) -> str:
    return n.replace("Dp", "D′")


class Example:
    """The example project, its truth and one run of it."""

    def __init__(self, run: Path):
        self.p = load_project(PROJ)
        self.net = self.p.network
        self.model = MassAction.from_network(self.net)
        self.w = self.p.observable_weights()
        self.sigma = float(self.p.identification["noise_sigma"])
        self.truth = json.loads((PROJ / "data" / "truth.json").read_text(encoding="utf-8"))
        self.k_truth = self.net.rate_vector(self.truth["kf"], self.truth["kr"])
        self.run = run
        self.samples = load_samples(run)
        fits = [json.loads(f.read_text(encoding="utf-8")) for f in sorted((run / "3_fit" / "baseline").glob("*.json"))]
        self.classical = [c for c in fits if "kf" in c]
        self.fit = json.loads((run / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
        self.design = json.loads((run / "5_design" / "design_summary.json").read_text(encoding="utf-8"))

    def series(self, name: str):
        s = read_series(self.p.experiments[name].file)
        return np.asarray(s.t, float), np.asarray(s.y, float)

    def curve(self, k, name: str, t) -> np.ndarray:
        e = self.p.experiments[name]
        return curve(self.net, k, e.initial, e.Et, t, self.w, model=self.model)

    def rates(self, m: dict) -> np.ndarray:
        return self.net.rate_vector(m["kf"], m["kr"])


# ---------------------------------------------------------------- 1 reaction scheme
def mechanism(ex: Example) -> None:
    from matplotlib.patches import FancyArrowPatch

    plt = _mpl()
    fig, ax = plt.subplots(figsize=(DOUBLE * 0.62, DOUBLE * 0.52), layout="constrained")
    ax.set_aspect("equal")
    ax.axis("off")
    forms = ["E", "EN", "ENP", "EQL", "EQ"]
    ang = [90 - 72 * i for i in range(5)]  # clockwise from the top

    def at(a_deg: float, r: float):
        a = math.radians(a_deg)
        return r * math.cos(a), r * math.sin(a)

    for f, a in zip(forms, ang, strict=True):
        x, y = at(a, 1.0)
        ax.text(
            x, y, f"{f}\n{LONG[f]}", ha="center", va="center", fontsize=6.5,
            bbox={"boxstyle": "round,pad=0.35", "fc": "white", "ec": GREY, "lw": 0.7},
        )  # fmt: skip
    exchange = {0: ("NADH", True), 1: ("Pyr", True), 2: ("chemistry", None), 3: ("Lac", False), 4: ("NAD⁺", False)}
    for i in range(5):
        a0, a1 = ang[i], ang[(i + 1) % 5]
        am = a0 - 36.0
        off = 24 if i == 2 else 13  # the two wide boxes at the bottom
        arc = FancyArrowPatch(
            at(a0 - off, 1.0), at(a1 + off, 1.0), connectionstyle="arc3,rad=-0.22", arrowstyle="<|-|>",
            mutation_scale=7, lw=0.9, color=GREY,
        )  # fmt: skip
        ax.add_patch(arc)
        lx, ly = at(am, 0.70)
        ax.text(lx, ly, f"{_k(i + 1, '₊')}\n{_k(i + 1, '₋')}", ha="center", va="center", fontsize=6.5, color=GREY)
        name, into = exchange[i]
        if into is None:
            tx, ty = at(am, 1.2)
            ax.text(tx, ty, name, ha="center", va="center", fontsize=6.5, style="italic", color=GREY)
            continue
        outer, inner = at(am, 1.62), at(am, 1.2)
        ax.annotate(
            "", xy=inner if into else outer, xytext=outer if into else inner,
            arrowprops={"arrowstyle": "-|>", "lw": 0.8, "color": color(0 if into else 2), "mutation_scale": 7},
        )  # fmt: skip
        tx, ty = at(am, 1.8)
        c = color(0 if into else 2)
        ax.text(tx, ty, ("+ " if into else "− ") + name, ha="center", va="center", fontsize=7, color=c)
    ax.text(0, 0.1, "overall\nNADH + Pyr ⇌ NAD⁺ + Lac", ha="center", va="center", fontsize=6)
    ax.text(0, -0.18, "K′ = 23", ha="center", va="center", fontsize=6)
    ax.set_xlim(-2.05, 2.05)
    ax.set_ylim(-1.4, 1.7)
    _save(fig, "fig1_mechanism")


# ---------------------------------------------------------------- 2 forward problem
def forward(ex: Example) -> None:
    e = ex.p.experiments["A"]
    st = SolverSettings()
    t = np.concatenate([[0.0], np.geomspace(1e-5, 900.0, 700)])
    X = simulate(
        ex.model, ex.k_truth, x0_vector(ex.net, e.initial, e.Et), t,
        atol=atol_vector(ex.net.species, ex.net.enzyme_species, e.Et, st), settings=st,
    )  # fmt: skip
    i = ex.net.index()
    fig, ax = new_figure(DOUBLE, 2.25, ncols=3)
    a = ax[0, 0]
    a.plot(t, X[:, i["NADH"]], color=color(0), lw=1.8, label="NADH")
    a.plot(t, X[:, i["Pyr"]], color=color(1), ls="--", label="Pyr")
    a.plot(t, X[:, i["NAD"]], color=color(2), lw=1.8, label="NAD⁺")
    a.plot(t, X[:, i["Lac"]] - e.initial["Lac"], color=color(3), ls="--", label=f"Lac − {e.initial['Lac']:g}")
    a.text(880, 0.157, "NADH and Pyr fall together (1 : 1)", ha="right", va="bottom", fontsize=6, color=GREY)
    a.text(880, 0.040, "NAD⁺ and Lac rise together", ha="right", va="top", fontsize=6, color=GREY)
    a.set_xlabel("time (s)")
    a.set_ylabel("concentration (mM)")
    a.set_title("(a) small molecules", loc="left")
    a.legend(frameon=False, loc="center right")
    a = ax[0, 1]
    for j, s in enumerate(ex.net.enzyme_species):
        a.plot(t[1:], X[1:, i[s]] / e.Et, color=color(j), label=s)
    a.set_xscale("log")
    a.set_xlabel("time (s, log)")
    a.set_ylabel("fraction of enzyme")
    a.set_title("(b) enzyme forms (sum = Et = 24 nM)", loc="left")
    a.legend(frameon=False, loc="upper left", ncols=2)
    a = ax[0, 2]
    A = sum(c * X[:, i[s]] for s, c in ex.w.items())
    bound = 6.22 * (X[:, i["EN"]] + X[:, i["ENP"]])
    a.plot(t, A, color=GREY, label="A340, noise-free")
    td, _ = ex.series("A")
    a.plot(
        td, ex.curve(ex.k_truth, "A", td), "o", ms=2.2, mfc="none", mew=0.5, color=color(1), label="readings every 10 s"
    )
    note = f"A340 = 6.22 × (NADH + EN + ENP)\nbound NADH adds ≤ {bound.max():.1e}\n(noise σ = {ex.sigma:g})"
    a.text(0.97, 0.28, note, transform=a.transAxes, ha="right", fontsize=6, color=GREY)
    a.set_xlabel("time (s)")
    a.set_ylabel("A340")
    a.set_title("(c) what the photometer sees", loc="left")
    a.legend(frameon=False, loc="upper right")
    _save(fig, "fig2_forward")


# ---------------------------------------------------------------- 3 data
GROUPS = [
    ("(a) training: A, B", ["A", "B"]),
    ("(b) held out: C, D′", ["C", "Dp"]),
    ("(c) design check: D1, D0", ["D1", "D0"]),
]


def _label(ex: Example, name: str) -> str:
    e = ex.p.experiments[name]
    ini = ", ".join(f"{s} {v:g}" for s, v in e.initial.items() if v > 0)
    return f"{_name(name)}: {ini} mM, Et {e.Et * 1e6:.0f} nM"


def data(ex: Example) -> None:
    fig, ax = new_figure(DOUBLE, 2.25, ncols=3, sharey=True)
    for j, (title, names) in enumerate(GROUPS):
        a = ax[0, j]
        for c, n in enumerate(names):
            t, y = ex.series(n)
            a.plot(t, y, "o", ms=2.0, mfc="none", mew=0.5, color=color(c), label=_label(ex, n))
            clean = ex.truth["sets"][n].get("clean_A340")
            if clean is not None and len(clean) == len(t):
                a.plot(t, clean, color=color(c), lw=0.8, alpha=0.8)
        a.set_title(title, loc="left")
        a.set_xlabel("time (s)")
        a.legend(frameon=False, loc="upper right" if j == 2 else "lower right", fontsize=5.3)
    ax[0, 0].set_ylabel("A340")
    ax[0, 0].set_ylim(-0.05, 1.45)
    _save(fig, "fig3_data")


# ---------------------------------------------------------------- 4 inverse problem (classical least squares)
def inverse(ex: Example) -> dict:
    p = ex.p
    lay = p.layout
    prior = p.prior_vector()
    train = ["A", "B"]
    exps = [(p.experiments[n].initial, p.experiments[n].Et, *ex.series(n), []) for n in train]
    args = (
        ex.net, lay, exps, ex.w, ex.sigma, prior,
        float(p.identification["pinn"].get("w_prior", 1e-2)), float(p.prior.get("sd_decades", 1.5)),
    )  # fmt: skip
    log: list[tuple[np.ndarray, float]] = []
    n_par = len(prior)

    def f(theta, a):
        r = _residuals(theta, a)
        log.append((np.array(theta, float), float(np.sqrt(np.sum(r[:-n_par] ** 2)))))
        return r

    least_squares(f, prior, args=(args,), method="trf", x_scale=1.0, max_nfev=200, diff_step=1e-4)
    best, walk = np.inf, []
    for n, (th, z) in enumerate(log):
        if z < best:
            best = z
            walk.append((n + 1, th, z))
    picks = [walk[0]]
    for thr in (10.0, 3.0):
        hit = next((w for w in walk if w[2] <= thr and w[0] > picks[-1][0]), None)
        if hit is not None:
            picks.append(hit)
    if walk[-1][0] != picks[-1][0]:
        picks.append(walk[-1])
    names = (
        ["start (prior centre)"] + [f"evaluation {w[0]}" for w in picks[1:-1]] + [f"end (evaluation {picks[-1][0]})"]
    )
    fig, ax = new_figure(DOUBLE, 2.25, ncols=3)
    for j, n in enumerate(train):
        a = ax[0, j]
        t, y = ex.series(n)
        a.plot(t, y, "o", ms=2.0, mfc="none", mew=0.5, color=GREY, label="data")
        tt = np.linspace(0, t.max(), 300)
        for c, ((_, th, z), lab) in enumerate(zip(picks, names, strict=True)):
            kf, kr = lay.rates(th)
            a.plot(tt, ex.curve(ex.net.rate_vector(kf, kr), n, tt), color=color(c), lw=1.0, label=f"{lab}: {z:.1f}σ")
        a.set_title(f"({'ab'[j]}) training set {n}", loc="left")
        a.set_xlabel("time (s)")
        a.set_ylabel("A340")
        a.legend(frameon=False, loc="upper right" if n == "A" else "lower right")
    a = ax[0, 2]
    a.plot([w[0] for w in walk], [w[2] for w in walk], color=GREY, drawstyle="steps-post")
    for c, (n_, _, z) in enumerate(picks):
        a.plot([n_], [z], "o", color=color(c), ms=3.5)
    a.axhline(1.0, color=color(2), lw=0.7, ls=":")
    a.text(len(log), 1.08, "noise level", ha="right", va="bottom", fontsize=6, color=color(2))
    a.set_yscale("log")
    a.set_xlabel("model evaluations (A and B solved each time)")
    a.set_ylabel("training RMSE / σ (best so far)")
    a.set_title("(c) least squares walking down", loc="left")
    _save(fig, "fig4_inverse")
    kf, kr = lay.rates(picks[-1][1])
    return {
        "evaluations": len(log),
        "steps": [(w[0], round(w[2], 2)) for w in picks],
        "kf": kf.tolist(),
        "kr": kr.tolist(),
    }


# ---------------------------------------------------------------- 5 PINN training and restart choice
def pinn(ex: Example) -> None:
    fig, ax = new_figure(DOUBLE, 2.3, ncols=3)
    a = ax[0, 0]
    s0 = ex.samples[0]
    hist = s0["history"]
    hist = ast.literal_eval(hist) if isinstance(hist, str) else hist
    adam = [h for h in hist if isinstance(h[0], int)]
    n_adam = int(ex.p.identification["pinn"]["adam_iters"])
    x_end = n_adam + 400
    xs = [h[0] for h in adam] + [x_end]
    end = hist[-1][2]
    terms = [("data", "data: mean (residual/σ)²"), ("slow", "physics residual"), ("ic", "initial condition"),
             ("prior", "prior")]  # fmt: skip
    for c, (key, lab) in enumerate(terms):
        a.plot(xs, [h[2][key] for h in adam] + [end[key]], color=color(c), marker="o", ms=1.8, label=lab)
    a.axvspan(n_adam, x_end, color="0.92", lw=0)
    a.text(n_adam + 200, 30, "L-BFGS", ha="center", fontsize=6, color=GREY)
    a.set_yscale("log")
    a.set_xlabel("Adam iteration")
    a.set_ylabel("loss term")
    a.set_title(f"(a) one PINN training (seed 0, restart {s0['restart']})", loc="left")
    a.legend(frameon=False, loc="lower left", fontsize=5.5)
    aw = a.twinx()
    ramp = int(0.7 * n_adam)
    it = np.arange(0, n_adam + 1)
    aw.plot(it, 10 ** (4 * np.minimum(1, it / ramp)), color=GREY, lw=0.7, ls=":")
    aw.plot([n_adam, x_end], [1e4, 1e4], color=GREY, lw=0.7, ls=":")
    aw.set_yscale("log")
    aw.set_ylabel("physics weight (dotted)", color=GREY)
    a = ax[0, 1]
    sel = ex.fit["restart_selection"]
    for s, v in sel.items():
        z = [r["ode_train_rmse"] / ex.sigma for r in v["restarts"]]
        x = int(s) + np.linspace(-0.25, 0.25, len(z))
        a.plot(x, z, "o", ms=2.6, mfc="none", mew=0.6, color=GREY)
        ch = v["chosen_restart"]
        a.plot([x[ch]], [z[ch]], "o", ms=4.2, color=color(0))
    a.axhline(3.0, color=color(5), lw=0.7, ls="--")
    a.text(3.45, 3.05, "3σ: sample dropped above", ha="right", va="bottom", fontsize=6, color=GREY)
    a.axhline(1.0, color=color(2), lw=0.7, ls=":")
    a.set_xticks(range(len(sel)))
    a.set_xticklabels([f"seed {s}" for s in sel])
    a.set_ylabel("training RMSE / σ (ODE re-solve)")
    a.set_title("(b) 8 restarts per seed, best one kept", loc="left")
    a = ax[0, 2]
    pz = [r["pinn_rmse"] / ex.sigma for v in sel.values() for r in v["restarts"]]
    oz = [r["ode_train_rmse"] / ex.sigma for v in sel.values() for r in v["restarts"]]
    a.plot(pz, oz, "o", ms=2.6, mfc="none", mew=0.6, color=GREY, label="restart")
    chp = [v["restarts"][v["chosen_restart"]]["pinn_rmse"] / ex.sigma for v in sel.values()]
    cho = [v["restarts"][v["chosen_restart"]]["ode_train_rmse"] / ex.sigma for v in sel.values()]
    a.plot(chp, cho, "o", ms=4.2, color=color(0), label="kept")
    hi = max(max(pz), max(oz)) * 1.08
    a.plot([0.9, hi], [0.9, hi], color=GREY, lw=0.6, ls=":")
    a.text(hi * 0.97, hi * 0.9, "equal", ha="right", va="top", fontsize=6, color=GREY)
    a.set_xlim(0.9, hi)
    a.set_ylim(0.9, hi)
    a.set_xlabel("network curve RMSE / σ")
    a.set_ylabel("ODE re-solve RMSE / σ")
    a.set_title("(c) the ODE re-solve is the judge", loc="left")
    a.legend(frameon=False, loc="lower right")
    _save(fig, "fig5_pinn")


# ---------------------------------------------------------------- 6 what the data pin down
def half_time(t, y) -> float:
    d = y - y[0]
    tot = d[-1]
    j = int(np.argmax(np.abs(d) >= 0.5 * abs(tot)))
    if j == 0:
        return float(t[0])
    return float(t[j - 1] + (t[j] - t[j - 1]) * (0.5 * tot - d[j - 1]) / (d[j] - d[j - 1]))


def _half_times(ex: Example, models: list[dict]) -> np.ndarray:
    """log10(half-reaction time / truth) per model (rows) and set (columns)."""
    out = np.zeros((len(models), len(SETS)))
    for j, n in enumerate(SETS):
        t, _ = ex.series(n)
        tt = np.linspace(0.0, float(t.max()), 2000)
        ref = half_time(tt, ex.curve(ex.k_truth, n, tt))
        for i, m in enumerate(models):
            out[i, j] = np.log10(half_time(tt, ex.curve(ex.rates(m), n, tt)) / ref)
    return out


def _set_axis(a, trained: list[str]) -> None:
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(range(len(SETS)))
    a.set_xticklabels([_name(n) + ("*" if n in trained else "") for n in SETS])
    a.set_xlabel("set (* used for training)")
    for j in (1.5, 3.5):
        a.axvline(j, color="0.85", lw=0.6)
    for x, g in SET_GROUPS:
        a.text(x, 1.0, g, transform=a.get_xaxis_transform(), ha="center", va="bottom", fontsize=6, color=GREY)
    a.set_ylabel("log10(half-time / truth)")


def identifiability(ex: Example) -> None:
    fig, ax = new_figure(DOUBLE, 2.5, ncols=2, gridspec_kw={"width_ratios": [1.45, 1.0]})
    a = ax[0, 0]
    labels = [_k(i, "₊") for i in range(1, 6)] + [_k(i, "₋") for i in range(1, 6)] + ["K′"]
    truth = np.log10(np.r_[ex.truth["kf"], ex.truth["kr"], 23.0])

    def vec(m):
        return np.log10(np.r_[m["kf"], m["kr"], m.get("Kp") or 23.0])

    prior = ex.p.prior_vector()
    pk, pr = ex.p.layout.rates(prior)
    pv = np.log10(np.r_[pk, pr, 10 ** prior[-1]])
    sd = float(ex.p.prior.get("sd_decades", 1.5))
    x = np.arange(len(labels))
    free = np.array([True] * 9 + [False, True])  # k₋5 follows from Haldane
    for j in x[free]:
        a.add_patch(_mpl().Rectangle((j - 0.3, pv[j] - truth[j] - sd), 0.6, 2 * sd, color="0.93", lw=0))
    a.plot(x[free], (pv - truth)[free], "_", color=GREY, ms=9, mew=1.0, label=f"prior centre ± {sd:g} decades")
    for c, m in enumerate(ex.samples):
        a.plot(x - 0.15 + 0.1 * c, vec(m) - truth, "o", ms=3.0, color=color(0), alpha=0.9,
               label="PINN samples" if c == 0 else None)  # fmt: skip
    for c, m in enumerate(ex.classical):
        a.plot(x + 0.05 + 0.08 * c, vec(m) - truth, "s", ms=2.8, mfc="none", mew=0.6, color=color(1),
               label="classical fits" if c == 0 else None)  # fmt: skip
    a.axhline(0, color="black", lw=0.6)
    a.set_xticks(x)
    a.set_xticklabels(labels)
    a.set_ylabel("log10(value / truth)")
    a.set_title("(a) the constants themselves: spread over decades", loc="left")
    a.legend(frameon=False, loc="lower left", fontsize=5.5, ncols=3)
    a.text(9, 0.97, "Haldane", transform=a.get_xaxis_transform(), ha="center", va="top", fontsize=5.5, color=GREY)
    a = ax[0, 1]
    hp, hc = _half_times(ex, ex.samples), _half_times(ex, ex.classical)
    for j in range(len(SETS)):
        a.plot(j - 0.15 + 0.1 * np.arange(len(hp)), hp[:, j], "o", ms=3.0, color=color(0))
        a.plot(j + 0.05 + 0.08 * np.arange(len(hc)), hc[:, j], "s", ms=2.8, mfc="none", mew=0.6, color=color(1))
    _set_axis(a, ["A", "B"])
    a.set_title("(b) half-reaction time of each set", loc="left", pad=11)
    _save(fig, "fig6_identifiability")


# ---------------------------------------------------------------- 7 predictions of the samples
def predictions(ex: Example) -> None:
    names = [("A", "train"), ("Dp", "held out"), ("D1", "design check"), ("D0", "design check")]
    fig, ax = new_figure(DOUBLE, 2.2, ncols=4)
    for j, (n, g) in enumerate(names):
        a = ax[0, j]
        t, y = ex.series(n)
        tt = np.linspace(0.0, float(t.max()), 400)
        a.plot(t, y, "o", ms=2.0, mfc="none", mew=0.5, color=GREY, label="data")
        for c, m in enumerate(ex.samples):
            a.plot(tt, ex.curve(ex.rates(m), n, tt), color=color(c), lw=0.9, label=f"PINN sample {m['seed']}")
        a.plot(tt, ex.curve(ex.k_truth, n, tt), **TRUTH_STYLE, label="truth (nominal Et)")
        a.set_title(f"({'abcd'[j]}) {_name(n)}: {g}", loc="left")
        a.set_xlabel("time (s)")
        if j == 0:
            a.set_ylabel("A340")
            a.legend(frameon=False, loc="upper right", fontsize=5.3)
        e = ex.p.experiments[n]
        txt = f"Lac {e.initial.get('Lac', 0):g} mM, Pyr {e.initial.get('Pyr', 0):g} mM\nEt {e.Et * 1e6:.0f} nM"
        if n == "A":
            a.text(0.97, 0.3, txt, transform=a.transAxes, ha="right", va="bottom", fontsize=5.5, color=GREY)
        else:
            a.text(0.97, 0.96, txt, transform=a.transAxes, ha="right", va="top", fontsize=5.5, color=GREY)
    _save(fig, "fig7_predictions")


# ---------------------------------------------------------------- 8 from samples to the design
def _design_setup(ex: Example):
    p = ex.p
    des = p.settings(quick=False)["design"]
    dnet = p.design_network()
    ov_i = next((i for i, r in enumerate(p.reactions) if des["analyte"] in r["overall"].split()), None)
    Kp = float(des.get("Kp") or p.reactions[ov_i]["Kp"])
    d = AssayDesign(
        dnet, des["analyte"], des.get("partner"), des["free_enzyme"], float(des["analyte0_mM"]),
        float(des["conversion"]), float(des["time_s"]), Kp, overall=p.reactions[ov_i]["overall"],
    )  # fmt: skip

    def rates(m: dict):
        return design_rates(dnet, m, des, Kp, p.haldane_step, p.overall, ov_i)

    truth = rates({"kf": ex.truth["kf"], "kr": ex.truth["kr"], "Kp": Kp, "Kp_all": [Kp]})
    return d, rates, truth, float(ex.design["rho_star"])


def _left_vs_enzyme(a, d, rho, models, k_truth, lines, tmin) -> None:
    Et = np.geomspace(15e-6, 400e-6, 45)
    for lab, k, c in models:
        a.plot(Et * 1e6, [d.analyte_at(k, E, rho) * 1e3 for E in Et], color=c, lw=1.0, label=lab)
    a.plot(Et * 1e6, [d.analyte_at(k_truth, E, rho) * 1e3 for E in Et], **TRUTH_STYLE, label="truth")
    a.axhline(
        d.target * 1e3, color=color(5), lw=0.7, label=f"goal: ≤ {d.target * 1e3:g} µM left ({d.conversion:.0%} used)"
    )
    for v, lab, c in lines:
        a.axvline(v, color=c, lw=0.8, ls="--")
        a.text(v * 1.04, 0.04, f"{lab}\n{v:.0f} nM", transform=a.get_xaxis_transform(), fontsize=6, color=c)
    a.plot([tmin], [d.target * 1e3], "x", color="black", ms=5, mew=1.0, ls="none", label=f"truth needs {tmin:.0f} nM")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_ylim(0.15, 90)
    ticks = [20, 30, 50, 100, 200, 400]
    a.set_xticks(ticks)
    a.set_xticklabels([str(v) for v in ticks])
    a.minorticks_off()
    a.set_xlabel("enzyme Et (nM)")
    a.set_ylabel(f"NADH left at {d.time_s:g} s (µM)")
    a.legend(loc="upper right", fontsize=5.5, facecolor="white", edgecolor="none", framealpha=0.92)


def design(ex: Example) -> dict:
    d, rates, k_truth, rho = _design_setup(ex)
    models = [(f"PINN sample {m['seed']}", rates(m), color(c)) for c, m in enumerate(ex.samples)]
    nom, rob = ex.design["nominal_Et_at_rho_star_nM"], ex.design["robust_Et_star_nM"]
    tmin = min_Et(d, rho, [k_truth]) * 1e6
    fig, ax = new_figure(DOUBLE * 0.62, 2.5)
    a = ax[0, 0]
    _left_vs_enzyme(a, d, rho, models, k_truth, [(nom, "nominal", GREY), (rob, "robust", color(0))], tmin)
    a.set_title(f"NADH 0.1 mM, ρ = Pyr/NADH = {rho:g}, Lac 0", loc="left")
    _save(fig, "fig8_design")
    return {"rho": rho, "nominal_nM": nom, "robust_nM": rob, "truth_min_nM": tmin}


# ---------------------------------------------------------------- 9 closing the loop: a design-check set in training
def loop(ex: Example) -> dict:
    import torch

    from enzopt.pinn.general import PinnExperiment, train_pinn
    from enzopt.pipeline.fit import RESTART_STRIDE, best_restarts, pinn_config

    p = ex.p
    ident = p.settings(quick=False)["identification"]
    seeds = list(range(int(ident["seeds"])))
    nr = int(ident["pinn"]["restarts"])
    dev, prec = ("cuda", "float32") if torch.cuda.is_available() else ("cpu", "float64")
    names = ["A", "B", "D0"]
    train = [(n, p.experiments[n], *ex.series(n)) for n in names]
    cfg = pinn_config(p, ident, [s + RESTART_STRIDE * j for s in seeds for j in range(nr)], dev, prec, ex.sigma)
    res = train_pinn([PinnExperiment(n, e.initial, e.Et, t, y) for n, e, t, y in train], p.prior_vector(), cfg)
    res, _ = best_restarts(p, res, seeds, train, {n: [] for n in names}, ex.sigma)
    new = [
        {"seed": r.seed, "kf": [float(v) for v in r.kf], "kr": [float(v) for v in r.kr], "Kp": r.Kp, "Kp_all": r.Kp_all}
        for r in res
    ]
    fit = {}
    for m in new:
        k = ex.rates(m)
        fit[m["seed"]] = {
            n: round(float(np.sqrt(np.mean((ex.curve(k, n, t) - y) ** 2))) / ex.sigma, 2) for n, _, t, y in train
        }
    fig, ax = new_figure(DOUBLE, 2.5, ncols=2)
    a = ax[0, 0]
    before, after = _half_times(ex, ex.samples), _half_times(ex, new)
    for j in range(len(SETS)):
        a.plot(j - 0.2 + 0.08 * np.arange(len(before)), before[:, j], "o", ms=2.8, mfc="none", mew=0.6, color=GREY,
               label="trained on A, B" if j == 0 else None)  # fmt: skip
        a.plot(j + 0.05 + 0.08 * np.arange(len(after)), after[:, j], "o", ms=3.0, color=color(0),
               label="trained on A, B, D0" if j == 0 else None)  # fmt: skip
    _set_axis(a, names)
    a.legend(frameon=False, loc="upper left", fontsize=5.5)
    a.set_title("(a) PINN samples before and after adding D0", loc="left", pad=11)
    a = ax[0, 1]
    d, rates, k_truth, rho = _design_setup(ex)
    ks = [rates(m) for m in new]
    rob_new = min_Et(d, rho, ks) * 1e6
    tmin = min_Et(d, rho, [k_truth]) * 1e6
    models = [(f"sample {m['seed']}", k, color(c)) for c, (m, k) in enumerate(zip(new, ks, strict=True))]
    lines = [(ex.design["robust_Et_star_nM"], "robust before", GREY), (rob_new, "robust after", color(0))]
    _left_vs_enzyme(a, d, rho, models, k_truth, lines, tmin)
    a.set_title(f"(b) design after adding D0 (ρ = {rho:g})", loc="left", pad=11)
    _save(fig, "fig9_loop")
    return {
        "robust_before_nM": ex.design["robust_Et_star_nM"],
        "robust_after_nM": rob_new,
        "truth_min_nM": tmin,
        "train_rmse_sigma": fit,
        "half_time_log10_after": after.round(3).tolist(),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--run", type=Path, required=True, help="a full run of projects/example_ldh_synth")
    ap.add_argument("--only", nargs="*", help="figure functions to draw (default: all)")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ex = Example(a.run)
    figs = {
        "mechanism": mechanism, "forward": forward, "data": data, "inverse": inverse, "pinn": pinn,
        "identifiability": identifiability, "predictions": predictions, "design": design, "loop": loop,
    }  # fmt: skip
    notes = {}
    for name, fn in figs.items():
        if a.only and name not in a.only:
            continue
        r = fn(ex)
        if r:
            notes[name] = r
    if notes:
        print(json.dumps(notes, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
