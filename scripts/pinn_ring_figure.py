"""How the PINN learns with the quasi-steady enzyme ring (docs/flow, section 5.3).

Trains one seed of ldh_wang2025 again with the run's settings (seed 0, its chosen restart) and records snapshots every
100 Adam and 20 L-BFGS iterations: the network curve of one well, its slope next to the ring's turnover rate, the loss
terms and the rate constants. Writes docs/flow/pinn_ring_training.png and .csv (needs torch; a GPU takes about a
minute).

    uv run python scripts/pinn_ring_figure.py [run folder]   # default: the latest completed ldh_wang2025 run
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from enzopt.core.project import load_project  # noqa: E402
from enzopt.figures import DOUBLE, color, new_figure, write_csv  # noqa: E402
from enzopt.pipeline.fit import RESTART_STRIDE, pinn_config  # noqa: E402
from enzopt.pipeline.runner import list_runs, read_json  # noqa: E402

PROJECT = ROOT / "projects" / "ldh_wang2025"
OUT = ROOT / "docs" / "flow"
WELL = "r2_c01_A"
NOMINAL_NADH = {"2": 0.5, "3": 0.4, "4": 0.6}  # nominal NADH of the training runs (project.yaml)
SEED = 0


def latest_run() -> Path:
    for d in list_runs(ROOT / "runs" / PROJECT.name):
        if (read_json(d / "manifest.json", {}) or {}).get("status") == "completed":
            return d
    raise SystemExit("ldh_wang2025의 완료된 실행이 없음")


def main() -> None:
    import torch

    from enzopt.pinn.general import GeneralPinn, PinnExperiment, resolve_device

    run = Path(sys.argv[1]) if len(sys.argv) > 1 else latest_run()
    p = load_project(run / "inputs")
    ident = p.settings(False)["identification"]
    fit = read_json(run / "3_fit" / "fit_summary.json")
    restart = fit["restart_selection"][str(SEED)]["chosen_restart"]
    sigma = float(ident["noise_sigma"])
    train = [(n, e) for n, e in p.experiments.items() if e.role == "train"]
    exps = []
    for n, e in train:
        d = np.loadtxt(run / "2_data" / f"exp_{n}.csv", delimiter=",", skiprows=1, ndmin=2)
        exps.append(PinnExperiment(n, e.initial, e.Et, d[:, 0], d[:, 1]))
    b = [n for n, _ in train].index(WELL)  # batch index of the well (one seed: b = experiment index)
    device, dtype, _ = resolve_device("auto", "auto")
    cfg = pinn_config(p, ident, [SEED + RESTART_STRIDE * restart], device, dtype, sigma)
    model = GeneralPinn(exps, p.prior_vector(), cfg)
    n_st, nf = model.n_steps, len(model.lay.free_kr)
    snaps: list[dict] = []

    def snapshot(it: int, w: float) -> None:
        with torch.no_grad():
            kf, kr = model.rates(model.theta)
            kfB = kf.unsqueeze(1).expand(model.S, model.E, n_st).reshape(model.B, 1, n_st)
            krB = kr.unsqueeze(1).expand(model.S, model.E, n_st).reshape(model.B, 1, n_st)
            X, _xs, phi, dxi, _sig, _ = model.states(model.tau_obs, kfB, krB)
            f, g = model.step_rates(X.clamp_min(0.0), kfB, krB)
            J = torch.cat([f[j] - g[j] for j in range(n_st)], dim=-1) @ model.Wt  # ring turnover (slow coordinate)
            slope = dxi / model.Tb  # network slope d(free NADH)/dt
            pred = (X * model.coef_obs).sum(dim=-1)
            _total, parts = model.loss(w)
        th = model.theta.detach().double().cpu().numpy()[0]
        snaps.append(
            {
                "it": it,
                "w": w,
                "data": float(parts["data"][0]),
                "slow": float(parts["slow"][0]),
                "log_kf": np.log10(kf[0].double().cpu().numpy()),
                "log_kr": np.log10(kr[0].double().cpu().numpy()),
                "log_Kp": float(th[n_st + nf]),
                "pred": pred[b].double().cpu().numpy(),
                "slope": slope[b, :, 0].double().cpu().numpy(),
                "ring": J[:, :, 0].double().cpu().numpy(),  # every well
                "phi": phi[b].double().cpu().numpy(),
            }
        )

    lw0, lw1 = math.log(cfg.w_phys_start), math.log(cfg.w_phys_final)
    n_ramp = max(1, int(0.7 * cfg.adam_iters))

    def prog(_frac: float, msg: str) -> None:
        kind, num = msg.split()[0], int(msg.split()[1].split("/")[0])
        if kind == "Adam":
            it = num - 1
            w = math.exp(lw0 + (lw1 - lw0) * min(1.0, it / n_ramp))
        else:
            it = cfg.adam_iters + num
            w = cfg.w_phys_final
        snapshot(it, w)

    snapshot(0, cfg.w_phys_start)
    res = model.train(prog)[0]
    print(f"seed {SEED} restart {restart}: data {res.loss_parts['data']:.4f}, K' {res.Kp:.0f}, {len(snaps)} snapshots")
    figure(p, exps, b, snaps, cfg.adam_iters, sigma)


def figure(p, exps, b: int, snaps: list[dict], n_adam: int, sigma: float) -> None:
    import matplotlib.pyplot as plt

    fig, axes = new_figure(DOUBLE, 0.66 * DOUBLE, ncols=3, nrows=2)
    rows = []
    t = exps[b].t_obs
    its = np.array([s["it"] for s in snaps])
    # (a) the network curve of one well during training
    ax = axes[0, 0]
    ax.plot(t, exps[b].a_obs * 1e3, "o", ms=2.2, mfc="none", mew=0.5, color="0.2", label="data")
    picks = [0, 100, 400, 1000, 1999]
    shades = plt.get_cmap("viridis")(np.linspace(0.0, 0.85, len(picks) + 1))
    for c, it in enumerate(picks):
        s = snaps[int(np.argmin(np.abs(its - it)))]
        ax.plot(t, s["pred"] * 1e3, color=shades[c], lw=0.9, label=f"iter {s['it']}")
    ax.plot(t, snaps[-1]["pred"] * 1e3, color=shades[-1], lw=1.4, ls="--", label="end (L-BFGS)")
    ax.set(xlabel="time (s)", ylabel="NADH (µM)", title=f"(a) network curve, well {exps[b].name}")
    ax.legend(frameon=False, fontsize=5.5)
    # (b) slope of the curve vs the ring's turnover rate
    ax = axes[0, 1]
    for c, (it, ls) in enumerate(((100, ":"), (400, "--"), (snaps[-1]["it"], "-"))):
        s = snaps[int(np.argmin(np.abs(its - it)))]
        lab = "end" if s is snaps[-1] else f"iter {s['it']}"
        ax.plot(t, s["slope"] * 1e3, color=color(c), lw=1.0, ls=ls, label=lab)
        ax.plot(t, s["ring"][b] * 1e3, color=color(c), lw=0.0, marker="o", ms=1.8)
        rows += [["b_slope", s["it"], float(x), float(y)] for x, y in zip(t, s["slope"] * 1e3, strict=True)]
        rows += [["b_ring", s["it"], float(x), float(y)] for x, y in zip(t, s["ring"][b] * 1e3, strict=True)]
    ax.set(xlabel="time (s)", ylabel="d[NADH]/dt (µM/s)", title="(b) physics term: slope = ring rate?")
    ax.set_ylim(top=0.35)
    ax.legend(
        frameon=False,
        fontsize=5.5,
        ncol=3,
        loc="upper right",
        title="lines: curve slope · dots: ring rate",
        title_fontsize=5.5,
    )
    # (c) loss terms and the physics weight
    ax = axes[0, 2]
    ax.semilogy(its, [s["data"] for s in snaps], color=color(0), lw=1.0, label="data term (mean z²)")
    ax.semilogy(its, [s["slow"] for s in snaps], color=color(3), lw=1.0, label="physics residual")
    ax.axvline(n_adam, color="0.6", lw=0.6, ls=":")
    ax.text(n_adam + 30, 0.45, "L-BFGS", transform=ax.get_xaxis_transform(), fontsize=5.5, va="top", color="0.4")
    bx = ax.twinx()
    bx.semilogy(its, [s["w"] for s in snaps], color="0.55", lw=0.8, ls="--")
    bx.set_ylabel("physics weight (dashed)", fontsize=6, color="0.4")
    bx.tick_params(labelsize=5.5)
    ax.set(xlabel="iteration", ylabel="loss term", title="(c) losses while the weight rises")
    ax.legend(frameon=False, fontsize=5.5, loc="lower left")
    rows += [["c_loss", s["it"], s["data"], s["slow"]] for s in snaps]
    # (d) rate constants during training
    ax = axes[1, 0]
    rev = [i for i, st in enumerate(p.network.steps) if st.reversible]
    for i in range(len(p.network.steps)):
        ax.plot(its, [s["log_kf"][i] for s in snaps], color=color(i), lw=0.9, label=f"k+{i + 1}")
    for i in rev:
        dep = (i + 1) == p.haldane_step
        ax.plot(its, [s["log_kr"][i] for s in snaps], color=color(i), lw=0.9, ls=":" if dep else "--")
    ax.plot(its, [s["log_Kp"] for s in snaps], color="k", lw=1.0, ls="-.", label="K′")
    ax.axvline(n_adam, color="0.6", lw=0.6, ls=":")
    ax.set(xlabel="iteration", ylabel="log10 k", ylim=(0, 5.6), title="(d) rate constants (solid k+, dashed k−)")
    ax.legend(frameon=False, fontsize=5, ncol=3, loc="lower right")
    for s in snaps:
        rows.append(["d_logk", s["it"], *np.round(s["log_kf"], 4), *np.round(s["log_kr"], 4), round(s["log_Kp"], 4)])
    # (e) the ring's output along the final curve: enzyme distribution
    ax = axes[1, 1]
    phi = snaps[-1]["phi"]
    names = list(p.network.enzyme_species)
    ax.stackplot(t, phi.T * 100, labels=names, colors=[color(i) for i in range(len(names))], alpha=0.85)
    ax.set(xlabel="time (s)", ylabel="share of enzyme (%)", ylim=(0, 100), title="(e) ring output: enzyme forms")
    ax.legend(fontsize=5.5, loc="upper center", ncol=len(names), framealpha=0.9, handlelength=1.0, columnspacing=0.8)
    rows += [["e_phi", "end", float(x), *np.round(v, 5)] for x, v in zip(t, phi, strict=True)]
    # (f) what the constants are fitted to: rate against concentration over all training wells
    ax = axes[1, 2]
    ring0 = -snaps[-1]["ring"][:, 0] * 1e3  # consumption rate at t = 0 for every well
    groups: dict[str, list] = {}
    for i, e in enumerate(exps):
        sel = e.t_obs <= 50.0
        slope = -np.polyfit(e.t_obs[sel], e.a_obs[sel], 1)[0] * 1e3
        run_no = e.name.split("_")[0][1:]
        key = f"run {run_no} (NADH {NOMINAL_NADH.get(run_no, '?')} mM)"
        groups.setdefault(key, []).append((e.initial["Pyr"], slope, ring0[i]))
        rows.append(["f_rate", e.name, e.initial["Pyr"], slope, ring0[i]])
    for c, (key, v) in enumerate(sorted(groups.items())):
        v = np.array(sorted(v))
        ax.plot(v[:, 0], v[:, 1], "o", ms=2.2, mfc="none", mew=0.5, color=color(c))
        ax.plot(v[:, 0], v[:, 2], "-", lw=0.9, color=color(c), label=key)
    ax.set_xscale("log")
    ax.set(xlabel="pyruvate at t = 0 (mM)", ylabel="NADH used (µM/s)", title="(f) rate vs concentration, 108 wells")
    ax.set_ylim(-0.35, 1.95)
    ax.legend(
        frameon=False,
        fontsize=5.5,
        loc="upper left",
        title="circles: data, first 50 s · lines: ring at t = 0",
        title_fontsize=5.5,
    )
    fig.suptitle(
        f"How the PINN learns with the enzyme ring (ldh_wang2025, seed {SEED}, σ = {sigma * 1e3:g} µM)", fontsize=7.5
    )
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "pinn_ring_training.png", dpi=200)
    plt.close(fig)
    write_csv(
        OUT / "pinn_ring_training.csv", ["panel", "key", "a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k"], rows
    )
    print(f"wrote {OUT / 'pinn_ring_training.png'}")


if __name__ == "__main__":
    main()
