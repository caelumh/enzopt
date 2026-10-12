"""Standard figure set (spec area 13), one function per figure. Every figure = PDF + PNG + CSV of plotted values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .figures import DOUBLE, SINGLE, color, linestyle, marker, new_figure, save, sig3, table_figure


# ------------------------------------------------------------------------------------------------ ① verdict
def fig_network(out: Path, vd: dict, name: str = "fig1a_network") -> dict:
    """Complex graph: one row per linkage class (box colour), nodes coloured by strongly connected component,
    one-way reactions drawn as bold red arrows."""
    lcs = vd["linkage_classes"]
    sccs = vd["strongly_connected_components"]
    labels = vd["complexes"]
    scc_of = {i: k for k, comp in enumerate(sccs) for i in comp}
    lc_of = {i: k for k, comp in enumerate(lcs) for i in comp}
    # colour SCCs in linkage-class order: when SCC = linkage class (weakly reversible) node and box colours agree
    scc_col: dict[int, int] = {}
    for comp in lcs:
        for i in comp:
            scc_col.setdefault(scc_of[i], len(scc_col))
    col = {i: c for comp in lcs for c, i in enumerate(comp)}
    edges = [tuple(e) for e in vd["edges"]]
    eset = set(edges)
    # an edge that skips over other complexes in its row is routed as an elbow (up, across, down) above the row — the
    # reverse of a reversible pair below it — each on its own lane, and the row's box grows to hold the lanes
    skip: dict[tuple, tuple[int, int]] = {}
    lanes: dict[tuple[int, int], list[list[tuple[int, int]]]] = {}
    for u, v in sorted(edges, key=lambda e: abs(col[e[1]] - col[e[0]])):
        if abs(col[v] - col[u]) < 2:
            continue
        side = 1 if ((v, u) not in eset or u < v) else -1
        lo, hi = sorted((col[u], col[v]))
        used = lanes.setdefault((lc_of[u], side), [])
        k = next(k for k in range(len(used) + 1) if k == len(used) or all(hi < a or lo > b for a, b in used[k]))
        if k == len(used):
            used.append([])
        used[k].append((lo, hi))
        skip[(u, v)] = (side, k)
    ends: dict[tuple[int, int], list[tuple]] = {}
    for (u, v), (side, _) in skip.items():
        ends.setdefault((u, side), []).append((u, v))
        ends.setdefault((v, side), []).append((u, v))
    lane0, lane_step, label_edge = 0.32, 0.12, 0.17

    def half(r: int, side: int) -> float:
        n = len(lanes.get((r, side), []))
        return 0.28 if n == 0 else lane0 + lane_step * (n - 1) + 0.1

    yc = [0.0]
    for r in range(1, len(lcs)):
        yc.append(yc[-1] - half(r - 1, -1) - 0.44 - half(r, 1))
    pos = {i: (col[i] * 1.6, yc[lc_of[i]]) for comp in lcs for i in comp}
    top, bottom = half(0, 1) + 0.22, yc[-1] - half(len(lcs) - 1, -1) - 0.22
    width = DOUBLE if max(len(c) for c in lcs) > 3 else SINGLE
    fig, ax = new_figure(width, 0.45 + 0.42 * (top - bottom))
    from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
    from matplotlib.path import Path as MplPath

    ax = ax[0, 0]
    for row, comp in enumerate(lcs):
        xs = [pos[i][0] for i in comp]
        ax.add_patch(
            FancyBboxPatch(
                (min(xs) - 0.6, yc[row] - half(row, -1)),
                max(xs) - min(xs) + 1.2,
                half(row, 1) + half(row, -1),
                boxstyle="round,pad=0.02",
                fc="none",
                ec=color(row),
                lw=0.8,
                ls="--",
            )
        )
    rows = []
    for u, v in edges:
        (x0, y0), (x1, y1) = pos[u], pos[v]
        rev = (v, u) in eset
        off = 0.06 if rev else 0.0
        sgn = 1 if u < v else -1
        one_way = not rev
        style = dict(
            arrowstyle="-|>",
            lw=1.6 if one_way else 0.6,
            color="#D55E00" if one_way else "0.25",
            shrinkA=0,
            shrinkB=0,
            mutation_scale=7,
        )
        if (u, v) in skip:
            side, k = skip[(u, v)]
            ly = y0 + side * (lane0 + lane_step * k)
            # several elbows ending at one complex are spread so their vertical legs do not overlap
            xa, xb = (
                x + (ends[(n, side)].index((u, v)) - (len(ends[(n, side)]) - 1) / 2) * 0.14
                for x, n in ((x0, u), (x1, v))
            )
            path = MplPath(
                [(xa, y0 + side * label_edge), (xa, ly), (xb, ly), (xb, y1 + side * label_edge)],
                [MplPath.MOVETO, MplPath.LINETO, MplPath.LINETO, MplPath.LINETO],
            )
            style = {key: val for key, val in style.items() if not key.startswith("shrink")}
            ax.add_patch(FancyArrowPatch(path=path, **style))
        else:
            ax.annotate(
                "",
                xy=(x1 - 0.45 * np.sign(x1 - x0 or 1), y1 + sgn * off),
                xytext=(x0 + 0.45 * np.sign(x1 - x0 or 1), y0 + sgn * off),
                arrowprops=style,
            )
        rows.append(["edge", labels[u], labels[v], lc_of[u], scc_of[u], "", "", "one-way" if one_way else "reversible"])
    for i, (x, y) in pos.items():
        ax.text(
            x,
            y,
            labels[i],
            ha="center",
            va="center",
            fontsize=6.5,
            bbox=dict(
                boxstyle="round,pad=0.25", fc=color(scc_col[scc_of[i]]) + "33", ec=color(scc_col[scc_of[i]]), lw=0.7
            ),
        )
        rows.append(["node", labels[i], "", lc_of[i], scc_of[i], x, y, ""])
    xmax = max(p[0] for p in pos.values())
    ax.set_xlim(-0.9, xmax + 0.9)
    ax.set_ylim(bottom, top)
    ax.axis("off")
    wr = "weakly reversible" if vd["weakly_reversible"] else "NOT weakly reversible"
    lc = f"{len(lcs)} linkage class" + ("" if len(lcs) == 1 else "es")
    steps = f"{len(vd['steps'])} steps, " if vd.get("steps") else ""
    ax.set_title(
        f"{vd['name']}: {steps}{len(labels)} complexes, {lc}, deficiency {vd['deficiency']}, {wr}",
        loc="left",
    )
    return save(fig, out, name, ["kind", "complex", "to", "linkage_class", "scc", "x", "y", "direction"], rows)


def fig_verdict_table(out: Path, verdicts: dict, name: str = "fig1b_verdict_table") -> dict:
    header = [
        "network",
        "species",
        "complexes",
        "linkage classes",
        "rank",
        "deficiency",
        "weakly reversible",
        "conservation laws",
        "149 grade",
    ]
    rows = []
    for label, v in verdicts.items():
        rows.append(
            [
                label,
                len(v["species"]),
                v["n_complexes"],
                v["n_linkage_classes"],
                v["rank"],
                v["deficiency"],
                "yes" if v["weakly_reversible"] else "no",
                len(v["conservation_laws"]),
                v["grade"],
            ]
        )
    return table_figure(
        out, name, header, rows, title="CRNT verdict (149 grade A: persistence + one equilibrium per class)"
    )


# ------------------------------------------------------------------------------------------------ ② data
def fig_data(
    out: Path,
    series: dict,
    temps: dict,
    roles: dict,
    name: str = "fig2_data",
    iso_range: float = 0.2,
    obs: str = "A340",
) -> dict:
    """(a) progress curves; (b) temperature per experiment: min–max bar and mean, with the isothermal band."""
    fig, axes = new_figure(DOUBLE, 0.36 * DOUBLE, ncols=2, gridspec_kw={"width_ratios": [1.6, 1]})
    ax, bx = axes[0]
    rows = []
    for i, (k, (t, y)) in enumerate(series.items()):
        ax.plot(t, y, marker(i), ms=2.2, mfc="none", mew=0.6, color=color(i), label=f"{k} ({roles[k]})")
        rows += [[k, obs, float(a), float(b)] for a, b in zip(t, y, strict=True)]
    ax.set(xlabel="time (s)", ylabel=obs, title="(a) progress curves")
    ax.legend(loc="lower right", bbox_to_anchor=(1.0, 0.1), ncol=2, frameon=False)
    have = [(i, k, v) for i, (k, v) in enumerate(temps.items()) if v is not None]
    if have:
        means = []
        for j, (i, k, (t, y)) in enumerate(have):
            lo, hi, mu = float(y.min()), float(y.max()), float(y.mean())
            means.append(mu)
            bx.plot([j, j], [lo, hi], "-", lw=2.2, color=color(i), solid_capstyle="butt")
            bx.plot([j], [mu], marker="_", ms=9, mew=1.2, color="k")
            rows += [[k, "T_C_min", float(t[0]), lo], [k, "T_C_mean", float(t[0]), mu], [k, "T_C_max", float(t[0]), hi]]
        c = float(np.mean(means))
        bx.axhspan(
            c - iso_range / 2, c + iso_range / 2, color="0.88", zorder=0, label=f"isothermal band ({iso_range} °C)"
        )
        bx.set_xticks(range(len(have)), [k for _, k, _ in have])
        bx.legend(loc="upper right", frameon=False)
    bx.set(xlabel="experiment", ylabel="temperature (°C)", title="(b) cuvette temperature (min–max, mean)")
    return save(fig, out, name, ["experiment", "quantity", "time_s", "value"], rows)


# ------------------------------------------------------------------------------------------------ helpers
def write_rows(path, header, rows):
    from .figures import write_csv

    return write_csv(path, header, rows)


# ------------------------------------------------------------------------------------------------ ③ identification
def _names(p) -> dict:
    """Display names of the comparison models. The CSV ids stay "WR …"/"nonWR …"/"PINN (WR)" (tests, dashboard); when
    the network as written is not weakly reversible itself, figures call it "identified" and name the control by the
    steps it makes one-way."""
    from .core.crnt import verdict

    wr = bool(verdict(p.network.reactions(), p.network.name).weakly_reversible)
    steps = (p.control or {}).get("one_way_steps") or []
    ctrl = "nonWR" if wr else (f"control: step {', '.join(str(s) for s in steps)} one-way" if steps else "control")
    return {
        "wr": wr,
        "control": ctrl,
        "legend": {
            "WR": "WR" if wr else "identified",
            "WR-excluded": "WR-excluded" if wr else "excluded",
            "nonWR": ctrl,
        },
        "table": {"PINN (WR)": "PINN (WR)" if wr else "PINN", "nonWR": ctrl},
    }


def _model_curves(p, every, out_dir: Path, coeff: dict | None = None) -> dict:
    """{label: {exp: (t, signal)}} for WR samples, the non-WR control and the classical fit (from stage-③ files);
    ``coeff`` picks the signal (default: the primary observable)."""
    import json

    from .core.massaction import MassAction
    from .pipeline.common import safe_curve

    coeff = coeff or p.observable_weights()
    curves = {}
    fines = {}
    for name, _e, t, _y in every:
        fines[name] = np.linspace(0.0, float(t[-1]), 181)[1:]

    def add(label, net, k, activity=None):
        model = MassAction.from_network(net)
        curves[label] = {}
        for name, e, _t, _y in every:
            a = safe_curve(net, k, e.initial, e.Et, fines[name], coeff, model=model, activity=activity)
            if a is not None:
                curves[label][name] = (fines[name], a)

    for f in sorted((out_dir / "samples").glob("sample_seed*.json")):
        s = json.loads(f.read_text(encoding="utf-8"))
        tag = "WR" if s["kept"] else "WR-excluded"
        add(f"{tag} seed{s['seed']}", p.network, p.network.rate_vector(s["kf"], s["kr"]), s.get("activity"))
    ctrl = p.control_network()
    for f in sorted((out_dir / "control").glob("control_seed*.json")) if (out_dir / "control").exists() else []:
        s = json.loads(f.read_text(encoding="utf-8"))
        add(f"nonWR seed{s['seed']}", ctrl, ctrl.rate_vector(s["kf"], s["kr"]), s.get("activity"))
    for f in sorted((out_dir / "baseline").glob("fit_start*.json")) if (out_dir / "baseline").exists() else []:
        s = json.loads(f.read_text(encoding="utf-8"))
        if "kf" in s:
            add(f"classical start{s['start']}", p.network, p.network.rate_vector(s["kf"], s["kr"]))
    return curves


def _style(label: str, i: int):
    if label.startswith("WR-excluded"):
        return dict(color="0.6", ls=":", lw=0.7)
    if label.startswith("WR"):
        return dict(color="0.15", ls="-", lw=0.7, alpha=0.8)
    if label.startswith("nonWR"):
        return dict(color="#D55E00", ls=(0, (1, 1)), lw=1.2)
    return dict(color="#0072B2", ls="-.", lw=0.9)


def fit_figures(out: Path, p, every, samples: list, summary: dict) -> list:
    curves = _model_curves(p, every, out)
    data = {name: (t, y, e.role) for name, e, t, y in every}
    nm = _names(p)
    figs = []
    titles = (
        (
            "(a) training sets: data and identified WR models",
            "(b) held-out sets: WR models, non-WR control, classical fit",
        )
        if nm["wr"]
        else (
            "(a) training sets: data and identified models",
            f"(b) held-out sets: identified models, {nm['control']}, classical fit",
        )
    )
    for fig_name, role, title in (
        ("fig3a_train_fit", "train", titles[0]),
        ("fig3b_holdout", "holdout", titles[1]),
    ):
        names = [n for n, (_, _, r) in data.items() if r == role]
        if not names:
            continue
        ncol = min(4, len(names))
        nrow = -(-len(names) // ncol)
        fig, axes = new_figure(DOUBLE, 0.27 * DOUBLE * nrow + 0.06 * DOUBLE, ncols=ncol, nrows=nrow)
        rows = []
        handles = {}
        for j, n in enumerate(names):
            ax = axes[j // ncol, j % ncol]
            t, y, _ = data[n]
            ax.plot(t, y, "o", ms=2.0, mfc="none", mew=0.5, color=color(j))
            handles.setdefault("data", ax.lines[-1])
            rows += [["data", n, float(a), float(b)] for a, b in zip(t, y, strict=True)]
            for i, (label, cv) in enumerate(curves.items()):
                if role == "train" and not label.startswith("WR"):
                    continue
                if n not in cv:
                    continue
                tt, aa = cv[n]
                (ln,) = ax.plot(tt, aa, **_style(label, i))
                handles.setdefault(label.split(" ")[0], ln)
                rows += [[label, n, float(a), float(b)] for a, b in zip(tt, aa, strict=True)]
            ax.set_title(n, fontsize=7)
            ax.tick_params(labelsize=6)
            if j % ncol == 0:
                ax.set_ylabel(p.observable["name"])
            if j // ncol == nrow - 1 or j + ncol >= len(names):
                ax.set_xlabel("time (s)")
        for k in range(len(names), nrow * ncol):
            axes[k // ncol, k % ncol].axis("off")
        fig.legend(
            list(handles.values()),
            [nm["legend"].get(k, k) for k in handles],
            loc="lower center",
            ncol=len(handles),
            frameon=False,
            bbox_to_anchor=(0.5, -0.08),
        )
        fig.suptitle(title, fontsize=7.5)
        figs.append(save(fig, out, fig_name, ["model", "experiment", "time_s", p.observable["name"]], rows))
    # (c) rate constants
    rev = [i for i, st in enumerate(p.network.steps) if st.reversible]
    n_st = len(p.network.steps)
    names = [f"k+{i}" for i in range(1, n_st + 1)] + [f"k-{i + 1}" for i in rev]

    def logk(s):  # k₊ of every step, k₋ of the reversible ones
        return np.log10(np.concatenate([s["kf"], [s["kr"][i] for i in rev]]))

    fig, ax = new_figure(SINGLE if len(names) <= 12 else DOUBLE, 0.62 * SINGLE)
    ax = ax[0, 0]
    rows = []
    x = np.arange(len(names))
    for i, s in enumerate(samples):
        v = logk(s)
        st = (
            dict(marker=marker(i), ms=3, lw=0.6, color=color(i))
            if s["kept"]
            else dict(marker="x", ms=3, lw=0.4, color="0.6")
        )
        ax.plot(x, v, label=f"seed {s['seed']}" + ("" if s["kept"] else " (excluded)"), **st)
        rows += [
            [f"seed{s['seed']}", "kept" if s["kept"] else "excluded", nm, float(vv)]
            for nm, vv in zip(names, v, strict=True)
        ]
    import json

    labelled = False
    for f in sorted((out / "baseline").glob("fit_start*.json")) if (out / "baseline").exists() else []:
        s = json.loads(f.read_text(encoding="utf-8"))
        if "kf" in s:
            v = logk(s)
            ax.plot(x, v, "+", ms=5, color="#0072B2", label=None if labelled else "classical fit (each start)")
            labelled = True
            rows += [
                [f"classical start{s['start']}", "classical", nm, float(vv)] for nm, vv in zip(names, v, strict=True)
            ]
    ax.set_xticks(x, names)
    ax.set(ylabel="log10 k", title="(c) identified rate constants (log10): not identifiable")
    ax.legend(frameon=False, ncol=2)
    figs.append(save(fig, out, "fig3c_rate_constants", ["model", "status", "constant", "log10_k"], rows))
    if any(s.get("activity") for s in samples):
        figs.append(activity_figure(out, samples))
    # comparison table (F-9 / F-10)
    cmp = out / "holdout_comparison.csv"
    if cmp.exists():
        from .figures import read_csv

        _, rr = read_csv(cmp)
        hold = [n for n, (_, _, r) in data.items() if r == "holdout"]
        models = []
        for m, ident, *_rest in rr:
            if (m, ident) not in models:
                models.append((m, ident))
        # with further signals the table shows RMSE ÷ σ over every signal (column 6), else the primary RMSE
        col = 5 if p.signals and rr and len(rr[0]) > 5 else 4
        trows = []
        for m, ident in models:
            vals = {r[2]: r[col] for r in rr if (r[0], r[1]) == (m, ident)}
            trows.append(
                [nm["table"].get(m, m), ident]
                + [float(vals[n]) if vals.get(n) not in (None, "", "None") else "—" for n in hold]
            )
        title = (
            "Held-out extrapolation RMSE / σ (all signals)"
            if col == 5
            else f"Held-out extrapolation RMSE ({p.observable['name']})"
        )
        figs.append(
            table_figure(
                out,
                "fig3d_holdout_table",
                ["model", "id"] + [f"RMSE {n}" for n in hold],
                trows,
                title=title,
            )
        )
    if p.signals:
        figs.append(signals_figure(out, p, every, nm))
    return figs


def signals_figure(out: Path, p, every, nm: dict) -> dict:
    """(f) every signal of every training and held-out set: data and the comparison models (projects with further
    signals; panels (a)(b) show the primary signal only)."""
    run_dir = out.parent
    sigs = [(p.observable["name"], p.observable_weights())] + [(s["name"], dict(s["coefficients"])) for s in p.signals]
    names = [n for n, *_ in every]
    nrow, ncol = len(names), len(sigs)
    fig, axes = new_figure(DOUBLE, 0.2 * DOUBLE * nrow + 0.06 * DOUBLE, ncols=ncol, nrows=nrow)
    rows, handles = [], {}
    for c, (sig, coeff) in enumerate(sigs):
        curves = _model_curves(p, every, out, coeff)
        for r, (name, e, t, y) in enumerate(every):
            ax = axes[r, c]
            if c > 0:
                d = np.loadtxt(run_dir / "2_data" / f"sig_{name}__{sig}.csv", delimiter=",", skiprows=1, ndmin=2)
                t, y = d[:, 0], d[:, 1]
            ax.plot(t, y, "o", ms=1.8, mfc="none", mew=0.5, color="0.1")
            handles.setdefault("data", ax.lines[-1])
            rows += [["data", name, sig, float(a), float(b)] for a, b in zip(t, y, strict=True)]
            for i, (label, cv) in enumerate(curves.items()):
                if name not in cv or label.startswith("WR-excluded"):
                    continue
                tt, aa = cv[name]
                (ln,) = ax.plot(tt, aa, **_style(label, i))
                handles.setdefault(label.split(" ")[0], ln)
                rows += [[label, name, sig, float(a), float(b)] for a, b in zip(tt, aa, strict=True)]
            ax.tick_params(labelsize=5)
            if r == 0:
                ax.set_title(sig, fontsize=6.5)
            if c == 0:
                ax.set_ylabel(f"{name}{'' if e.role == 'train' else ' (held out)'}", fontsize=6)
            if r == nrow - 1:
                ax.set_xlabel("time (s)", fontsize=6)
    fig.legend(
        list(handles.values()),
        [nm["legend"].get(k, k) for k in handles],
        loc="lower center",
        ncol=len(handles),
        frameon=False,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle("(f) every signal: data and models", fontsize=7.5)
    return save(fig, out, "fig3f_signals", ["model", "experiment", "signal", "time_s", "value"], rows)


def activity_figure(out: Path, samples: list) -> dict:
    """(e) enzyme activity a(t) learned with the rate constants (option ``inactivation: shared``)."""
    from .core.activity import activity_curve, horizon

    T = max(horizon(s.get("activity")) or 0.0 for s in samples)
    t = np.linspace(0.0, T, 181)
    fig, ax = new_figure(SINGLE, 0.62 * SINGLE)
    ax = ax[0, 0]
    rows = []
    for i, s in enumerate(samples):
        a = activity_curve(s.get("activity"), t)
        st = dict(color=color(i), lw=0.9, ls=linestyle(i)) if s["kept"] else dict(color="0.6", lw=0.6, ls=":")
        ax.plot(t, a, label=f"seed {s['seed']}" + ("" if s["kept"] else " (excluded)"), **st)
        tag = "kept" if s["kept"] else "excluded"
        rows += [[f"seed{s['seed']}", tag, float(x), float(y)] for x, y in zip(t, a, strict=True)]
    ax.set(
        xlabel="time since enzyme added (s)",
        ylabel="active enzyme fraction a(t)",
        ylim=(0, 1.05),
        title="(e) enzyme activity learned with the rate constants",
    )
    ax.legend(frameon=False, ncol=2)
    return save(fig, out, "fig3e_activity", ["model", "status", "time_s", "activity"], rows)


# ------------------------------------------------------------------------------------------------ ④ simulation
def sim_figures(out: Path, rows: list, traj: list, hop: dict, analyte: str, obs: str = "A340") -> list:
    figs = []
    # (a) same class → same box, other class → other equilibrium
    fig, ax = new_figure(SINGLE, 0.7 * SINGLE)
    ax = ax[0, 0]
    data = {}
    for sample, run, cls, t, x, _a in traj:
        data.setdefault((sample, run, cls), ([], []))
        data[(sample, run, cls)][0].append(t)
        data[(sample, run, cls)][1].append(x)
    csv_rows = []
    lab = set()
    for (sample, run, cls), (t, x) in data.items():
        t = np.asarray(t)
        x = np.asarray(x)
        m = t > 0
        c = "0.35" if cls == "same" else "#D55E00"
        name = "same class (random starts)" if cls == "same" else "other class"
        ax.plot(t[m], x[m], color=c, lw=0.5, alpha=0.7, label=None if name in lab else name)
        lab.add(name)
        csv_rows += [[sample, run, cls, float(a), float(b)] for a, b in zip(t, x, strict=True)]
    ax.set_xscale("log")
    ax.set(xlabel="time (s)", ylabel=f"{analyte} (mM)", title="(a) same compatibility class → same equilibrium")
    ax.legend(frameon=False)
    figs.append(save(fig, out, "fig4a_classes", ["sample", "run", "class", "time_s", f"{analyte}_mM"], csv_rows))
    # (b) validation table (per sample)
    by = {}
    for r in rows:
        by.setdefault(r["sample"], []).append(r)
    trows = []
    for smp, rr in by.items():

        def mx(key, rr=rr):
            v = [x[key] for x in rr if x.get(key) is not None]
            return max(v) if v else float("nan")

        def mn(key, rr=rr):
            v = [x[key] for x in rr if x.get(key) is not None]
            return min(v) if v else float("nan")

        trows.append(
            [
                smp,
                len(rr),
                sum(1 for x in rr if x["ok"]),
                sum(1 for x in rr if x["suspect_numerical_error"]),
                mx("conservation_max_rel_drift"),
                mn("min_concentration"),
                mx("equilibrium_max_log_error"),
                mx("lyapunov_max_increase"),
            ]
        )
    figs.append(
        table_figure(
            out,
            "fig4b_validation",
            [
                "sample",
                "runs",
                "pass",
                "re-run (dt/10)",
                "max conservation drift",
                "min concentration (mM)",
                "max |ln Q/K|",
                "max dG increase",
            ],
            trows,
            title="CRNT validation (limits: drift < 1e-6, min > 0, |ln Q/K| < 0.01, dG <= 0)",
        )
    )
    # (c) Hopsan vs scipy, representative sample, full mechanism
    full = hop.get("full_representative", {})
    fig, axes = new_figure(SINGLE, 0.75 * SINGLE)
    ax = axes[0, 0]
    t = np.asarray(full.get("t", []))
    a = np.asarray(full.get("scipy_A340", []))
    csv_rows = [["scipy", float(x), float(y)] for x, y in zip(t, a, strict=True)]
    ax.plot(t, a, "-", color="#0072B2", lw=1.6, label="scipy (Radau)")
    if "hopsan_t" in full:
        th = np.asarray(full["hopsan_t"])
        ah = np.asarray(full["hopsan_A340"])
        ax.plot(
            th,
            ah,
            "--",
            color="#E69F00",
            lw=1.0,
            label=f"Hopsan (dt {sig3(full['timestep'])} s), max diff {sig3(full['max_abs_diff_A340'])}",
        )
        csv_rows += [["hopsan", float(x), float(y)] for x, y in zip(th, ah, strict=True)]
        title = f"(c) representative sample {full.get('sample')}, set {full.get('experiment')}: Hopsan vs scipy"
    else:
        title = f"(c) representative sample {full.get('sample')}: scipy only (Hopsan not used)"
    ax.set(xlabel="time (s)", ylabel=obs, title=title)
    ax.legend(frameon=False)
    figs.append(save(fig, out, "fig4c_hopsan_vs_scipy", ["backend", "time_s", obs], csv_rows))
    return figs


# ------------------------------------------------------------------------------------------------ ⑤ design
def design_figures(out: Path, res, summ: dict, check: dict) -> list:
    figs = []
    # (a) U-curve
    fig, ax = new_figure(SINGLE, 0.72 * SINGLE)
    ax = ax[0, 0]
    r = np.asarray(res.rhos)
    en = np.asarray(res.Et_nominal) * 1e6
    er = np.asarray(res.Et_robust) * 1e6
    ax.loglog(r, en, "o--", color="#E69F00", mfc="none", label="nominal (one sample)")
    ax.loglog(r, er, "s-", color="#0072B2", label=f"robust (all {summ['n_samples']} samples)")
    if res.rho_min:
        ax.axvline(
            res.rho_min, color="#D55E00", ls=":", lw=1.0, label=f"CRNT equilibrium limit ρ_min = {sig3(res.rho_min)}"
        )
    ax.plot(
        [res.rho_star],
        [res.Et_star * 1e6],
        "*",
        ms=8,
        color="k",
        label=f"ρ* = {sig3(res.rho_star)}, Et* = {sig3(res.Et_star * 1e6)} nM",
    )
    ax.set(
        xlabel="pyruvate / NADH ratio ρ",
        ylabel="min enzyme Et (nM)",
        title=f"(a) {summ['conversion'] * 100:.0f}% NADH conversion within {summ['time_s']:g} s "
        f"(K' = {sig3(summ['Kp'])})",
    )
    ax.legend(frameon=False)
    figs.append(
        save(
            fig,
            out,
            "fig5a_u_curve",
            ["rho", "Et_nominal_nM", "Et_robust_nM"],
            [[a, b, c] for a, b, c in zip(r, en, er, strict=True)],
        )
    )
    # (b) conversion of both designs on every sample
    fig, ax = new_figure(SINGLE, 0.62 * SINGLE)
    ax = ax[0, 0]
    rows = []
    for j, (label, mk, col) in enumerate((("nominal", "o", "#E69F00"), ("robust", "s", "#0072B2"))):
        cc = [c for c in res.check if c["design"] == label]
        x = np.arange(len(cc)) + (j - 0.5) * 0.18
        ax.plot(
            x,
            [c["conversion"] for c in cc],
            mk,
            color=col,
            mfc="none" if label == "nominal" else col,
            label=f"{label} design (Et {sig3(cc[0]['Et_mM'] * 1e6)} nM)",
        )
        rows += [[label, c["sample"], c["Et_mM"], c["conversion"]] for c in cc]
    ax.axhline(summ["conversion"], color="k", lw=0.6, ls=":", label=f"target {summ['conversion']}")
    ax.set_xticks(range(summ["n_samples"]), [f"s{s}" for s in summ["samples"]])
    ax.set(
        xlabel="rate-constant sample",
        ylabel=f"conversion at {summ['time_s']:g} s",
        title=f"(b) designs at ρ* = {sig3(res.rho_star)} applied to every sample",
    )
    ax.legend(frameon=False)
    figs.append(save(fig, out, "fig5b_conversion", ["design", "sample", "Et_mM", "conversion"], rows))
    # (c) design-condition prediction curves (+ Hopsan representative, + confirmation experiments)
    fig, ax = new_figure(SINGLE, 0.72 * SINGLE)
    ax = ax[0, 0]
    rows = []
    lab = set()
    for (label, smp), (t, x) in check["curves"].items():
        st = dict(color="#0072B2", ls="-", lw=0.7) if label == "robust" else dict(color="#E69F00", ls="--", lw=0.7)
        nm = f"{label} design (scipy, all samples)"
        ax.plot(t, np.asarray(x) * 1e3, label=None if nm in lab else nm, **st)
        lab.add(nm)
        rows += [[f"scipy {label}", smp, float(a), float(b) * 1e3] for a, b in zip(t, x, strict=True)]
    hop = check.get("_hopsan")
    if hop is not None:
        ax.plot(
            hop["t"],
            np.asarray(hop["x"]) * 1e3,
            ":",
            color="k",
            lw=1.2,
            label=f"Hopsan, representative (max diff {sig3(hop['max_abs_diff_mM'] * 1e3)} µM)",
        )
        rows += [
            ["hopsan robust", "representative", float(a), float(b) * 1e3]
            for a, b in zip(hop["t"], hop["x"], strict=True)
        ]
    for i, (name, c) in enumerate(check.get("_conf", {}).items()):
        ax.plot(
            c["t"],
            np.asarray(c["x"]) * 1e3,
            marker(i + 2),
            ms=2.5,
            mfc="none",
            color=color(i + 2),
            label=f"experiment {name} (ρ {sig3(c['rho'])}, Et {sig3(c['Et_nM'])} nM)",
        )
        rows += [[f"experiment {name}", name, float(a), float(b) * 1e3] for a, b in zip(c["t"], c["x"], strict=True)]
    target = (1 - summ["conversion"]) * summ["analyte0_mM"] * 1e3
    ax.axhline(target, color="k", lw=0.5, ls=":")
    ax.axvline(summ["time_s"], color="0.5", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylim(max(target / 30, 1e-3), summ["analyte0_mM"] * 1e3 * 1.3)
    ax.set(xlabel="time (s)", ylabel="NADH (µM)", title="(c) predicted NADH at the design condition")
    # figure-level legend below the axes, so long labels never squeeze the plot
    h, lb = ax.get_legend_handles_labels()
    fig.legend(h, lb, loc="outside lower center", frameon=False, fontsize=5.5, ncol=2)
    figs.append(save(fig, out, "fig5c_design_check", ["model", "id", "time_s", "NADH_uM"], rows))
    # (d) equilibrium limit table
    if res.equilibrium_limit:
        figs.append(
            table_figure(
                out,
                "fig5d_equilibrium_limit",
                ["rho", "conversion_eq", "NADH_left_uM"],
                [[e["rho"], e["conversion_eq"], e["NADH_left_uM"]] for e in res.equilibrium_limit],
                title=f"CRNT equilibrium limit (K' = {sig3(summ['Kp'])})\nindependent of rate constants",
                width=SINGLE,
            )
        )
    return figs


# ------------------------------------------------------------------------------------------ missing-step suggestion
def suggest_figure(
    out: Path, chosen: list, per_seed: list, summ: dict, rank: list, ver: list, sigma: float, obs: str = "A340"
) -> dict:
    """(a) where the written steps fall short: exp(−g) − 1 over time, per training experiment (median seed);
    (b) every candidate fitted by the PINN, training RMSE of the re-solved ODE in units of σ (dots: seeds);
    (c) the best candidates and the five steps fitted again by least squares: training and held-out RMSE."""
    i_med = int(np.argsort([s["rms_g"] for s in per_seed])[len(per_seed) // 2])
    pts = chosen[i_med].discrepancy
    y = np.expm1(-np.asarray(pts["g"], float))
    t = np.asarray(pts["t"], float)
    fig, ax = new_figure(DOUBLE, 2.5, ncols=3)
    a = ax[0, 0]
    rows = []
    for j, n in enumerate(dict.fromkeys(pts["experiment"])):
        sel = np.array([e == n for e in pts["experiment"]])
        a.plot(t[sel], y[sel], color=color(j), ls=linestyle(j), lw=1.0, label=n)
        rows += [[n, float(u), float(v)] for u, v in zip(t[sel], y[sel], strict=True)]
    a.axhline(0, color="0.5", lw=0.5)
    a.set(xlabel="time (s)", ylabel="rate shortfall exp(−g) − 1", title="(a) where the steps fall short")
    a.legend(frameon=False, fontsize=5, ncol=2)
    a = ax[0, 1]
    for xi, r in enumerate(rank):
        v = np.array([q["train_rmse"] for q in r["per_seed"]]) / sigma
        c = "0.4" if r["step"] is None else color(xi)
        a.plot(xi + np.linspace(-0.15, 0.15, len(v)), v, "o", ms=3, mfc="none", color=c)
        a.plot([xi - 0.25, xi + 0.25], [np.median(v)] * 2, color=c, lw=1.5)
        base_med = next(np.median([q["train_rmse"] for q in rr["per_seed"]]) for rr in rank if rr["step"] is None)
        if r["Ki_mM_median"] and np.median(v) * sigma < 0.9 * base_med:
            a.annotate(
                f"Ki {r['Ki_mM_median']:.2g}",
                (xi, np.median(v)),
                xytext=(0, 5),
                textcoords="offset points",
                ha="center",
                fontsize=5,
            )
    a.axhline(1, color=color(3), ls=":", lw=0.7)
    a.axhline(3, color=color(3), ls="--", lw=0.6)
    a.set_yscale("log")
    a.set_xticks(range(len(rank)), [r["name"].replace(" (as written)", "") for r in rank], fontsize=6)
    a.set(ylabel="training RMSE / σ (ODE re-solved)", title="(b) PINN fit with each candidate")
    a = ax[0, 2]
    if ver:
        ok = [v for v in ver if "error" not in v]
        lab = [v["network"].replace(" (as written)", "") for v in ok]
        x = np.arange(len(ok))
        a.bar(x - 0.18, [v["train_rmse"] / sigma for v in ok], 0.36, color="0.7", label="training")
        if all(v["holdout_rmse"] is not None for v in ok):
            a.bar(x + 0.18, [v["holdout_rmse"] / sigma for v in ok], 0.36, color=color(0), label="held-out")
        a.axhline(1, color=color(3), ls=":", lw=0.7)
        a.set_xticks(x, lab, fontsize=6)
        a.set_yscale("log")
        a.set(ylabel="RMSE / σ", title="(c) least-squares check")
        a.legend(frameon=False, fontsize=5.5)
    else:
        a.axis("off")
        a.text(0.5, 0.5, "no least-squares check", ha="center", va="center")
    fig.suptitle(
        f"missing-step search ({obs}, σ = {sigma:g}): rms g = {summ['rms_g_median']:.3f} "
        + "(includes the PINN's own misfit; the decision is the least-squares check)",
        fontsize=7.5,
    )
    return save(fig, out, "fig_suggest", ["experiment", "time_s", "shortfall"], rows)


def peak_figure(out: Path, res, product: str) -> dict:
    """⑤ goal ``peak``: (a) yield of the product against time at ρ* for every sample, with the robust stop time and
    its window; (b) worst-sample yield at the robust stop time for every ρ."""
    fig, axes = new_figure(DOUBLE, 0.36 * DOUBLE, ncols=2)
    ax, bx = axes[0, 0], axes[0, 1]
    rows = []
    s = res.star()
    for i, y in enumerate(res.yields):
        ax.plot(res.t, y * 100, color=color(i), lw=0.9, ls=linestyle(i), label=f"sample {i}")
        rows += [[f"sample{i}", float(res.rho_star), float(a), float(b)] for a, b in zip(res.t, y, strict=True)]
    ax.axvspan(s["window_lo_s"], s["window_hi_s"], color="0.85", lw=0, label="stop window")
    ax.axvline(s["stop_robust_s"], color="0.2", ls="--", lw=0.8, label=f"robust stop {s['stop_robust_s']:.0f} s")
    ax.set(
        xlabel="time (s)",
        ylabel=f"{product} yield (%)",
        title=f"(a) yield at ρ* = {res.rho_star:g}, Et {res.Et * 1e6:.3g} nM",
    )
    ax.legend(frameon=False, fontsize=5.5)
    rho = [r["rho"] for r in res.rows]
    bx.plot(
        rho, [r["yield_robust"] * 100 for r in res.rows], "o-", color=color(0), ms=3, label="worst sample, robust stop"
    )
    bx.plot(
        rho, [r["yield_nominal"] * 100 for r in res.rows], "s:", color=color(1), ms=3, label="sample 0, its own peak"
    )
    bx.set(xlabel="ρ (partner / analyte)", ylabel="yield (%)", title="(b) best yield against ρ")
    bx.legend(frameon=False, fontsize=5.5)
    return save(fig, out, "fig5a_peak", ["sample", "rho", "time_s", "yield"], rows)
