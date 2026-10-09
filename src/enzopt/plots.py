"""Standard figure set (spec area 13), one function per figure. Every figure = PDF + PNG + CSV of plotted values."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .figures import DOUBLE, SINGLE, color, marker, new_figure, save, table_figure


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
    pos = {}
    for row, comp in enumerate(lcs):
        for col, i in enumerate(comp):
            pos[i] = (col * 1.6, -row)
    edges = [tuple(e) for e in vd["edges"]]
    eset = set(edges)
    width = DOUBLE if max(len(c) for c in lcs) > 3 else SINGLE
    fig, ax = new_figure(width, 0.45 + 0.42 * len(lcs))
    from matplotlib.patches import FancyBboxPatch

    ax = ax[0, 0]
    for row, comp in enumerate(lcs):
        xs = [pos[i][0] for i in comp]
        ax.add_patch(
            FancyBboxPatch(
                (min(xs) - 0.6, -row - 0.28),
                max(xs) - min(xs) + 1.2,
                0.56,
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
        ax.annotate(
            "",
            xy=(x1 - 0.45 * np.sign(x1 - x0 or 1), y1 + sgn * off),
            xytext=(x0 + 0.45 * np.sign(x1 - x0 or 1), y0 + sgn * off),
            arrowprops=dict(
                arrowstyle="-|>",
                lw=1.6 if one_way else 0.6,
                color="#D55E00" if one_way else "0.25",
                shrinkA=0,
                shrinkB=0,
                mutation_scale=7,
            ),
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
    ax.set_ylim(-len(lcs) + 0.5, 0.5)
    ax.axis("off")
    wr = "weakly reversible" if vd["weakly_reversible"] else "NOT weakly reversible"
    ax.set_title(
        f"{vd['name']}: {len(labels)} complexes, {len(lcs)} linkage classes, deficiency {vd['deficiency']}, {wr}",
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
    out: Path, series: dict, temps: dict, roles: dict, name: str = "fig2_data", iso_range: float = 0.2
) -> dict:
    """(a) progress curves; (b) temperature per experiment: min–max bar and mean, with the isothermal band."""
    fig, axes = new_figure(DOUBLE, 0.36 * DOUBLE, ncols=2, gridspec_kw={"width_ratios": [1.6, 1]})
    ax, bx = axes[0]
    rows = []
    for i, (k, (t, y)) in enumerate(series.items()):
        ax.plot(t, y, marker(i), ms=2.2, mfc="none", mew=0.6, color=color(i), label=f"{k} ({roles[k]})")
        rows += [[k, "A340", float(a), float(b)] for a, b in zip(t, y, strict=True)]
    ax.set(xlabel="time (s)", ylabel="A340", title="(a) progress curves")
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
