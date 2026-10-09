"""Paper-quality figures: PDF (vector) + PNG (300 dpi) + a CSV with exactly the plotted values, same base name.

Style (spec area 13): journal column widths 85 mm / 170 mm, colour-blind-safe Okabe–Ito palette with line styles
and markers as a second cue, 3 significant digits, units in axis labels, English text.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from pathlib import Path

import numpy as np

MM = 1 / 25.4
SINGLE = 85 * MM
DOUBLE = 170 * MM
OKABE_ITO = ["#0072B2", "#E69F00", "#009E73", "#D55E00", "#CC79A7", "#56B4E9", "#F0E442", "#000000"]
MARKERS = ["o", "s", "^", "D", "v", "P", "X", "*"]
LINESTYLES = ["-", "--", "-.", ":", (0, (5, 1)), (0, (3, 1, 1, 1)), (0, (1, 1)), (0, (5, 2, 1, 2))]


def _mpl():
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update(
        {
            "font.size": 7,
            "axes.titlesize": 7.5,
            "axes.labelsize": 7,
            "legend.fontsize": 6,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.5,
            "axes.linewidth": 0.6,
            "lines.linewidth": 1.0,
            "lines.markersize": 3,
            "pdf.fonttype": 42,
            "savefig.bbox": "tight",
            "axes.prop_cycle": matplotlib.cycler(color=OKABE_ITO),
        }
    )
    return plt


def color(i: int) -> str:
    return OKABE_ITO[i % len(OKABE_ITO)]


def marker(i: int) -> str:
    return MARKERS[i % len(MARKERS)]


def linestyle(i: int):
    return LINESTYLES[i % len(LINESTYLES)]


def sig3(x) -> str:
    """3 significant digits; exponents as 1.5×10⁻⁴."""
    try:
        x = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(x):
        return "—"
    if x == 0:
        return "0"
    e = int(np.floor(np.log10(abs(x))))
    if -3 <= e <= 4:
        return f"{x:.{max(0, 2 - e)}f}"
    m = x / 10**e
    sup = str(e).translate(str.maketrans("-0123456789", "⁻⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{m:.2f}×10{sup}"


def new_figure(width: float = SINGLE, height: float | None = None, ncols: int = 1, nrows: int = 1, **kw):
    plt = _mpl()
    h = height if height is not None else width * 0.72 / max(1, ncols) * nrows
    kw.setdefault("layout", "constrained")
    return plt.subplots(nrows, ncols, figsize=(width, h), squeeze=False, **kw)


def write_csv(path: str | Path, header: Sequence[str], rows: Sequence[Sequence]) -> Path:
    path = Path(path)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, lineterminator="\n")
        w.writerow(header)
        for r in rows:
            w.writerow([_fmt(v) for v in r])
    return path


def _fmt(v):
    if isinstance(v, float | np.floating):
        return f"{float(v):.10g}"
    if isinstance(v, np.integer):
        return int(v)
    return v


def read_csv(path: str | Path) -> tuple[list[str], list[list[str]]]:
    with open(path, encoding="utf-8", newline="") as fh:
        rows = list(csv.reader(fh))
    return rows[0], rows[1:]


def save(fig, out_dir: str | Path, name: str, header: Sequence[str], rows: Sequence[Sequence]) -> dict:
    """Save ``name``.pdf, ``name``.png (300 dpi) and ``name``.csv in ``out_dir``; close the figure."""
    plt = _mpl()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / f"{name}.pdf")
    fig.savefig(out / f"{name}.png", dpi=300)
    plt.close(fig)
    write_csv(out / f"{name}.csv", header, rows)
    return {"name": name, "pdf": f"{name}.pdf", "png": f"{name}.png", "csv": f"{name}.csv"}


def table_figure(
    out_dir,
    name: str,
    header: Sequence[str],
    rows: Sequence[Sequence],
    title: str = "",
    width: float = DOUBLE,
    col_widths: Sequence[float] | None = None,
) -> dict:
    """A table rendered as a figure (values formatted to 3 significant digits) + the same table as CSV."""
    nrow = len(rows)
    fig, ax = new_figure(width, 0.35 + 0.17 * (nrow + 1))
    ax = ax[0, 0]
    ax.axis("off")
    cells = [[v if isinstance(v, str) else sig3(v) for v in r] for r in rows]
    tb = ax.table(cellText=cells, colLabels=list(header), loc="center", cellLoc="center", colWidths=col_widths)
    tb.auto_set_font_size(False)
    tb.set_fontsize(6)
    tb.scale(1, 1.15)
    for (r, _c), cell in tb.get_celld().items():
        cell.set_linewidth(0.4)
        if r == 0:
            cell.set_facecolor("#e8e8e8")
    if title:
        ax.set_title(title, loc="left")
    return save(fig, out_dir, name, header, rows)
