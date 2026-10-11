"""Experiment data reading and checks (F-2, SC5-6).

Standard formats (UTF-8 CSV with a header line):
  progress curve  ``time_s,A340``   (one row per observation)
  temperature     ``time_s,T_C``    (1 s interval)

A file with time going backwards, an empty value, a non-number or A340 > 2 is rejected with the file name
and the 1-based line number. Instrument exports in other layouts go through ``convert`` (configurable
generic converter) first; the converted series is written to the run folder, never over ``data/``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

A340_MAX = 2.0
TIME_UNITS = {"s": 1.0, "min": 60.0, "h": 3600.0, "ms": 1e-3}


class DataError(ValueError):
    """Bad experiment file; the message names the file and line."""


@dataclass
class Series:
    path: Path
    t: np.ndarray
    y: np.ndarray
    columns: tuple[str, str]


def _fail(path: Path, line: int | None, msg: str) -> DataError:
    where = f"{path.name}" + (f" {line}번째 줄" if line else "")
    return DataError(f"{where}: {msg} ({path})")


def read_series(path: str | Path, value_column: str = "A340", max_value: float | None = A340_MAX) -> Series:
    """Read a standard two-column CSV (``time_s,<value_column>``) and check it."""
    path = Path(path)
    if not path.exists():
        raise _fail(path, None, "파일이 없음")
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeDecodeError:
        raise _fail(path, None, "UTF-8로 읽을 수 없음(표준 형식은 UTF-8 CSV)") from None
    lines = text.splitlines()
    if not lines:
        raise _fail(path, None, "빈 파일")
    header = [h.strip() for h in lines[0].split(",")]
    lower = [h.lower() for h in header]
    if "time_s" not in lower or value_column.lower() not in lower:
        raise _fail(path, 1, f"머리글에 time_s와 {value_column} 열이 있어야 함(읽은 머리글: {lines[0].strip()})")
    it, iv = lower.index("time_s"), lower.index(value_column.lower())
    t, y = [], []
    for n, raw in enumerate(lines[1:], start=2):
        if not raw.strip():
            continue
        cells = [c.strip() for c in raw.split(",")]
        if len(cells) <= max(it, iv):
            raise _fail(path, n, "열 개수가 모자람(빈 값)")
        ct, cv = cells[it], cells[iv]
        if ct == "" or cv == "":
            raise _fail(path, n, "빈 값")
        try:
            vt, vv = float(ct), float(cv)
        except ValueError:
            raise _fail(path, n, f"숫자가 아님 ('{ct}', '{cv}')") from None
        if not (np.isfinite(vt) and np.isfinite(vv)):
            raise _fail(path, n, "숫자가 아님(NaN·무한대)")
        if t and vt <= t[-1]:
            raise _fail(path, n, f"시간 역행 또는 중복 ({t[-1]:g} s 다음 {vt:g} s)")
        if max_value is not None and vv > max_value:
            raise _fail(path, n, f"{value_column} = {vv:g} > {max_value:g} (분광기 선형 범위 밖)")
        t.append(vt)
        y.append(vv)
    if len(t) < 2:
        raise _fail(path, None, "데이터 줄이 2개 미만")
    return Series(path, np.array(t), np.array(y), ("time_s", value_column))


def read_temperature(path: str | Path) -> Series:
    s = read_series(path, value_column="T_C", max_value=None)
    bad = np.where((s.y < -5) | (s.y > 95))[0]
    if len(bad):
        raise _fail(s.path, None, f"온도 범위 밖({s.y[bad[0]]:g} ℃)")
    return s


def convert(path: str | Path, fmt: dict) -> tuple[np.ndarray, np.ndarray]:
    """Generic instrument-export converter.

    ``fmt`` keys: ``delimiter`` (default ","; "tab"/"\\t" for tab), ``decimal`` ("." or ","), ``skip_rows`` (lines
    before the first data row, default 1), ``time_col``/``value_col`` (0-based), ``time_unit`` (s|min|h|ms),
    ``encoding`` (default utf-8-sig; e.g. cp949 for Korean Windows exports), ``value_scale`` (default 1).
    Lines that do not parse as numbers after ``skip_rows`` (footers, blank lines) end the table.
    """
    path = Path(path)
    delim = fmt.get("delimiter", ",")
    delim = "\t" if delim in ("tab", "\\t") else delim
    dec = fmt.get("decimal", ".")
    skip = int(fmt.get("skip_rows", 1))
    tc, vc = int(fmt.get("time_col", 0)), int(fmt.get("value_col", 1))
    unit = TIME_UNITS[fmt.get("time_unit", "s")]
    scale = float(fmt.get("value_scale", 1.0))
    text = path.read_text(encoding=fmt.get("encoding", "utf-8-sig"))
    t, y = [], []
    for n, raw in enumerate(text.splitlines()[skip:], start=skip + 1):
        if not raw.strip():
            if t:
                break
            continue
        cells = [c.strip().strip('"') for c in (raw.split(delim) if delim != " " else raw.split())]
        try:
            vt = float(cells[tc].replace(dec, ".") if dec != "." else cells[tc])
            vv = float(cells[vc].replace(dec, ".") if dec != "." else cells[vc])
        except (ValueError, IndexError):
            if t:
                break
            raise _fail(path, n, "변환기: 숫자 행을 찾지 못함(skip_rows·delimiter 설정 확인)") from None
        t.append(vt * unit)
        y.append(vv * scale)
    if len(t) < 2:
        raise _fail(path, None, "변환기: 데이터 행이 2개 미만")
    return np.array(t), np.array(y)


def write_series(path: str | Path, t: np.ndarray, y: np.ndarray, columns=("time_s", "A340")) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(",".join(columns) + "\n")
        for a, b in zip(t, y, strict=True):
            fh.write(f"{a:.10g},{b:.10g}\n")


def _header(path: Path) -> list[str]:
    with open(path, encoding="utf-8-sig") as fh:
        return [h.strip().lower() for h in fh.readline().split(",")]


def load_experiment(exp, converted_dir: Path | None = None, value_column: str = "A340") -> tuple[Series, Series | None]:
    """Read (and if needed convert + check) one experiment's progress curve and temperature record.

    Temperature comes from ``temp_file``; without it, from a ``T_C`` column of the same standard CSV
    (``time_s,A340,T_C``) or from ``format.temp_col`` of a converted export. ``time_offset_s`` is added to all times.
    ``value_column`` is the observable's name (the CSV column); the A340 > 2 linear-range check applies only to A340.
    """
    vmax = A340_MAX if value_column.upper() == "A340" else None
    if exp.file is None:
        raise DataError(f"실험 {exp.name}: file 항목이 없음")
    if converted_dir is None and exp.format:  # data/ is read-only: never write next to the original
        import tempfile

        converted_dir = Path(tempfile.mkdtemp(prefix="enzopt_convert_"))
    temp = None
    if exp.format:
        t, y = convert(exp.file, exp.format)
        out = Path(converted_dir) / f"{exp.name}_converted.csv"
        write_series(out, t, y, ("time_s", value_column))
        series = read_series(out, value_column, vmax)
        if exp.temp_file is None and exp.format.get("temp_col") is not None:
            tt, yt = convert(exp.file, dict(exp.format, value_col=exp.format["temp_col"], value_scale=1.0))
            tout = Path(converted_dir) / f"{exp.name}_temp_converted.csv"
            write_series(tout, tt, yt, ("time_s", "T_C"))
            temp = read_temperature(tout)
    else:
        series = read_series(exp.file, value_column, vmax)
        if exp.temp_file is None and "t_c" in _header(exp.file):
            temp = read_temperature(exp.file)
    if exp.temp_file is not None:
        temp = read_temperature(exp.temp_file)
    off = float(getattr(exp, "time_offset_s", 0.0) or 0.0)
    if off:
        series.t = series.t + off
        if temp is not None:
            temp.t = temp.t + off
    return series, temp


def equilibrium_observable(project, initial: dict) -> float | None:
    """Observable at the overall-reaction equilibrium (K' from the project; enzyme-bound amounts neglected).

    For ``a A + b B <=> c C + d D`` the extent ξ solves K' = Π(products)/Π(reactants). Returns None when the
    project has no overall reaction or none of the observable species takes part in it.
    """
    from scipy.optimize import brentq

    from .network import Step

    ov = project.thermodynamics.get("overall")
    if not ov or project.Kp is None:  # no overall reaction, or one without K′ (driven by one-way steps)
        return None
    st = Step.parse(ov)
    lhs, rhs = st.lhs, st.rhs
    x0 = {s: float(initial.get(s, 0.0)) for s in list(lhs) + list(rhs)}
    coeff = project.observable_weights()
    if not any(s in x0 for s in coeff):
        return None
    hi = min(x0[s] / c for s, c in lhs.items())
    lo = -min(x0[s] / c for s, c in rhs.items())
    ln_k = float(np.log(project.Kp))

    def f(xi):
        num = sum(c * np.log(max(x0[s] + c * xi, 1e-300)) for s, c in rhs.items())
        den = sum(c * np.log(max(x0[s] - c * xi, 1e-300)) for s, c in lhs.items())
        return num - den - ln_k

    eps = 1e-12 * max(1.0, hi - lo)
    xi = 0.0 if hi - lo <= 2 * eps else brentq(f, lo + eps, hi - eps, xtol=1e-15)
    conc = dict(x0)
    for s, c in lhs.items():
        conc[s] = x0[s] - c * xi
    for s, c in rhs.items():
        conc[s] = x0[s] + c * xi
    return float(sum(w * conc.get(s, 0.0) for s, w in coeff.items()))


def final_slope(path, window_s: float = 60.0) -> float:
    """Least-squares slope (per s) over the last ``window_s`` seconds of a standard series (≈ 0 at equilibrium)."""
    d = np.loadtxt(path, delimiter=",", skiprows=1, ndmin=2)
    t, y = d[:, 0], d[:, 1]
    m = t >= t[-1] - window_s
    if m.sum() < 3:
        m = np.arange(len(t)) >= len(t) - 3
    return float(np.polyfit(t[m], y[m], 1)[0])
