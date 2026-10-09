"""Stages ①–⑤. Each stage reads earlier stages only from files in the run folder (so cached/resumed stages work)."""

from __future__ import annotations

import numpy as np

from .. import plots
from ..core.crnt import NO_149_BANNER, verdict
from ..core.dataio import load_experiment, write_series
from .runner import RunContext, StageFailed, write_json

ISOTHERMAL_RANGE_C = 0.2  # methodology §6: sensor T(t) variation < 0.2 °C in ordinary sets


# ------------------------------------------------------------------------------------------------ ① verdict
def stage_verdict(ctx: RunContext) -> None:
    p = ctx.project
    out = ctx.stage_dir()
    nets = {"main": p.network}
    ctrl = p.control_network()
    if ctrl is not None:
        nets["control"] = ctrl
    if ctx.settings["design"].get("enabled", True) and ctx.settings["design"].get("dead_end"):
        nets["design"] = p.design_network()
    vds = {k: verdict(n.reactions(), n.name).to_dict() for k, n in nets.items()}
    for k, n in nets.items():
        vds[k]["steps"] = n.describe()
    main = vds["main"]
    write_json(out / "verdict.json", {"networks": vds, "main": "main"})
    lines = [f"[판정] {main['headline']}", main["guarantee_line"], ""]
    titles = {"main": "식별 반응망", "control": "비약가역 대조 반응망", "design": "설계 반응망(막다른 복합체 포함)"}
    for k, v in vds.items():
        lines.append(f"== {titles.get(k, k)}: {v['name']} ==")
        lines += [f"  {s}" for s in v["steps"]]
        lines.append(f"  {v['headline']}")
        lines.append(
            f"  복합체 {v['n_complexes']}, 연결 성분 {v['n_linkage_classes']}, 랭크 {v['rank']}, "
            f"결핍 {v['deficiency']}, "
            f"약가역 {'예' if v['weakly_reversible'] else '아니오'}, 보존 법칙 {len(v['conservation_laws'])}개"
        )
        lines += [f"    보존량 = {t}" for t in v["conservation_text"]]
        if v["irreversible_complexes"]:
            lines.append(f"  비가역이 걸린 복합체: {', '.join(v['irreversible_complexes'])}")
        lines.append("")
    lines.append(f"해석 백엔드: 표본 × 조건은 scipy(Radau). 시스템 수준: {hopsan_status(ctx)}")
    if not main["guarantee_149"]:
        lines.insert(0, f"⚠ {NO_149_BANNER} — 계속 진행하지만 외삽·설계에 정리 149의 보장이 없다.")
    (out / "report.txt").write_text("\n".join(lines), encoding="utf-8")
    ctx.flag("guarantee_149", bool(main["guarantee_149"]))
    ctx.flag("verdict_headline", main["headline"])
    ctx.flag("unique_equilibrium", bool(main["unique_equilibrium"]))
    if not main["guarantee_149"]:
        ctx.flag("banner", NO_149_BANNER)
        ctx.warn(f"{NO_149_BANNER} (반응망 {main['name']}이 약가역이 아님)")
    ctx.progress(0.5, "그림")
    figs = [plots.fig_network(out, main, "fig1a_network")]
    if "control" in vds:
        figs.append(plots.fig_network(out, vds["control"], "fig1c_network_control"))
    figs.append(plots.fig_verdict_table(out, vds))
    write_json(out / "figures.json", figs)


def hopsan_status(ctx: RunContext) -> str:
    """'Hopsan 사용 (path)' or 'Hopsan 미사용(reason)' — also recorded in the manifest flags."""
    if not ctx.settings["simulation"]["hopsan"].get("enabled", True):
        st = "Hopsan 미사용(설정에서 끔" + (" · 짧은 모드)" if ctx.quick else ")")
    else:
        try:
            from ..hopsan.cli import find_cli

            st = f"Hopsan 사용({' '.join(find_cli())})"
        except Exception:  # noqa: BLE001
            st = "Hopsan 미사용(HopsanCLI를 찾지 못함 → scipy로 대신)"
    ctx.flag("hopsan_status", st)
    return st


# ------------------------------------------------------------------------------------------------ ② data
def stage_data(ctx: RunContext) -> None:
    p = ctx.project
    out = ctx.stage_dir()
    coeff = p.observable_weights()
    summary, series, temps, roles = {}, {}, {}, {}
    names = list(p.experiments)
    for i, (name, exp) in enumerate(p.experiments.items()):
        ctx.progress(i / max(1, len(names)), f"실험 {name}")
        s, temp = load_experiment(exp, converted_dir=out)
        write_series(out / f"exp_{name}.csv", s.t, s.y, ("time_s", p.observable["name"]))
        a0_expected = sum(c * exp.initial.get(sp, 0.0) for sp, c in coeff.items())
        info = {
            "role": exp.role,
            "file": exp.file.name if exp.file else None,
            "initial": exp.initial,
            "Et_nM": exp.Et_nM,
            "n": int(len(s.t)),
            "t_first_s": float(s.t[0]),
            "t_last_s": float(s.t[-1]),
            "A_first": float(s.y[0]),
            "A_last": float(s.y[-1]),
            "A_min": float(s.y.min()),
            "A_max": float(s.y.max()),
            "A0_expected": float(a0_expected),
            "converted": bool(exp.format),
        }
        if temp is not None:
            write_series(out / f"temp_{name}.csv", temp.t, temp.y, ("time_s", "T_C"))
            rng = float(temp.y.max() - temp.y.min())
            info["temperature"] = {
                "mean_C": float(temp.y.mean()),
                "min_C": float(temp.y.min()),
                "max_C": float(temp.y.max()),
                "range_C": rng,
                "isothermal": rng < ISOTHERMAL_RANGE_C,
            }
            if rng >= ISOTHERMAL_RANGE_C:
                ctx.warn(f"실험 {name}: 온도 변동 {rng:.2f} ℃ ≥ {ISOTHERMAL_RANGE_C} ℃ (등온 전제 확인)")
            temps[name] = (temp.t, temp.y)
        else:
            temps[name] = None
        summary[name] = info
        series[name] = (s.t, s.y)
        roles[name] = exp.role
    if not any(v["role"] == "train" for v in summary.values()):
        raise StageFailed("학습 실험이 없음")
    write_json(out / "data.json", {"observable": p.observable, "experiments": summary})
    ctx.progress(0.9, "그림")
    figs = [plots.fig_data(out, series, temps, roles)]
    write_json(out / "figures.json", figs)


# ------------------------------------------------------------------------------------------------ ③–⑤
def stage_fit(ctx: RunContext) -> None:
    from .fit import run_fit_stage

    run_fit_stage(ctx)


def stage_sim(ctx: RunContext) -> None:
    from .sim import run_sim_stage

    run_sim_stage(ctx)


def stage_design(ctx: RunContext) -> None:
    from .design import run_design_stage

    run_design_stage(ctx)


def load_series(ctx: RunContext, name: str) -> tuple[np.ndarray, np.ndarray]:
    d = np.loadtxt(ctx.run_dir / "2_data" / f"exp_{name}.csv", delimiter=",", skiprows=1, ndmin=2)
    return d[:, 0], d[:, 1]
