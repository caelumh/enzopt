"""Stage ⑤ design: CRNT equilibrium limit + robust minimum enzyme over the identified samples, pipette table, and the
design-condition prediction curves (F-11: scipy over all samples + Hopsan for the representative sample)."""

from __future__ import annotations

import numpy as np

from .. import plots
from ..core.crnt import verdict
from ..core.design import (
    GOALS,
    AssayDesign,
    default_rho_grid,
    equilibrium_conversion_of,
    rho_min_of,
    run_design,
    run_peak_design,
    run_time_design,
)
from .common import design_rates, load_samples
from .runner import RunContext, StageFailed, write_json
from .sim import HOPSAN_DT_FACTOR
from .stages import load_series

FORWARD_ONLY_GRID = [1.5, 3.0, 5.0, 7.0, 9.0, 12.0, 16.0, 22.0, 30.0, 45.0, 70.0, 100.0]
FORWARD_ONLY_GRID_QUICK = [3.0, 8.0, 15.0, 30.0]


def rho_upper_bound(p, des: dict, design: AssayDesign) -> tuple[float | None, str | None]:
    """``design.rho_max``: None, a number, or "train" = the largest initial partner concentration of the training
    experiments ÷ analyte0, so that the design stays inside the conditions the model was fitted to."""
    v = des.get("rho_max")
    if v is None or not design.partner:
        return None, None
    if isinstance(v, str):
        if v != "train":
            raise StageFailed(f"design.rho_max '{v}'는 숫자 또는 train")
        c = [float(e.initial.get(design.partner, 0.0)) for e in p.by_role("train")]
        if not c or max(c) <= 0:
            raise StageFailed(f"design.rho_max: train — 학습 실험에 {design.partner} 처음 농도가 없음")
        return max(c) / design.analyte0, (
            f"학습 실험의 최대 {design.partner} {max(c):.4g} mM ÷ {design.analyte} {design.analyte0:g} mM"
        )
    return float(v), "설정값"


def run_design_stage(ctx: RunContext) -> None:
    p = ctx.project
    des = ctx.settings["design"]
    out = ctx.stage_dir()
    if not des.get("enabled", True):
        write_json(out / "design_summary.json", {"skipped": "design.enabled: false"})
        return
    goal = str(des.get("target", "assay"))
    if goal not in GOALS:
        raise StageFailed(f"design.target '{goal}'는 {', '.join(GOALS)} 중 하나. 연속 반응기 설계는 F-14")
    fit = ctx.read_stage_json("3_fit", "fit_summary.json")
    if fit.get("design_blocked"):
        raise StageFailed(f"남은 표본 {len(fit.get('kept', []))}개 < 2 — 설계 중단(SC5-3). ③ 식별의 제외 사유 확인")
    samples = load_samples(ctx.run_dir)
    dnet = p.design_network()
    ov_i = next((i for i, r in enumerate(p.reactions) if des["analyte"] in r["overall"].split()), None)
    ov = p.reactions[ov_i]["overall"] if ov_i is not None else None
    Kp = des.get("Kp")
    if Kp is None and ov_i is not None:
        Kp = p.reactions[ov_i]["Kp"]
    Kp = float(Kp) if Kp is not None else None
    rates = [design_rates(dnet, s, des, Kp, p.haldane_step, p.overall, ov_i or 0) for s in samples]
    acts = [s.get("activity") for s in samples]
    if any(acts):
        from ..core.activity import horizon

        T = min(h for h in (horizon(a) for a in acts) if h)
        t_need = max(float(des["time_s"]), float(des.get("check_t_end_s", 600.0)))
        if t_need > T:
            ctx.warn(f"효소 활성 a(t)는 학습 실험 길이 {T:.0f} s까지만 근거가 있음: {t_need:.0f} s까지는 외삽")
    vd = verdict(dnet.reactions(), dnet.name)
    why = limit_reason(vd, Kp, ov, len(p.reactions))
    use_limit = why is None
    # the equilibrium limit sets the automatic ρ grid and the conversion ceiling of assay goals; a yield or selectivity
    # goal on a given ρ grid does not use it, so its absence is recorded (design_summary) without a warning
    if not use_limit and (goal in ("assay", "time") or isinstance(des.get("rho_grid", "auto"), str)):
        ctx.warn(f"설계 반응망 {why}: 평형 한계를 쓰지 않고 정방향 해석만으로 설계 범위를 탐색")
    design = AssayDesign(
        dnet,
        des["analyte"],
        des.get("partner"),
        des["free_enzyme"],
        float(des["analyte0_mM"]),
        float(des["conversion"]),
        float(des["time_s"]),
        Kp if Kp is not None else 1.0,
        fixed={k: float(v) for k, v in (des.get("fixed_ratio") or {}).items()},
        goal=goal,
        product=des.get("product"),
        side_product=des.get("side_product"),
        yield_min=float(des.get("yield", 0.9)),
        selectivity_min=float(des.get("selectivity", 0.9)),
        overall=ov,
    )
    grid = des.get("rho_grid", "auto")
    if not design.partner:  # one substrate: only the enzyme amount is chosen
        rhos = [1.0]
    elif use_limit:
        rmin = rho_min_of(design)
        if rmin is None or not np.isfinite(rmin):
            raise StageFailed(f"어떤 ρ에서도 평형 전환율이 {design.conversion:g}에 닿지 않음(K' 확인)")
        rhos = (
            default_rho_grid(rmin, quick=(grid == "auto_quick")) if isinstance(grid, str) else [float(r) for r in grid]
        )
    else:
        rhos = (
            (FORWARD_ONLY_GRID_QUICK if grid == "auto_quick" else FORWARD_ONLY_GRID)
            if isinstance(grid, str)
            else [float(r) for r in grid]
        )
    lo, hi = (float(x) for x in des.get("Et_bounds_mM", [1e-6, 1e-2]))
    if goal == "time":
        _time_design(ctx, p, des, design, rates, rhos, acts, out)
        return
    if goal == "peak":
        _peak_design(ctx, p, des, design, rates, rhos, acts, out)
        return

    rho_max, rho_max_reason = rho_upper_bound(p, des, design)
    if rho_max is not None:
        rmin = rho_min_of(design) if use_limit else None
        if rmin is not None and rho_max <= rmin:
            raise StageFailed(f"ρ 상한 {rho_max:.4g}({rho_max_reason}) ≤ 평형 한계 {rmin:.4g}: 목표를 만족할 수 없음")
        if all(abs(r - rho_max) > 1e-9 * rho_max for r in rhos):
            rhos = sorted([*rhos, rho_max])

    def prog(f, m):
        ctx.progress(0.75 * f, f"U자 곡선 · {m}")

    res = run_design(
        design,
        rates,
        rhos,
        iters=int(des["bisection_iters"]),
        Et_bounds=(lo, hi),
        use_equilibrium_limit=use_limit,
        progress=prog,
        activities=acts,
        rho_max=rho_max,
    )
    summ = res.summary()
    summ["rho_max_reason"] = rho_max_reason
    train = p.by_role("train")
    E0 = train[0].Et if train else None
    summ.update(
        {
            "use_equilibrium_limit": use_limit,
            "equilibrium_limit_reason": why,
            "design_network": dnet.name,
            "Kp": Kp,
            "conversion": design.conversion,
            "time_s": design.time_s,
            "analyte0_mM": design.analyte0,
            "E0_mM": E0,
            "robust_Et_over_E0": (res.Et_star / E0) if E0 else None,
            "nominal_Et_over_E0": (res.Et_nominal_at_star / E0) if E0 else None,
            "samples": [s["seed"] for s in samples],
            "dead_end": des.get("dead_end"),
        }
    )
    plots.write_rows(
        out / "design_table.csv",
        ["rho", "Et_nominal_mM", "Et_robust_mM", "Et_nominal_nM", "Et_robust_nM"],
        [[r, a, b, a * 1e6, b * 1e6] for r, a, b in zip(res.rhos, res.Et_nominal, res.Et_robust, strict=True)],
    )
    plots.write_rows(
        out / "design_check.csv",
        ["design", "sample", "Et_mM", "NADH_at_target_mM", "conversion", "fail"],
        [
            [c["design"], samples[c["sample"]]["seed"], c["Et_mM"], c["NADH_at_target_mM"], c["conversion"], c["fail"]]
            for c in res.check
        ],
    )
    if res.equilibrium_limit:
        plots.write_rows(
            out / "equilibrium_limit.csv",
            ["rho", "conversion_eq", "NADH_left_uM"],
            [[r["rho"], r["conversion_eq"], r["NADH_left_uM"]] for r in res.equilibrium_limit],
        )
    summ["goal"] = goal
    pip = pipette_table(des, res, E0, design, use_limit)
    plots.write_rows(out / "pipette.csv", list(pip[0].keys()), [list(r.values()) for r in pip])
    (out / "pipette.txt").write_text(pipette_text(pip, summ), encoding="utf-8")
    summ["pipette"] = pip
    ctx.flag(
        "design",
        {"rho_star": res.rho_star, "Et_star_nM": res.Et_star * 1e6, "Et_star_over_E0": summ["robust_Et_over_E0"]},
    )
    # design-condition prediction curves (F-11)
    ctx.progress(0.8, "설계 조건 예측 곡선")
    check = design_check_curves(ctx, design, rates, samples, res, out)
    summ["design_check"] = {k: v for k, v in check.items() if k != "curves" and not k.startswith("_")}
    summ["analyte"], summ["partner"] = des["analyte"], des.get("partner")  # figure labels
    write_json(out / "design_summary.json", summ)
    ctx.progress(0.95, "그림")
    figs = plots.design_figures(out, res, summ, check)
    write_json(out / "figures.json", figs)


def limit_reason(vd, Kp: float | None, overall: str | None, n_reactions: int) -> str | None:
    """Why the CRNT equilibrium limit cannot be used for the design (None: it can)."""
    if not vd.weakly_reversible:
        return "약가역 아님"
    if vd.deficiency != 0:
        return f"결핍 {vd.deficiency} ≠ 0(평형 유일성 보장 없음)"
    if n_reactions > 1:
        return "전체 반응이 여럿(평형이 서로 얽힘)"
    if overall is None or Kp is None:
        return "분석 대상의 전체 반응이나 K'가 없음"
    return None


def pipette_table(des: dict, res, E0: float | None, design: AssayDesign, use_limit: bool) -> list[dict]:
    pp = des["pipette"]
    V = float(pp["volume_uL"])
    Ve = float(pp["enzyme_uL"])
    a0 = float(des["analyte0_mM"])
    stocks = pp["stocks_mM"]
    missing = [s for s in [design.analyte, design.partner, *design.fixed] if s and s not in stocks]
    if missing:
        raise StageFailed(f"design.pipette.stocks_mM에 저장액 농도가 없음: {', '.join(missing)}")
    va = a0 * V / float(stocks[design.analyte])
    fixed_uL = {s: float(m) * a0 * V / float(stocks[s]) for s, m in design.fixed.items()}
    rows = []
    goal = design.goal
    metric = {"assay": "전환율", "time": "전환율", "yield": "수율", "selectivity": "선택도"}[goal]
    mkey = {"assay": "conversion", "time": "conversion", "yield": "yield", "selectivity": "selectivity"}[goal]

    def row(name, purpose, rho, Et_mM, record, reps, pred):
        vp = rho * a0 * V / float(stocks[design.partner]) if design.partner else 0.0
        ve = Ve if Et_mM and Et_mM > 0 else 0.0
        buf = V - va - vp - sum(fixed_uL.values()) - ve
        part = {f"{design.partner}_stock_uL": vp} if design.partner else {}
        rows.append(
            {
                "set": name,
                "purpose": purpose,
                "rho": rho,
                f"{design.analyte}_stock_uL": va,
                **part,
                **{f"{s}_stock_uL": v for s, v in fixed_uL.items()},
                "buffer_uL": buf,
                "enzyme_uL": ve,
                "Et_final_nM": (Et_mM or 0.0) * 1e6,
                "Et_over_E0": (Et_mM / E0) if (E0 and Et_mM) else 0.0,
                "enzyme_working_uM": (Et_mM or 0.0) * 1e3 * V / Ve if ve else 0.0,
                "record": record,
                "repeats": reps,
                "predicted": pred,
            }
        )

    if use_limit and E0 and design.partner:
        lt = pp.get("limit_test_rhos", "auto")
        if lt in (None, "auto"):  # below ρ_min, where the equilibrium (not the enzyme) limits the conversion
            rmin = rho_min_of(design)
            lt = [float(f"{0.3 * rmin:.3g}"), float(f"{0.6 * rmin:.3g}")] if rmin and np.isfinite(rmin) else []
        pairs = zip(lt, pp.get("limit_test_enzyme_multiples", []), strict=False)
        for i, (rho, mults) in enumerate(pairs):
            f = equilibrium_conversion_of(design, float(rho))
            tag = f"L{i + 1}"
            for m in mults:
                row(
                    tag,
                    f"평형 한계 시험 ρ = {rho:g} (E₀×{m:g})",
                    float(rho),
                    E0 * float(m),
                    "평평해진 뒤 5분",
                    2,
                    f"평형 전환율 {f:.4f} (남는 {(1 - f) * a0 * 1e3:.2f} µM), 효소량과 무관",
                )
    nom = [c for c in res.check if c["design"] == "nominal"]
    rob = [c for c in res.check if c["design"] == "robust"]
    row(
        "D1",
        "강건 설계",
        res.rho_star,
        res.Et_star,
        f"{des['time_s']:g} s 끝값 + 5분",
        3,
        f"표본 {len(rob)}개 최악 {metric} {min(c.get(mkey, float('nan')) for c in rob):.4f}",
    )
    row(
        "D0",
        "공칭 설계",
        res.rho_star,
        res.Et_nominal_at_star,
        f"{des['time_s']:g} s 끝값 + 5분",
        3,
        f"표본 {len(nom)}개 중 미달 {sum(c['fail'] for c in nom)}개, "
        f"최악 {metric} {min(c.get(mkey, float('nan')) for c in nom):.4f}",
    )
    row("Z", "효소 없음(대조)", res.rho_star, 0.0, "10분", 1, "변화 없음")
    for r in rows:
        if r["buffer_uL"] < 0:
            r["predicted"] += " — 부피 초과(저장액 농도를 높일 것)"
    return rows


def pipette_text(rows: list[dict], summ: dict) -> str:
    lines = [
        f"피펫 표 (반응량 {sum(v for k, v in rows[0].items() if k.endswith('_uL')):.0f} µL, 효소는 마지막에 넣음)",
        f"강건 최적 ρ* = {summ['rho_star']:.3g}, Et* = {summ['robust_Et_star_nM']:.4g} nM"
        + (f" = E₀의 {summ['robust_Et_over_E0']:.3g}배" if summ.get("robust_Et_over_E0") else "")
        + f" · 공칭 Et = {summ['nominal_Et_at_rho_star_nM']:.4g} nM",
        *([f"ρ 상한 {summ['rho_max']:.4g}: {summ['rho_max_reason']}"] if summ.get("rho_max") else []),
        "",
    ]
    keys = list(rows[0].keys())
    lines.append(" | ".join(keys))
    for r in rows:
        lines.append(" | ".join(f"{v:.4g}" if isinstance(v, float) else str(v) for v in r.values()))
    return "\n".join(lines) + "\n"


def design_check_curves(ctx: RunContext, design: AssayDesign, rates, samples, res, out) -> dict:
    """Predicted analyte curves at the robust and nominal design conditions for every sample (scipy) and, when Hopsan is
    available, for the representative sample (full mechanism, small Δt); confirmation experiments as points."""
    p = ctx.project
    t_end = float(ctx.settings["design"].get("check_t_end_s", 600.0))
    t = np.linspace(0.0, t_end, 301)
    ia = design.network.index()[design.analyte]
    curves = {}
    for label, Et in (("robust", res.Et_star), ("nominal", res.Et_nominal_at_star)):
        for s, k in zip(samples, rates, strict=True):
            X = design.trajectory(k, Et, res.rho_star, t, s.get("activity"))
            curves[(label, f"seed{s['seed']}")] = (t, X[:, ia])
    info: dict = {
        "t_end": t_end,
        "rho_star": res.rho_star,
        "Et_robust_mM": res.Et_star,
        "Et_nominal_mM": res.Et_nominal_at_star,
    }
    # Hopsan: representative sample at the robust design
    hop = None
    if any(s.get("activity") for s in samples):
        info["hopsan"] = None
        info["hopsan_note"] = "Hopsan 모델에는 효소 실활 a(t)가 없어 설계 조건 대조를 생략"
    elif ctx.settings["simulation"]["hopsan"].get("enabled", True) and ctx.manifest["flags"].get("hopsan_used"):
        try:
            from ..core.simval import fastest_rate
            from ..hopsan.batch import full_mechanism_run

            k = rates[0]
            x0 = design.x0(res.Et_star, res.rho_star)
            lam = fastest_rate(design.model, k, x0)
            dt = HOPSAN_DT_FACTOR / max(lam, 1e-12)
            max_steps = float(ctx.settings["simulation"]["hopsan"].get("max_steps", 6e7))
            te = min(t_end, max_steps * dt)
            th, sh, hmf = full_mechanism_run(
                out / "hopsan",
                "design_robust_representative",
                design.network,
                k,
                {s: float(v) for s, v in zip(design.network.species, x0, strict=True)},
                te,
                dt,
            )
            ref = np.interp(th, t, curves[("robust", f"seed{samples[0]['seed']}")][1])
            hop = {
                "t": th,
                "x": sh[design.analyte],
                "timestep": dt,
                "t_end": te,
                "hmf": hmf.name,
                "max_abs_diff_mM": float(np.max(np.abs(sh[design.analyte] - ref))),
            }
            info["hopsan"] = {k: v for k, v in hop.items() if k not in ("t", "x")}
        except Exception as e:  # noqa: BLE001 — record and continue with scipy only
            info["hopsan_error"] = f"{type(e).__name__}: {e}"
            ctx.warn(f"설계 조건 Hopsan 해석 실패: {e}")
    else:
        info["hopsan"] = None
    # confirmation experiments (role design-check), converted to analyte via the observable coefficient
    conf = {}
    coeff = p.observable_weights()
    eps = coeff.get(design.analyte)
    for name, e in p.experiments.items():
        if e.role == "design-check" and eps:
            tt, y = load_series(ctx, name)
            conf[name] = {
                "t": tt,
                "x": y / eps,
                "Et_nM": e.Et_nM,
                "rho": e.initial.get(design.partner, 0) / design.analyte0,
            }
    info["confirmation"] = {k: {"Et_nM": v["Et_nM"], "rho": v["rho"]} for k, v in conf.items()}
    info["curves"] = curves
    info["_hopsan"] = hop
    info["_conf"] = conf
    return info


def _time_design(ctx: RunContext, p, des: dict, design: AssayDesign, rates, rhos, acts, out) -> None:
    """Goal ``time``: the enzyme amount fixed at ``Et_max_nM``; the ρ with the soonest worst-sample arrival."""
    if des.get("Et_max_nM") is None:
        raise StageFailed("design.target time: 효소량 상한 design.Et_max_nM이 필요함")
    Et = float(des["Et_max_nM"]) * 1e-6
    t_max = float(des.get("t_max_s", max(10 * float(des["time_s"]), 3600.0)))
    ctx.progress(0.2, "도달 시간 격자")
    res = run_time_design(design, rates, rhos, Et, t_max, activities=acts)
    summ = res.summary()
    train = p.by_role("train")
    E0 = train[0].Et if train else None
    summ.update({"conversion": design.conversion, "analyte0_mM": design.analyte0, "E0_mM": E0})
    plots.write_rows(
        out / "time_table.csv",
        ["rho", "time_nominal_s", "time_robust_s"],
        [[r, a, b] for r, a, b in zip(res.rhos, res.time_nominal, res.time_robust, strict=True)],
    )
    pp = des["pipette"]
    V, Ve, a0 = float(pp["volume_uL"]), float(pp["enzyme_uL"]), design.analyte0
    stocks = pp["stocks_mM"]
    va = a0 * V / float(stocks[design.analyte])
    vp = res.rho_star * a0 * V / float(stocks[design.partner]) if design.partner else 0.0
    vf = sum(float(m) * a0 * V / float(stocks[s_]) for s_, m in design.fixed.items())
    pip = [
        {
            "set": "D1",
            "purpose": "도달 시간 최소",
            "rho": res.rho_star,
            "buffer_uL": V - va - vp - vf - Ve,
            "Et_final_nM": Et * 1e6,
            "record": f"{res.time_star:.0f} s까지 + 5분",
            "predicted": f"표본 최악 도달 시간 {res.time_star:.0f} s",
        },
        {
            "set": "Z",
            "purpose": "효소 없음(대조)",
            "rho": res.rho_star,
            "buffer_uL": V - va - vp - vf,
            "Et_final_nM": 0.0,
            "record": "10분",
            "predicted": "변화 없음",
        },
    ]
    plots.write_rows(out / "pipette.csv", list(pip[0].keys()), [list(r.values()) for r in pip])
    summ["pipette"] = pip
    ctx.flag("design", {"rho_star": res.rho_star, "Et_nM": Et * 1e6, "robust_time_s": res.time_star})
    write_json(out / "design_summary.json", summ)


def _peak_design(ctx: RunContext, p, des: dict, design: AssayDesign, rates, rhos, acts, out) -> None:
    """Goal ``peak``: a product made and then broken down again (kinetically controlled synthesis). With the enzyme
    amount ``Et_nM`` (default: the first training experiment's), per ρ the stop time that maximises the worst
    sample's yield of ``product``; ρ* gives the highest worst-sample yield (``core.design.run_peak_design``)."""
    if not design.product:
        raise StageFailed("design.target peak: design.product(만들려는 생성물)가 필요함")
    train = p.by_role("train")
    Et = float(des["Et_nM"]) * 1e-6 if des.get("Et_nM") is not None else (train[0].Et if train else None)
    if Et is None:
        raise StageFailed("design.target peak: design.Et_nM(효소량)이 필요함")
    t_max = float(des.get("t_max_s", max(10 * float(des["time_s"]), 3600.0)))
    ctx.progress(0.2, "수율 곡선(ρ마다 표본마다)")
    res = run_peak_design(design, rates, rhos, Et, t_max, activities=acts)
    summ = res.summary()
    summ.update({"analyte0_mM": design.analyte0, "product": design.product, "samples": len(rates)})
    plots.write_rows(
        out / "peak_table.csv",
        ["rho", "peak_nominal_s", "yield_nominal", "stop_robust_s", "yield_robust", "window_lo_s", "window_hi_s"],
        [
            [r["rho"], r["peak_nominal_s"], r["yield_nominal"], r["stop_robust_s"], r["yield_robust"]]
            + [r["window_lo_s"], r["window_hi_s"]]
            for r in res.rows
        ],
    )
    s = res.star()
    pp = des["pipette"]
    V, Ve, a0 = float(pp["volume_uL"]), float(pp["enzyme_uL"]), design.analyte0
    stocks = pp["stocks_mM"]
    va = a0 * V / float(stocks[design.analyte])
    vp = res.rho_star * a0 * V / float(stocks[design.partner]) if design.partner else 0.0
    vf = sum(float(m) * a0 * V / float(stocks[s_]) for s_, m in design.fixed.items())
    pip = [
        {
            "set": "D1",
            "purpose": "최고 수율에서 멈춤(강건)",
            "rho": res.rho_star,
            "buffer_uL": V - va - vp - vf - Ve,
            "Et_final_nM": Et * 1e6,
            "record": (
                f"{s['window_lo_s']:.0f}~{s['window_hi_s']:.0f} s 사이에 멈춤(가장 좋은 {s['stop_robust_s']:.0f} s)"
            ),
            "predicted": f"표본 최악 수율 {s['yield_robust'] * 100:.0f} %",
        },
        {
            "set": "Z",
            "purpose": "효소 없음(대조)",
            "rho": res.rho_star,
            "buffer_uL": V - va - vp - vf,
            "Et_final_nM": 0.0,
            "record": f"{s['stop_robust_s']:.0f} s",
            "predicted": "변화 없음",
        },
    ]
    plots.write_rows(out / "pipette.csv", list(pip[0].keys()), [list(r.values()) for r in pip])
    summ["pipette"] = pip
    summ["figures"] = [plots.peak_figure(out, res, design.product)]
    write_json(out / "figures.json", summ["figures"])  # the dashboard lists a stage's figures from this file
    ctx.flag(
        "design",
        {"rho_star": res.rho_star, "Et_nM": Et * 1e6, "robust_stop_s": s["stop_robust_s"], "yield": s["yield_robust"]},
    )
    write_json(out / "design_summary.json", summ)
