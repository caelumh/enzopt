"""Stage ⑤ design: CRNT equilibrium limit + robust minimum enzyme over the identified samples, pipette table, and the
design-condition prediction curves (F-11: scipy over all samples + Hopsan for the representative sample)."""

from __future__ import annotations

import numpy as np

from .. import plots
from ..core.crnt import verdict
from ..core.design import AssayDesign, default_rho_grid, equilibrium_conversion, rho_min_equilibrium, run_design
from .common import design_rates, load_samples
from .runner import RunContext, StageFailed, write_json
from .sim import HOPSAN_DT_FACTOR
from .stages import load_series

FORWARD_ONLY_GRID = [1.5, 3.0, 5.0, 7.0, 9.0, 12.0, 16.0, 22.0, 30.0, 45.0, 70.0, 100.0]
FORWARD_ONLY_GRID_QUICK = [3.0, 8.0, 15.0, 30.0]


def run_design_stage(ctx: RunContext) -> None:
    p = ctx.project
    des = ctx.settings["design"]
    out = ctx.stage_dir()
    if not des.get("enabled", True):
        write_json(out / "design_summary.json", {"skipped": "design.enabled: false"})
        return
    if des.get("target", "assay") != "assay":
        raise StageFailed(
            f"design.target '{des.get('target')}'는 아직 지원하지 않음(assay). 반응기 설계는 F-14(③ 단계)"
        )
    fit = ctx.read_stage_json("3_fit", "fit_summary.json")
    if fit.get("design_blocked"):
        raise StageFailed(f"남은 표본 {len(fit.get('kept', []))}개 < 2 — 설계 중단(SC5-3). ③ 식별의 제외 사유 확인")
    samples = load_samples(ctx.run_dir)
    dnet = p.design_network()
    Kp = float(des["Kp"])
    rates = [design_rates(dnet, s, des, Kp, p.haldane_step) for s in samples]
    acts = [s.get("activity") for s in samples]
    if any(acts):
        from ..core.activity import horizon

        T = min(h for h in (horizon(a) for a in acts) if h)
        t_need = max(float(des["time_s"]), float(des.get("check_t_end_s", 600.0)))
        if t_need > T:
            ctx.warn(f"효소 활성 a(t)는 학습 실험 길이 {T:.0f} s까지만 근거가 있음: {t_need:.0f} s까지는 외삽")
    vd = verdict(dnet.reactions(), dnet.name)
    use_limit = vd.unique_equilibrium
    if not use_limit:
        why = "약가역 아님" if not vd.weakly_reversible else f"결핍 {vd.deficiency} ≠ 0"
        ctx.warn(f"설계 반응망 {why}: 평형 한계 대수식을 쓰지 않고 정방향 해석만으로 설계 범위를 탐색")
    design = AssayDesign(
        dnet,
        des["analyte"],
        des["partner"],
        des["free_enzyme"],
        float(des["analyte0_mM"]),
        float(des["conversion"]),
        float(des["time_s"]),
        Kp,
    )
    grid = des.get("rho_grid", "auto")
    if use_limit:
        rmin = rho_min_equilibrium(design.conversion, Kp)
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
    )
    summ = res.summary()
    train = p.by_role("train")
    E0 = train[0].Et if train else None
    summ.update(
        {
            "use_equilibrium_limit": use_limit,
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
    pip = pipette_table(des, res, E0, Kp, use_limit)
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
    write_json(out / "design_summary.json", summ)
    ctx.progress(0.95, "그림")
    figs = plots.design_figures(out, res, summ, check)
    write_json(out / "figures.json", figs)


def pipette_table(des: dict, res, E0: float | None, Kp: float, use_limit: bool) -> list[dict]:
    pp = des["pipette"]
    V = float(pp["volume_uL"])
    Ve = float(pp["enzyme_uL"])
    a0 = float(des["analyte0_mM"])
    sa = float(pp["stocks_mM"][des["analyte"]])
    sp = float(pp["stocks_mM"][des["partner"]])
    va = a0 * V / sa
    rows = []

    def row(name, purpose, rho, Et_mM, record, reps, pred):
        vp = rho * a0 * V / sp
        ve = Ve if Et_mM and Et_mM > 0 else 0.0
        buf = V - va - vp - ve
        rows.append(
            {
                "set": name,
                "purpose": purpose,
                "rho": rho,
                f"{des['analyte']}_stock_uL": va,
                f"{des['partner']}_stock_uL": vp,
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

    if use_limit and E0:
        pairs = zip(pp.get("limit_test_rhos", []), pp.get("limit_test_enzyme_multiples", []), strict=False)
        for i, (rho, mults) in enumerate(pairs):
            f = equilibrium_conversion(float(rho), Kp)
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
        f"표본 {len(rob)}개 최악 전환율 {min(c['conversion'] for c in rob):.4f}",
    )
    row(
        "D0",
        "공칭 설계",
        res.rho_star,
        res.Et_nominal_at_star,
        f"{des['time_s']:g} s 끝값 + 5분",
        3,
        f"표본 {len(nom)}개 중 미달 {sum(c['fail'] for c in nom)}개, "
        f"최악 전환율 {min(c['conversion'] for c in nom):.4f}",
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
