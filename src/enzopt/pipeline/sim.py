"""Stage ④ simulation + CRNT validation.

* scipy (Radau): every kept sample × (same-class random initial states + one other-class state), validated with the
  CRNT checks; a failing run is flagged "수치 오류 의심" and re-run once with the step / 10 (SC5-5).
* Hopsan (system level, I1-Hopsan): the apparent one-step reaction's class batch and one representative sample of
  the full mechanism with a small Δt, each compared with scipy. Without Hopsan the same runs use scipy and the run is
  marked "Hopsan 미사용" (SC5-4).
"""

from __future__ import annotations

import time

import numpy as np

from .. import plots
from ..core.massaction import MassAction, SolverSettings, simulate
from ..core.network import Network, Step, complex_str, x0_vector
from ..core.simval import fastest_rate, run_validated, scipy_solver
from ..core.validate import class_consistency, conserved_values, validate_closed
from .common import load_samples, observable
from .runner import RunContext, StageFailed, write_json

HOPSAN_NOT_USED = "Hopsan 미사용"
# explicit unit-delay scheme: Δt = factor / (largest Jacobian eigenvalue at the start); stays in the stable range
HOPSAN_DT_FACTOR = 0.2


# ------------------------------------------------------------------------------------------------ classes
def overall_step(p) -> Step | None:
    ov = p.thermodynamics.get("overall")
    return Step.parse(ov) if ov else None


def class_states(initial: dict, ov: Step, n: int, rng, margin: float = 1e-4) -> list[dict]:
    """Random states x = x_ref + ξ·ν(overall) with every overall species ≥ margin (same compatibility class)."""
    nu = {s: -c for s, c in ov.lhs.items()}
    for s, c in ov.rhs.items():
        nu[s] = nu.get(s, 0) + c
    lo, hi = -np.inf, np.inf
    for s, v in nu.items():
        x = initial.get(s, 0.0)
        if v > 0:
            lo = max(lo, (margin - x) / v)
        elif v < 0:
            hi = min(hi, (x - margin) / -v)
    if not (np.isfinite(lo) and np.isfinite(hi)) or hi <= lo:
        return []
    xs = rng.uniform(lo, hi, n)
    out = []
    for xi in xs:
        st = dict(initial)
        for s, v in nu.items():
            st[s] = initial.get(s, 0.0) + xi * v
        out.append(st)
    return out


def same_class(a: dict, b: dict, ov: Step, tol: float = 1e-9) -> bool:
    nu = {s: -c for s, c in ov.lhs.items()}
    for s, c in ov.rhs.items():
        nu[s] = nu.get(s, 0) + c
    keys = sorted(set(a) | set(b) | set(nu))
    d = np.array([b.get(k, 0.0) - a.get(k, 0.0) for k in keys])
    v = np.array([nu.get(k, 0.0) for k in keys], float)
    r = d - (d @ v) / (v @ v) * v
    return bool(np.max(np.abs(r)) < tol * max(1.0, np.max(np.abs(d))))


# ------------------------------------------------------------------------------------------------ stage
def run_sim_stage(ctx: RunContext) -> None:
    p = ctx.project
    sim = ctx.settings["simulation"]
    out = ctx.stage_dir()
    samples = load_samples(ctx.run_dir)
    if not samples:
        raise StageFailed("남은 표본이 없음(③ 식별 결과 확인)")
    net = p.network
    rxns = net.reactions()
    model = MassAction.from_network(net)
    ov = overall_step(p)
    if ov is None:
        raise StageFailed("network.thermodynamics.overall(전체 반응식)이 없어 적합류를 만들 수 없음")
    train = p.by_role("train")
    ref = train[0]
    rng = np.random.default_rng(int(sim.get("class_seed", 5)))
    n_same = int(sim["same_class_runs"])
    same_states = class_states(ref.initial, ov, n_same, rng)
    others = [e for e in p.experiments.values() if not same_class(ref.initial, e.initial, ov)]
    other = others[0] if others else None
    t_long = float(sim["t_long_s"])
    t = np.linspace(0.0, t_long, int(sim.get("n_points", 601)))
    coeff = p.observable_weights()
    rows, traj, finals_by_sample = [], [], {}
    n_total = len(samples) * (len(same_states) + (1 if other else 0))
    done = 0
    t0 = time.time()
    for s in samples:
        k = net.rate_vector(s["kf"], s["kr"])
        runs = [(f"same{i:02d}", st, ref.Et) for i, st in enumerate(same_states)]
        if other is not None:
            runs.append((f"other_{other.name}", other.initial, other.Et))
        finals, cons = [], []
        for name, init, Et in runs:
            ctx.progress(0.7 * done / max(1, n_total), f"표본 {s['seed']} · {name}")
            x0 = x0_vector(net, init, Et)
            vr = run_validated(name, scipy_solver(net, k, x0, t, Et, model=model), net.species, rxns, k, 1.0)
            row = vr.row()
            row.update(
                {
                    "sample": f"seed{s['seed']}",
                    "class": "other" if name.startswith("other") else "same",
                    "backend": "scipy",
                }
            )
            rows.append(row)
            if len(vr.X) == len(t):
                a = observable(net, vr.X, coeff)
                idx = net.index()
                for j in range(0, len(t), 4):
                    traj.append(
                        [
                            f"seed{s['seed']}",
                            name,
                            row["class"],
                            float(t[j]),
                            float(vr.X[j, idx[ov_analyte(ov, p.observable_weights())]]),
                            float(a[j]),
                        ]
                    )
                finals.append(vr.X[-1])
                cons.append(conserved_values(vr.X, rxns, net.species))
            done += 1
        cc = class_consistency(finals, cons) if finals else {}
        finals_by_sample[s["seed"]] = {
            "class_spread": [float(v) for v in cc.values()],
            "n_classes": len(cc),
            "final_analyte_same": float(
                np.mean([f[net.index()[ov_analyte(ov, p.observable_weights())]] for f in finals[: len(same_states)]])
            )
            if finals
            else None,
        }
    n_pass = sum(1 for r in rows if r["ok"])
    n_retry = sum(1 for r in rows if r["suspect_numerical_error"])
    if n_retry:
        ctx.warn(f"수치 오류 의심 {n_retry}건: Δt를 1/10로 줄여 다시 풀었음(검증 표 참조)")
    if n_pass < len(rows):
        msg = f"CRNT 검증 실패 {len(rows) - n_pass}/{len(rows)}"
        ctx.warn(msg + ("" if ctx.manifest["flags"].get("guarantee_149", True) else " — 149 보장 없음(멸종 방향 예상)"))
    plots.write_rows(out / "validation.csv", list(rows[0].keys()), [list(r.values()) for r in rows])
    plots.write_rows(
        out / "trajectories.csv",
        ["sample", "run", "class", "time_s", ov_analyte(ov, p.observable_weights()), "A340"],
        traj,
    )
    hop = hopsan_part(ctx, samples[0], same_states, other, ref, t_long)
    summary = {
        "backend_samples": "scipy (Radau)",
        "n_runs": len(rows),
        "n_pass": n_pass,
        "n_retried": n_retry,
        "reference_experiment": ref.name,
        "other_class_experiment": other.name if other else None,
        "same_class_runs": len(same_states),
        "t_long_s": t_long,
        "per_sample": finals_by_sample,
        "hopsan": hop,
        "seconds": time.time() - t0,
    }
    ctx.flag("validation", f"{n_pass}/{len(rows)}")
    ctx.progress(0.95, "그림")
    figs = plots.sim_figures(out, rows, traj, hop, ov_analyte(ov, p.observable_weights()), obs=p.observable["name"])
    write_json(out / "figures.json", figs)
    full = hop.get("full_representative", {})
    for key in [k for k in full if k in ("t", "hopsan_t") or k.startswith(("scipy_", "hopsan_A", "hopsan_an"))]:
        full.pop(key)
    write_json(out / "sim_summary.json", summary)


def ov_analyte(ov: Step, coeff: dict | None = None) -> str:
    """The observed substrate of the overall reaction (NADH for LDH): the left-hand species with the largest
    observable coefficient; without one, NADH if present, else the first species on the left."""
    seen = [s for s in ov.lhs if (coeff or {}).get(s)]
    if seen:
        return max(seen, key=lambda s: abs(coeff[s]))
    return next(iter(ov.lhs)) if "NADH" not in ov.lhs else "NADH"


def one_step_model(p) -> tuple[Network, list[dict], object]:
    """Apparent one-step network: each slow coordinate (overall reactions, then enzyme-free steps; ``core.qss``) as
    one mass-action step on the small molecules. Reversible with k₋ = k₁/K′ when the overall reaction has a K′."""
    from ..core.qss import structure

    qs = structure(p.network, p.overall)
    kp_of = {r["overall"]: r["Kp"] for r in p.reactions}
    steps, meta = [], []
    for k, label in enumerate(qs.labels):
        try:
            st = Step.parse(label)
        except ValueError:  # a completed basis vector: write it from V
            v = qs.V[:, k]
            lhs = {qs.small[i]: float(-v[i]) for i in range(len(v)) if v[i] < -1e-12}
            rhs = {qs.small[i]: float(v[i]) for i in range(len(v)) if v[i] > 1e-12}
            st = Step(lhs, rhs, False)
        Kp = kp_of.get(label)
        rev = st.reversible and Kp is not None
        steps.append(f"{complex_str(st.lhs)} {'<=>' if rev else '->'} {complex_str(st.rhs)}")
        meta.append({"label": label, "Kp": Kp if rev else None, "lhs": dict(st.lhs), "rhs": dict(st.rhs)})
    analyte = ov_analyte(overall_step(p), p.observable_weights())
    used = [x for x in qs.small if any(x in m["lhs"] or x in m["rhs"] for m in meta)]
    species = [analyte] + [x for x in used if x != analyte]
    return Network.from_strings("one_step", species, steps), meta, qs


# ------------------------------------------------------------------------------------------------ Hopsan
def hopsan_part(ctx: RunContext, sample: dict, same_states, other, ref, t_long: float) -> dict:
    p = ctx.project
    hcfg = ctx.settings["simulation"]["hopsan"]
    out = ctx.stage_dir() / "hopsan"
    out.mkdir(exist_ok=True)
    info: dict = {"used": False}
    if not hcfg.get("enabled", True):
        info["note"] = f"{HOPSAN_NOT_USED}(설정에서 끔{' · 짧은 모드' if ctx.quick else ''})"
    else:
        try:
            from ..hopsan.cli import find_cli

            info["cli"] = " ".join(find_cli())
            info["used"] = True
        except Exception as e:  # noqa: BLE001
            info["note"] = f"{HOPSAN_NOT_USED}(찾지 못함: {e})"
            ctx.warn(f"{HOPSAN_NOT_USED}: HopsanCLI를 찾지 못해 scipy로 대신함. 대회 결과 파일에는 Hopsan 결과가 필요")
    ctx.flag("hopsan_used", info["used"])
    if not info["used"]:
        ctx.flag("hopsan_note", info["note"])
    net = p.network
    ov = overall_step(p)
    analyte = ov_analyte(ov, p.observable_weights())
    # ---------------------------------------------------------------- apparent one-step class batch
    from ..pinn.ldh import matches_network

    one, meta, qs = one_step_model(p)
    one_name = "LDH_one_step" if matches_network(net) else "one_step"
    k_full = net.rate_vector(sample["kf"], sample["kr"])
    k1 = hcfg.get("one_step_k1", "auto")
    if k1 in (None, "auto"):
        k1s = apparent_k(net, k_full, ref.initial, ref.Et, meta, qs)
    else:
        k1s = [float(k1)] * len(meta)
    kr1 = [k / m["Kp"] if m["Kp"] else 0.0 for k, m in zip(k1s, meta, strict=True)]
    k_one = one.rate_vector(k1s, kr1)
    sub = [s for s in one.species]
    cases = [(f"same{i:02d}", {s: st.get(s, 0.0) for s in sub}) for i, st in enumerate(same_states)]
    if other is not None:
        cases.append((f"other_{other.name}", {s: other.initial.get(s, 0.0) for s in sub}))
    lam = 0.0
    for kf_, kr_, m in zip(k1s, kr1, meta, strict=True):
        lam += kf_ * sum(ref.initial.get(x, 0.0) for x in m["lhs"]) + kr_ * sum(
            ref.initial.get(x, 0.0) for x in m["rhs"]
        )
    dt = float(min(0.01 / max(lam, 1e-12), t_long / 1e5))
    info["one_step"] = {
        "k1": k1s[0],
        "Kp": meta[0]["Kp"],
        "steps": [str(st) for st in one.steps],
        "k1_all": k1s,
        "timestep": dt,
        "t_end": t_long,
        "runs": [],
    }
    one_model = MassAction.from_network(one)
    t_ref = np.linspace(0.0, t_long, 2048)
    max_diff = 0.0
    for i, (name, x0) in enumerate(cases):
        ctx.progress(0.72 + 0.1 * i / max(1, len(cases)), f"1단계 배치 {name}")
        x0v = np.array([x0[s] for s in one.species])
        ref_X = simulate(one_model, k_one, x0v, t_ref, atol=1e-12)
        if info["used"]:
            vr = run_validated(
                name,
                _hopsan_one_step_solver(out, name, one, k_one, x0, t_long, dt, one_name),
                one.species,
                one.reactions(),
                k_one,
                1.0,
            )
            if len(vr.t) > 1:
                d = np.max(np.abs(np.interp(t_ref, vr.t, vr.X[:, 0]) - ref_X[:, 0]))
                max_diff = max(max_diff, float(d))
            backend = "hopsan"
        else:
            S = {s: ref_X[:, j] for j, s in enumerate(one.species)}
            rep = validate_closed(t_ref, S, one.reactions(), k_one)
            vr = None
            backend = "scipy"
        rr = (
            vr.row()
            if vr is not None
            else {
                "run": name,
                "ok": rep["ok"],
                **{
                    k: rep.get(k)
                    for k in (
                        "conservation_max_rel_drift",
                        "min_concentration",
                        "equilibrium_max_log_error",
                        "lyapunov_max_increase",
                    )
                },
            }
        )
        rr.update({"backend": backend, f"final_{one.species[0]}_scipy": float(ref_X[-1, 0])})
        info["one_step"]["runs"].append(rr)
    # the class-batch model file is always written (submission / KIMM-SYS stand-in), run or not
    from ..hopsan.hmf import network_hmf

    (out / f"{one_name.lower()}.hmf").write_text(
        network_hmf(one_name, one, k_one, cases[0][1], t_long, dt, 2048), encoding="utf-8"
    )
    info["one_step"]["n_pass"] = sum(1 for r in info["one_step"]["runs"] if r["ok"])
    info["one_step"]["max_abs_diff_vs_scipy_mM"] = max_diff if info["used"] else None
    # ---------------------------------------------------------------- representative full mechanism
    rep_name = hcfg.get("representative_experiment") or ref.name
    e = p.experiments.get(rep_name, ref)
    x0 = x0_vector(net, e.initial, e.Et)
    model = MassAction.from_network(net)
    lam_max = fastest_rate(model, k_full, x0)
    dt_full = HOPSAN_DT_FACTOR / max(lam_max, 1e-12)
    t_end = float(hcfg.get("representative_t_end_s") or 900.0)
    max_steps = float(hcfg.get("max_steps", 6e7))
    note = None
    if t_end / dt_full > max_steps:
        t_end = max_steps * dt_full
        note = f"Δt {dt_full:.2e} s에서 단계 수 상한({max_steps:.0e}) 때문에 {t_end:.1f} s까지만 풂"
    tt = np.linspace(0.0, t_end, 2048)
    st = SolverSettings()
    from ..core.massaction import atol_vector

    Xs = simulate(model, k_full, x0, tt, atol=atol_vector(net.species, net.enzyme_species, e.Et, st), settings=st)
    full = {
        "experiment": e.name,
        "sample": f"seed{sample['seed']}",
        "lambda_max_per_s": lam_max,
        "timestep": dt_full,
        "t_end": t_end,
        "note": note,
        "t": tt.tolist(),
        "scipy_analyte": Xs[:, net.index()[analyte]].tolist(),
        "scipy_A340": observable(net, Xs, p.observable_weights()).tolist(),
    }
    if info["used"]:
        from ..hopsan.batch import full_mechanism_run

        ctx.progress(0.85, f"Hopsan 완전 메커니즘(Δt {dt_full:.1e} s, {t_end / dt_full:.1e} 단계)")
        t0 = time.time()
        th, sh, hmf = full_mechanism_run(
            out,
            "full_representative",
            net,
            k_full,
            {s: float(v) for s, v in zip(net.species, x0, strict=True)},
            t_end,
            dt_full,
        )
        Xh = np.column_stack([sh[s] for s in net.species])
        a_h = observable(net, Xh, p.observable_weights())
        full["hopsan_t"] = th.tolist()
        full["hopsan_analyte"] = sh[analyte].tolist()
        full["hopsan_A340"] = a_h.tolist()
        full["max_abs_diff_analyte_mM"] = float(
            np.max(np.abs(np.interp(tt, th, sh[analyte]) - Xs[:, net.index()[analyte]]))
        )
        full["max_abs_diff_A340"] = float(np.max(np.abs(np.interp(tt, th, a_h) - full_np(full["scipy_A340"]))))
        full["hopsan_seconds"] = time.time() - t0
        full["hmf"] = hmf.name
        S = {s: sh[s] for s in net.species}
        full["validation"] = {k: v for k, v in validate_closed(th, S, net.reactions(), k_full).items()}
    else:
        from ..hopsan.hmf import network_hmf

        x0d = {s: float(v) for s, v in zip(net.species, x0, strict=True)}
        (out / "full_representative.hmf").write_text(
            network_hmf("full_representative", net, k_full, x0d, t_end, dt_full, 2048), encoding="utf-8"
        )
        full["hmf"] = "full_representative.hmf"
    info["full_representative"] = full
    return info


def full_np(x):
    return np.asarray(x, float)


def apparent_k(net, k, initial: dict, Et: float, meta: list[dict], qs) -> list[float]:
    """Apparent one-step k₁ of every slow coordinate from the full model shortly after mixing: the small-molecule
    change over 1–2 s projected onto the slow basis (``core.qss``), divided by the mass-action drive."""
    model = MassAction.from_network(net)
    st = SolverSettings()
    from ..core.massaction import atol_vector

    tt = np.array([1.0, 2.0])
    X = simulate(
        model,
        k,
        x0_vector(net, initial, Et),
        tt,
        atol=atol_vector(net.species, net.enzyme_species, Et, st),
        settings=st,
    )
    i = net.index()
    dx = (X[1, qs.small_idx] - X[0, qs.small_idx]) / (tt[1] - tt[0])
    dxi = np.linalg.pinv(qs.V) @ dx
    xm = 0.5 * (X[0] + X[1])
    out = []
    for v, m in zip(dxi, meta, strict=True):
        drive = np.prod([xm[i[s]] ** c for s, c in m["lhs"].items()])
        if m["Kp"]:
            drive -= np.prod([xm[i[s]] ** c for s, c in m["rhs"].items()]) / m["Kp"]
        out.append(0.0 if abs(drive) < 1e-15 else float(v / drive))
    return out


def apparent_k1(net, k, initial: dict, Et: float, Kp: float) -> float:
    """Apparent one-step k₁ of Pyr + NADH ⇌ Lac + NAD from the full model's rate shortly after mixing (LDH)."""
    model = MassAction.from_network(net)
    st = SolverSettings()
    from ..core.massaction import atol_vector

    tt = np.array([1.0, 2.0])
    X = simulate(
        model,
        k,
        x0_vector(net, initial, Et),
        tt,
        atol=atol_vector(net.species, net.enzyme_species, Et, st),
        settings=st,
    )
    i = net.index()
    v = -(X[1, i["NADH"]] - X[0, i["NADH"]]) / (tt[1] - tt[0])
    xm = 0.5 * (X[0] + X[1])
    drive = xm[i["Pyr"]] * xm[i["NADH"]] - xm[i["Lac"]] * xm[i["NAD"]] / Kp
    if abs(drive) < 1e-15:
        return 0.0
    return float(v / drive)


def _hopsan_one_step_solver(out, name, one, k_one, x0, t_end, dt, model_name: str = "one_step"):
    from ..hopsan.cli import run_model
    from ..hopsan.hmf import network_hmf, write_params

    def solve(step: float):
        ts = dt * step
        hmf = out / f"one_step_{name}{'' if step == 1.0 else '_dt10'}.hmf"
        hmf.write_text(network_hmf(model_name, one, k_one, x0, t_end, ts, 2048), encoding="utf-8")
        pf = out / f"one_step_{name}.params.csv"
        write_params(pf, x0)
        th, sh = run_model(hmf, out / f"one_step_{name}{'' if step == 1.0 else '_dt10'}.csv", pf)
        return th, np.column_stack([sh[s] for s in one.species])

    return solve
