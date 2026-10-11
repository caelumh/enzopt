"""Missing-step suggestions for any network: candidate steps from the network's structure, compared by least squares.

``enzopt suggest`` keeps the PINN path for the LDH ordered bi-bi network (``pipeline.suggest``). Every other network
(and LDH with ``--method ode``) comes here:

1. **Candidates** from the structure, no list to write: a dead-end complex X + Y ⇌ X_Y for every active enzyme form X
   and small molecule Y that X does not bind yet; an inactivation step X → ED (one dead form, ``core.qss``); and each
   overall reaction without the enzyme (one-way, or reversible under its K′).
2. **Fit** the network as written and the network plus each candidate by least squares on the full ODE: every signal
   in units of its σ, the stage-③ prior, the new constants around generic centres. Warm start: the classical fit of
   the latest completed run of the project with the same network (``--run`` to choose), else a fresh multi-start fit.
3. **Rank and decide** with the rule of the LDH path: the network explains the training data when its RMSE is within
   2σ and every training set within 3σ; otherwise the candidate with the lowest training RMSE is suggested when it at
   least halves the RMSE over the missed sets without raising the training RMSE. ΔAIC and the held-out RMSE stand
   next to it (Q-02): a step that only fits the training sets shows up there, and so does one that only the held-out
   sets ask for.
"""

from __future__ import annotations

import json
import math
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from ..core.massaction import MassAction, SimulationError, atol_vector, simulate
from ..core.network import Network, Step, complex_str, x0_vector
from ..core.thermo import layout

KINDS = ("dead_end", "inactivation", "background")
KF_BIND_LOG10 = 3.0  # 1/(mM·s): binding of a candidate dead-end complex (as KF_DEAD_END of the LDH path)
KI_LOG10 = math.log10(5.0)  # mM: dissociation constant at the prior centre (as the LDH path's default K_i)
K_INACT_LOG10 = -2.0  # 1/s: inactivation of an enzyme form
K_BACKGROUND_LOG10 = -4.0  # 1/s (1/(mM·s) for two reactants): the reaction without the enzyme
BASE_OK_SIGMA = 2.0  # as pipeline.suggest: the network explains the training data within 2σ (every set within 3σ)
SET_OK_SIGMA = 3.0


@dataclass
class Candidate:
    name: str
    kind: str
    step: str
    new_enzyme: list = field(default_factory=list)  # new enzyme species (dead-end complex, dead form)
    prior_kf: float = 0.0  # log10 centre of the new step's k₊
    prior_kr: float | None = None  # log10 centre of its k₋ (reversible steps)


def dead_forms(net: Network) -> list[str]:
    """Enzyme forms that no step leaves (as ``core.qss``)."""
    ens = list(net.enzyme_species)
    leaves = set()
    for st in net.steps:
        leaves.update(s for s in st.lhs if s in ens)
        if st.reversible:
            leaves.update(s for s in st.rhs if s in ens)
    return [s for s in ens[1:] if s not in leaves]


def _fresh(name: str, taken: set) -> str:
    out, k = name, 2
    while out in taken:
        out, k = f"{name}{k}", k + 1
    taken.add(out)
    return out


def candidates(p, kinds=KINDS) -> list[Candidate]:
    """Candidate steps for the project's network (see the module docstring)."""
    net = p.network
    ens = list(net.enzyme_species)
    dead = dead_forms(net)
    active = [x for x in ens if x not in dead]
    small = [s for s in net.species if s not in ens]
    taken = set(net.species)
    out: list[Candidate] = []
    if "dead_end" in kinds:
        binds = set()
        for st in net.steps:
            for side in [st.lhs] + ([st.rhs] if st.reversible else []):
                es = [s for s in side if s in ens]
                sm = [s for s in side if s not in ens]
                if len(es) == 1 and len(sm) == 1 and side[es[0]] == 1 and side[sm[0]] == 1:
                    binds.add((es[0], sm[0]))
        for x in active:
            for y in small:
                if (x, y) in binds:
                    continue
                c = _fresh(f"{x}_{y}", taken)
                out.append(
                    Candidate(
                        f"{x}·{y}", "dead_end", f"{x} + {y} <=> {c}", [c], KF_BIND_LOG10, KF_BIND_LOG10 + KI_LOG10
                    )
                )
    if "inactivation" in kinds:
        d = dead[0] if dead else _fresh("ED", taken)
        has = {next(iter(st.lhs)) for st in net.steps if list(st.rhs) == [d] and len(st.lhs) == 1}
        for x in active:
            if x not in has:
                out.append(Candidate(f"{x} → {d}", "inactivation", f"{x} -> {d}", [d], K_INACT_LOG10, None))
    if "background" in kinds:
        free = {
            (tuple(sorted(st.lhs.items())), tuple(sorted(st.rhs.items())))
            for st in net.steps
            if not any(s in ens for s in [*st.lhs, *st.rhs])
        }
        for r in p.reactions:
            st = Step.parse(r["overall"])
            if (tuple(sorted(st.lhs.items())), tuple(sorted(st.rhs.items()))) in free:
                continue
            kp = r.get("Kp")
            step = f"{complex_str(st.lhs)} {'<=>' if kp else '->'} {complex_str(st.rhs)}"
            kr = K_BACKGROUND_LOG10 - math.log10(kp) if kp else None
            out.append(Candidate(f"no enzyme: {step}", "background", step, [], K_BACKGROUND_LOG10, kr))
    return out


# ------------------------------------------------------------------------------------------------ fitting
_MODELS: dict = {}


def _model(net: Network) -> MassAction:
    key = (tuple(net.species), tuple(str(s) for s in net.steps))
    if key not in _MODELS:
        _MODELS[key] = MassAction.from_network(net)
    return _MODELS[key]


def _z(net: Network, k, exp) -> list[np.ndarray]:
    """σ-normalised residuals of every signal of one experiment (one ODE solve on the union of the time grids)."""
    from .compare import FIT_SETTINGS

    _name, initial, Et, series = exp
    t_u = np.unique(np.concatenate([s[1] for s in series]))
    atol = atol_vector(net.species, net.enzyme_species, Et, FIT_SETTINGS)
    X = simulate(_model(net), k, x0_vector(net, initial, Et), t_u, atol=atol, settings=FIT_SETTINGS)
    idx = net.index()
    out = []
    for _sig, t, y, sg, cf in series:
        rows = np.searchsorted(t_u, t)
        a = sum(c * X[rows, idx[s]] for s, c in cf.items())
        out.append((a - y) / sg)
    return out


def _residuals(theta, net, lay, exps, prior, wp):
    kf, kr = lay.rates(theta)
    k = net.rate_vector(kf, kr)
    res = []
    for ex in exps:
        try:
            z = _z(net, k, ex)
        except (SimulationError, ValueError, FloatingPointError):
            z = None
        for j, (_sig, t, _y, _sg, _cf) in enumerate(ex[3]):
            w = 1.0 / np.sqrt(len(t) * len(exps) * len(ex[3]))
            r = np.full(len(t), 1e3) if z is None else np.nan_to_num(z[j], nan=1e3, posinf=1e3, neginf=-1e3)
            res.append(r * w)
    res.append(wp * (np.asarray(theta) - prior))
    return np.concatenate(res)


def _fit(job) -> dict:
    key, net, lay, theta0, prior, wp, exps, max_nfev = job
    t0 = time.time()
    try:
        sol = least_squares(
            _residuals, theta0, args=(net, lay, exps, prior, wp), method="trf", max_nfev=max_nfev, diff_step=1e-4
        )
    except Exception as e:  # noqa: BLE001 — a crashed fit is reported, not fatal
        return {"key": key, "error": f"{type(e).__name__}: {e}", "seconds": time.time() - t0}
    return {
        "key": key,
        "theta": sol.x.tolist(),
        "cost": float(sol.cost),
        "nfev": int(sol.nfev),
        "seconds": time.time() - t0,
    }


def evaluate(net: Network, lay, theta, exps) -> dict:
    """Per experiment: RMSE ÷ σ of every signal and their root mean square; SSR = Σ z² over all points."""
    kf, kr = lay.rates(theta)
    k = net.rate_vector(kf, kr)
    per, ssr, n = {}, {}, {}
    for ex in exps:
        try:
            z = _z(net, k, ex)
        except (SimulationError, ValueError, FloatingPointError):
            per[ex[0]], ssr[ex[0]], n[ex[0]] = {"__all__": float("inf")}, float("inf"), 0
            continue
        sig = {s[0]: float(np.sqrt(np.mean(zz**2))) for s, zz in zip(ex[3], z, strict=True)}
        sig["__all__"] = float(np.sqrt(np.mean([v**2 for v in sig.values()])))
        per[ex[0]] = sig
        ssr[ex[0]] = float(sum(np.sum(zz**2) for zz in z))
        n[ex[0]] = int(sum(len(zz) for zz in z))
    return {"per": per, "ssr": ssr, "n": n}


def _rms(per: dict, names: list[str]) -> float | None:
    v = [per[x]["__all__"] for x in names]
    return float(np.sqrt(np.mean(np.square(v)))) if v else None


# ------------------------------------------------------------------------------------------------ priors, start
def _kp_full(p, Kp_used: list | None, lay_base) -> list[float]:
    """K′ of every overall reaction: the base fit's values where it identified them, else the project's."""
    full = [float(r["Kp"]) if r.get("Kp") else 1.0 for r in p.reactions]
    for j, o in enumerate(lay_base.overall_used):
        if Kp_used and j < len(Kp_used) and Kp_used[j]:
            full[o] = float(Kp_used[j])
    return full


def setup(p, cand: Candidate | None):
    """(network, θ layout, prior centre θ) of the network as written (cand None) or plus one candidate step."""
    n0 = len(p.network.steps)
    prefer = [p.haldane_step] if p.haldane_step else None
    if cand is None:
        net = p.network
    else:
        net = p.network.with_extra_steps(
            [cand.step], name=f"{p.network.name}+{cand.kind}", extra_species=cand.new_enzyme
        )
        if cand.kind == "background" and Step.parse(cand.step).reversible:
            prefer = [n0 + 1]  # the new step's k₋ follows from K′
    lay = layout(net, p.overall, prefer)
    pv = p.prior_vector()
    base_lay = p.layout
    kf = [float(x) for x in pv[:n0]] + ([cand.prior_kf] if cand else [])
    kr_all = list(p.kr_prior_all()) + ([cand.prior_kr if cand.prior_kr is not None else np.nan] if cand else [])
    kr = [float(kr_all[i]) if np.isfinite(kr_all[i]) else 3.0 for i in lay.free_kr]
    kp = []
    for o in lay.overall_used:
        if o in base_lay.overall_used:
            kp.append(float(pv[n0 + len(base_lay.free_kr) + base_lay.overall_used.index(o)]))
        else:
            kp.append(math.log10(float(p.reactions[o]["Kp"])))
    return net, lay, np.array(kf + kr + kp, float)


def start_theta(p, lay, base: dict, cand: Candidate | None) -> np.ndarray:
    """Warm start: the base fit's constants plus the candidate's prior centre."""
    kf = list(base["kf"]) + ([10.0**cand.prior_kf] if cand else [])
    kr = list(base["kr"])
    if cand:
        if cand.prior_kr is not None:
            kr.append(10.0**cand.prior_kr)
        else:
            kr.append(0.0)
    return lay.theta(kf, kr, _kp_full(p, base.get("Kp_all"), p.layout))


def base_from_run(p, run: Path) -> dict | None:
    """The best classical fit of a completed run whose network equals the project's (None if it does not match)."""
    from ..core.project import load_project

    f = run / "3_fit" / "fit_summary.json"
    if not f.exists() or not (run / "inputs" / "project.yaml").exists():
        return None
    try:
        q = load_project(run / "inputs")
    except Exception:  # noqa: BLE001 — an old or broken run is skipped
        return None
    if [str(s) for s in q.network.steps] != [str(s) for s in p.network.steps] or q.network.species != p.network.species:
        return None
    fits = (json.loads(f.read_text(encoding="utf-8")).get("comparisons") or {}).get("classical", {}).get("fits") or []
    fits = [x for x in fits if "kf" in x]
    if not fits:
        return None
    best = min(fits, key=lambda x: x["cost"])
    return {"kf": best["kf"], "kr": best["kr"], "Kp_all": best.get("Kp_all"), "source": str(run)}


def latest_base(p, root: Path) -> dict | None:
    d = root / p.name
    if not d.exists():
        return None
    for run in sorted((x for x in d.iterdir() if x.is_dir() and not x.name.endswith("_suggest")), reverse=True):
        m = run / "manifest.json"
        if m.exists() and json.loads(m.read_text(encoding="utf-8")).get("status") == "completed":
            b = base_from_run(p, run)
            if b:
                return b
    return None


# ------------------------------------------------------------------------------------------------ decide, report
def decide(base: dict, rows: list[dict], train: list[str], hold: list[str]) -> dict:
    """The LDH path's rule on σ-normalised RMSE (module docstring); held-out RMSE and ΔAIC shown with it."""
    per = {n: base["per"][n]["__all__"] for n in train}
    misses = [n for n in train if per[n] > SET_OK_SIGMA]
    base_ok = base["train"] <= BASE_OK_SIGMA and not misses
    ok_rows = [r for r in rows if r.get("train") is not None and np.isfinite(r["train"])]
    top = min(ok_rows, key=lambda r: r["train"]) if ok_rows else None
    better = False
    if top:
        sets = misses or train
        better = _rms(top["per"], sets) <= 0.5 * _rms(base["per"], sets) and top["train"] <= base["train"]
    suggest = (not base_ok) and better
    best_ho = min((r for r in ok_rows if r.get("holdout") is not None), key=lambda r: r["holdout"], default=None)
    txt = f"학습 RMSE {base['train']:.2f}σ, 최악 {max(per, key=per.get)} {max(per.values()):.2f}σ"
    if base_ok:
        note = f"적어 둔 반응식으로 학습 데이터가 설명됨({txt})"
    elif suggest:
        note = f"적어 둔 반응식으로는 설명되지 않음({txt}). {top['step']}을 넣으면 학습 RMSE {top['train']:.2f}σ"
    else:
        note = f"적어 둔 반응식으로는 설명되지 않음({txt}). 후보 중 RMSE를 절반으로 줄이는 단계 없음"
    ho_note = None
    if hold and best_ho is not None and base.get("holdout") is not None:
        ho_note = (
            f"홀드아웃 기준: 가장 크게 줄이는 후보 {best_ho['step']} "
            f"(홀드아웃 RMSE {base['holdout']:.2f}σ → {best_ho['holdout']:.2f}σ, ΔAIC {best_ho['d_aic']:+.1f})"
        )
    return {
        "base_explains_data": bool(base_ok),
        "sets_missed_by_base": misses,
        "suggested_step": top["step"] if suggest else None,
        "suggested_holdout_improves": (
            bool(top["holdout"] < base["holdout"]) if suggest and top.get("holdout") is not None else None
        ),
        "best_holdout_step": best_ho["step"] if best_ho else None,
        "note": note,
        "holdout_note": ho_note,
    }


def _figure(out: Path, base: dict, rows: list[dict], n_top: int = 15) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    ok = [r for r in rows if r.get("train") is not None]
    pick = sorted(ok, key=lambda r: r["d_aic"])[:n_top]
    pick += sorted([r for r in ok if r.get("holdout") is not None], key=lambda r: r["holdout"])[:3]  # best held out
    top = sorted({id(r): r for r in pick}.values(), key=lambda r: r["d_aic"])
    if not top:
        return
    y = np.arange(len(top))[::-1]
    fig, ax = plt.subplots(figsize=(8.0, 0.35 * len(top) + 1.6))
    ax.barh(y + 0.18, [r["train"] for r in top], height=0.36, color="#0072B2", label="training")
    if any(r.get("holdout") is not None for r in top):
        ax.barh(y - 0.18, [r.get("holdout") or 0 for r in top], height=0.36, color="#E69F00", label="held out")
        ax.axvline(base["holdout"], color="#E69F00", ls="--", lw=1, label="held out, as written")
    ax.axvline(base["train"], color="#0072B2", ls="--", lw=1, label="training, as written")
    ax.set_yticks(y)
    ax.set_yticklabels([f"{r['step']}  (ΔAIC {r['d_aic']:+.0f})" for r in top], fontsize=8)
    ax.set_xlabel("RMSE / σ (all signals)")
    ax.set_title("candidate steps, fitted by least squares (best by ΔAIC and by held-out RMSE)", fontsize=10)
    ax.legend(fontsize=8, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.12), ncol=4)
    fig.tight_layout()
    fig.savefig(out / "candidates.png", dpi=120)
    plt.close(fig)


# ------------------------------------------------------------------------------------------------ command
def load_all(p, tmp: Path) -> list[tuple]:
    """(name, initial, Et, [(signal, t, y, σ, coefficients)]) of every training and held-out experiment, + role."""
    from ..core.dataio import load_experiment
    from .stages import _read_signal

    obs = p.observable["name"]
    sigma = float(p.identification["noise_sigma"])
    out = []
    for name, e in p.experiments.items():
        if e.role not in ("train", "holdout"):
            continue
        s, _ = load_experiment(e, converted_dir=tmp, value_column=obs)
        series = [(obs, np.asarray(s.t, float), np.asarray(s.y, float), sigma, dict(p.observable_weights()))]
        for sg in p.signals:
            ss = _read_signal(e, sg["name"])
            series.append(
                (
                    sg["name"],
                    np.asarray(ss.t, float),
                    np.asarray(ss.y, float),
                    float(sg["sigma"]),
                    dict(sg["coefficients"]),
                )
            )
        out.append(((name, e.initial, e.Et, series), e.role))
    return out


def run(p, out: Path, quick: bool, kinds=KINDS, run_dir: Path | None = None, root: Path | None = None, log=print):
    import tempfile

    from .. import plots
    from ..core.design import n_workers

    ident = p.identification
    sd = float(p.prior.get("sd_decades", 1.5))
    wp = float(np.sqrt(float(ident["pinn"].get("w_prior", 1e-2)) / (2 * sd**2)))
    data = load_all(p, Path(tempfile.mkdtemp(prefix="enzopt_suggest_")))
    exps = [x for x, _ in data]
    train = [x for x, r in data if r == "train"]
    tr_names = [x[0] for x in train]
    ho_names = [x[0] for x, r in data if r == "holdout"]
    cands = candidates(p, kinds)
    log(f"후보 {len(cands)}개: " + ", ".join(f"{k} {sum(c.kind == k for c in cands)}" for k in kinds))

    bf = ident.get("baseline_fit") or {}
    nfev_c = 15 if quick else 100
    base = None
    if run_dir is not None:
        base = base_from_run(p, Path(run_dir))
        if base is None:
            raise SystemExit(f"{run_dir}: 반응망이 같은 완료된 실행의 고전 피팅이 없음")
    elif root is not None:
        base = latest_base(p, root)
    net0, lay0, prior0 = setup(p, None)
    if base is None:
        starts, nfev = (1, 20) if quick else (int(bf.get("starts", 4)), int(bf.get("max_nfev", 200)))
        log(f"출발점: 적어 둔 반응식을 새로 맞춤(시작점 {starts} × 최대 {nfev}회)")
        jobs = [
            (
                s,
                net0,
                lay0,
                prior0 + np.random.default_rng(100 + s).uniform(-1, 1, len(prior0)),
                prior0,
                wp,
                train,
                nfev,
            )
            for s in range(starts)
        ]
        with ProcessPoolExecutor(max_workers=n_workers(len(jobs))) as ex:
            fits = [f for f in ex.map(_fit, jobs) if "theta" in f]
        if not fits:
            raise SystemExit("적어 둔 반응식의 고전 피팅이 모든 시작점에서 실패")
        kf, kr = lay0.rates(np.array(min(fits, key=lambda f: f["cost"])["theta"]))
        Kp = [
            float(10 ** min(fits, key=lambda f: f["cost"])["theta"][len(kf) + len(lay0.free_kr) + j])
            for j in range(len(lay0.overall_used))
        ]
        base = {"kf": kf.tolist(), "kr": kr.tolist(), "Kp_all": Kp, "source": "fresh fit"}
    else:
        log(f"출발점: {base['source']}의 고전 피팅")

    variants = [None] + cands
    jobs = []
    for i, c in enumerate(variants):
        net, lay, prior = setup(p, c)
        jobs.append((i, net, lay, start_theta(p, lay, base, c), prior, wp, train, nfev_c))
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers(len(jobs))) as ex:
        fits = list(ex.map(_fit, jobs))
    log(f"맞추기: 반응식 {len(jobs)}개 × 최대 {nfev_c}회 {time.time() - t0:.0f} s")

    rows = []
    n_train = None
    for (i, net, lay, _t0, _pr, _wp, _tr, _n), f in zip(jobs, fits, strict=True):
        c = variants[i]
        row = {"name": c.name if c else "as written", "kind": c.kind if c else None, "step": c.step if c else None}
        if "theta" not in f:
            row["error"] = f.get("error")
            rows.append(row)
            continue
        ev = evaluate(net, lay, np.array(f["theta"]), exps)
        ssr = sum(ev["ssr"][n] for n in tr_names)
        n_train = sum(ev["n"][n] for n in tr_names)
        kf, kr = lay.rates(np.array(f["theta"]))
        row.update(
            {
                "train": _rms(ev["per"], tr_names),
                "holdout": _rms(ev["per"], ho_names),
                "per": ev["per"],
                "ssr_train": ssr,
                "n_params": len(f["theta"]),
                "cost": f["cost"],
                "nfev": f["nfev"],
                "seconds": f["seconds"],
                "k_new": {"kf": float(kf[-1]), "kr": float(kr[-1])} if c else None,
            }
        )
        rows.append(row)
    base_row = rows[0]
    if "train" not in base_row:
        raise SystemExit(f"적어 둔 반응식의 고전 피팅 실패: {base_row.get('error')}")

    def aic(r):
        return n_train * math.log(max(r["ssr_train"], 1e-300) / n_train) + 2 * r["n_params"]

    for r in rows:
        if "train" in r:
            r["d_aic"] = aic(r) - aic(base_row)
    cand_rows = sorted([r for r in rows[1:] if "train" in r], key=lambda r: r["d_aic"])
    verdict = decide(base_row, cand_rows, tr_names, ho_names)

    log(
        f"적어 둔 반응식: 학습 {base_row['train']:.2f}σ"
        + (f", 홀드아웃 {base_row['holdout']:.2f}σ" if ho_names else "")
    )
    for r in cand_rows[:10]:
        ho = f", 홀드아웃 {r['holdout']:.2f}σ" if r.get("holdout") is not None else ""
        log(f"  {r['step']:32s} 학습 {r['train']:.2f}σ{ho}, ΔAIC {r['d_aic']:+.1f}")
    log(f"결론: {verdict['note']}" + (f" → {verdict['suggested_step']}" if verdict["suggested_step"] else ""))
    if verdict["holdout_note"]:
        log(verdict["holdout_note"])

    report = {
        "project": p.name,
        "method": "ode",
        "quick": quick,
        "kinds": list(kinds),
        "start": base["source"],
        "prior_centres_log10": {
            "dead_end_kf": KF_BIND_LOG10,
            "dead_end_Ki_mM": 10**KI_LOG10,
            "inactivation": K_INACT_LOG10,
            "background": K_BACKGROUND_LOG10,
        },
        "verdict": verdict,
        "as_written": {k: v for k, v in base_row.items() if k != "per"},
        "candidates": [{**{k: v for k, v in r.items() if k != "per"}} for r in cand_rows],
        "per_experiment": {
            r["step"] or "as written": {n: v["__all__"] for n, v in r["per"].items()} for r in [base_row, *cand_rows]
        },
        "candidate_list": [asdict(c) for c in cands],
    }
    (out / "suggest.json").write_text(json.dumps(report, indent=1, ensure_ascii=False), encoding="utf-8", newline="\n")
    plots.write_rows(
        out / "candidates.csv",
        ["step", "kind", "train_rmse_sigma", "holdout_rmse_sigma", "d_aic", "k_new_kf", "k_new_kr"],
        [
            [
                r["step"] or "as written",
                r["kind"] or "",
                r["train"],
                r.get("holdout"),
                r.get("d_aic"),
                (r.get("k_new") or {}).get("kf"),
                (r.get("k_new") or {}).get("kr"),
            ]
            for r in [base_row, *cand_rows]
        ],
    )
    _figure(out, base_row, cand_rows)
    return report
