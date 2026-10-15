"""``project.yaml`` schema: one file that fixes the network, experiments, identification, simulation and design.

Units: concentrations mM, time s, enzyme amounts nM in the file (``Et_nM``) and mM inside the code.
"""

from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path

from .network import Network, NetworkError, Step
from .yamlio import load_yaml

ROLES = ("train", "holdout", "design-check")
EXPERIMENT_KEYS = ("file", "temp_file", "Et_nM", "role", "format", "time_offset_s", "signal_files")
RESERVED_SIGNALS = ("time_s", "t_c")

DEFAULTS: dict = {
    "identification": {
        "method": "pinn",
        "seeds": 4,
        "noise_sigma": 0.003,
        "device": "auto",
        "precision": "auto",
        "pinn": {
            "adam_iters": 2000,
            "lbfgs_iters": 1000,  # 2026-10-10 F-13: 300 → 1000 (training fit 1.2–2σ → ≈1.1σ on the appendix C networks)
            "width": 32,
            "depth": 3,
            "w_prior": 1e-2,
            "restarts": 8,
            "range_margin": 0.05,  # sigmoid range widened 5% on each side (2026-10-10 Q-03 (d))
            "time_scale": "per_experiment",
            "inactivation": "none",  # "shared": learn one enzyme-activity curve a(t) for all experiments
            "inact_k": 12,
            "w_inact": 1e-2,
        },
        "init_jitter_decades": 1.0,
        "exclusion_rmse": "ode",  # SC5-3: training RMSE of the ODE re-solved with the sample ("pinn": network fit)
        "exclusion_sigma": 3.0,  # exclude a seed when that RMSE > exclusion_sigma × noise_sigma
        "baseline_fit": {"enabled": True, "starts": 4, "max_nfev": 200},
        "control": {"enabled": True, "seeds": 2},
    },
    "simulation": {
        "t_long_s": 3000.0,
        "n_points": 601,
        "same_class_runs": 12,
        "class_seed": 5,
        "hopsan": {
            "enabled": True,
            "one_step_k1": "auto",
            "representative_experiment": None,
            "representative_t_end_s": None,
            "max_steps": 1.2e8,
        },
    },
    "design": {
        "enabled": True,
        "target": "assay",
        "analyte": "NADH",
        "partner": "Pyr",
        "free_enzyme": "E",
        "analyte0_mM": 0.1,
        "conversion": 0.99,
        "time_s": 300.0,
        "Kp": None,
        "dead_end": None,
        "rho_grid": "auto",
        "rho_max": None,  # upper bound of ρ for the optimum: None, a number, or "train" (max partner in training wells)
        "Et_bounds_mM": [1e-6, 1e-2],
        "bisection_iters": 18,
        "check_t_end_s": 600.0,
        "pipette": {
            "volume_uL": 1000.0,
            "enzyme_uL": 10.0,
            "stocks_mM": {"NADH": 2.0, "Pyr": 10.0},
            "limit_test_rhos": "auto",  # auto: ρ_min × 0.3 and × 0.6 (below the equilibrium limit)
            "limit_test_enzyme_multiples": [[10, 20], [10]],
        },
    },
    "quick": {
        "seeds": 2,
        "adam_iters": 2000,  # PINN iterations are not reduced: fewer leave seeds unconverged (excluded by SC5-3)
        "lbfgs_iters": None,  # None: as in full mode
        "same_class_runs": 3,
        "rho_grid": "auto_quick",
        "bisection_iters": 10,
        "baseline_starts": 1,
        "baseline_max_nfev": 3,
        "control_seeds": 1,
        "hopsan": False,
    },
}


class ProjectError(ValueError):
    """Invalid project.yaml (message is shown to the user)."""


def _merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


@dataclass
class Experiment:
    name: str
    initial: dict
    Et_nM: float
    role: str
    file: Path | None = None
    temp_file: Path | None = None
    format: dict | None = None
    time_offset_s: float = 0.0  # added to recorded times: time since enzyme addition (e.g. mixing dead time)
    signal_files: dict = field(default_factory=dict)  # further signals sampled at other times: {name: path}

    @property
    def Et(self) -> float:
        return self.Et_nM * 1e-6


@dataclass
class Project:
    root: Path
    name: str
    description: str
    raw: dict
    network: Network
    observable: dict
    thermodynamics: dict
    prior: dict
    control: dict | None
    experiments: dict[str, Experiment]
    identification: dict
    simulation: dict
    design: dict
    quick_overrides: dict = field(default_factory=dict)
    mechanism: object = None  # core.mechanisms.Expanded when network.mechanism chose the steps
    signals: list = field(default_factory=list)  # further signals [{name, coefficients, sigma}] (F-13 G6)
    layout: object = None  # core.thermo.ThermoLayout: θ layout and dependent k₋ (F-13 G4)

    # ------------------------------------------------------------------ helpers
    @property
    def Kp(self) -> float | None:
        """K′ of the first overall reaction (None when no overall reaction has one)."""
        v = self.thermodynamics.get("Kp")
        return None if v is None else float(v)

    @property
    def reactions(self) -> list[dict]:
        """Overall reactions [{overall, Kp}] (one for the old single-reaction form)."""
        return list(self.thermodynamics.get("reactions") or [])

    @property
    def overall(self) -> list[str]:
        return [r["overall"] for r in self.reactions]

    def prior_vector(self, layout=None):
        """Prior centre as θ = log10 [kf, free kr, K′ of the constraining overall reactions] (``core.thermo``)."""
        import numpy as np

        lay = layout or self.layout
        pr = self.prior
        kf = [float(x) for x in pr["kf_log10"]]
        kr = [float(x) for x in pr["kr_log10"]]
        n_rev = sum(1 for st in self.network.steps if st.reversible)
        if len(kr) == n_rev and n_rev != lay.n_free_kr:  # one value per reversible step: keep the free ones
            rev = [i for i, st in enumerate(self.network.steps) if st.reversible]
            kr = [kr[rev.index(i)] for i in lay.free_kr]
        kp = pr.get("Kp_log10")
        if kp is None:
            kp = [float(np.log10(self.reactions[o]["Kp"])) for o in lay.overall_used]
        elif not isinstance(kp, list | tuple):
            kp = [float(kp)] * len(lay.overall_used)
        else:
            kp = [float(x) for x in kp][: len(lay.overall_used)]
        return np.array(kf + kr + kp, float)

    def kr_prior_all(self):
        """log10 k₋ prior of every step (free: the prior; dependent: implied at the prior centre; one-way: nan)."""
        import numpy as np

        lay = self.layout
        th = self.prior_vector()
        out = np.full(len(self.network.steps), np.nan)
        n = len(self.network.steps)
        for j, i in enumerate(lay.free_kr):
            out[i] = th[n + j]
        if lay.dependent:
            dep = lay.A @ th
            for j, i in enumerate(lay.dependent):
                out[i] = dep[j]
        return out

    @property
    def haldane_step(self) -> int | None:
        return self.thermodynamics.get("haldane_step")

    def by_role(self, role: str) -> list[Experiment]:
        return [e for e in self.experiments.values() if e.role == role]

    def control_network(self) -> Network | None:
        if not self.control:
            return None
        steps = self.control.get("one_way_steps") or []
        return self.network.with_one_way(steps, name=f"{self.network.name}_control")

    def design_network(self) -> Network:
        de = self.design.get("dead_end")
        if not de:
            return self.network
        return self.network.with_extra_steps(
            [de["step"]], name=f"{self.network.name}_design", extra_species=de.get("enzyme_species", [])
        )

    def input_files(self) -> list[Path]:
        files = [self.root / "project.yaml"]
        for e in self.experiments.values():
            for f in (e.file, e.temp_file, *e.signal_files.values()):
                if f is not None:
                    files.append(f)
        return files

    def observable_weights(self) -> dict:
        return dict(self.observable["coefficients"])

    def all_signals(self, sigma: float | None = None) -> list[dict]:
        """Primary observable then the further signals: [{name, coefficients, sigma}] (primary σ = noise_sigma)."""
        s0 = float(self.identification["noise_sigma"]) if sigma is None else float(sigma)
        prim = {"name": self.observable["name"], "coefficients": self.observable_weights(), "sigma": s0}
        return [prim] + [dict(x) for x in self.signals]

    def settings(self, quick: bool) -> dict:
        """Effective identification/simulation/design settings for the chosen mode."""
        ident = copy.deepcopy(self.identification)
        sim = copy.deepcopy(self.simulation)
        des = copy.deepcopy(self.design)
        if quick:
            q = self.quick_overrides
            ident["seeds"] = q["seeds"]
            ident["pinn"]["adam_iters"] = q["adam_iters"]
            if q.get("lbfgs_iters") is not None:
                ident["pinn"]["lbfgs_iters"] = q["lbfgs_iters"]
            ident["baseline_fit"]["starts"] = q["baseline_starts"]
            ident["baseline_fit"]["max_nfev"] = q["baseline_max_nfev"]
            ident["control"]["seeds"] = q["control_seeds"]
            sim["same_class_runs"] = q["same_class_runs"]
            if not q.get("hopsan", False):
                sim["hopsan"]["enabled"] = False
            des["rho_grid"] = q["rho_grid"]
            des["bisection_iters"] = q["bisection_iters"]
        return {"identification": ident, "simulation": sim, "design": des, "quick": bool(quick)}


def _num(v, what: str) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        raise ProjectError(f"{what}: 숫자가 아님 ({v!r})") from None


def load_project(path: str | Path, force_vendored_yaml: bool = False) -> Project:
    """Load ``<dir>/project.yaml`` (or the yaml path itself) and validate it."""
    p = Path(path)
    yml = p / "project.yaml" if p.is_dir() else p
    if not yml.exists():
        raise ProjectError(f"project.yaml을 찾지 못함: {yml}")
    root = yml.parent.resolve()
    raw = load_yaml(yml, force_vendored=force_vendored_yaml) or {}
    if not isinstance(raw, dict):
        raise ProjectError(f"{yml}: 최상위가 사전(key: value)이 아님")

    net = raw.get("network")
    if not isinstance(net, dict):
        raise ProjectError("network 항목이 없음")
    mech = None
    if net.get("mechanism") is not None:  # chosen from the mechanism library: steps are generated
        from .mechanisms import MechanismError, expand

        if net.get("steps") or net.get("species") or net.get("enzyme_species"):
            raise ProjectError("network: mechanism을 쓰면 species·enzyme_species·steps는 적지 않음(둘 중 하나만)")
        try:
            mech = expand(net["mechanism"])
        except MechanismError as e:
            raise ProjectError(f"network.mechanism: {e}") from None
        net = {**net, "species": mech.species, "enzyme_species": mech.enzyme_species, "steps": mech.steps}
        th = dict(net.get("thermodynamics") or {})
        th.setdefault("overall", mech.overall)
        if mech.haldane_step is not None:
            th.setdefault("haldane_step", mech.haldane_step)
        net["thermodynamics"] = th
    for key in ("species", "steps"):
        if not net.get(key):
            raise ProjectError(f"network.{key} 항목이 없음")
    try:
        network = Network(
            name=str(net.get("name", "network")),
            species=[str(s) for s in net["species"]],
            steps=[Step.parse(str(s)) for s in net["steps"]],
            enzyme_species=[str(s) for s in net.get("enzyme_species", [])],
        )
    except NetworkError as e:
        raise ProjectError(f"network: {e}") from None

    obs = net.get("observable") or {}
    if not obs.get("coefficients"):
        raise ProjectError("network.observable.coefficients(관측식 계수)가 없음")
    for s in obs["coefficients"]:
        if s not in network.species:
            raise ProjectError(f"관측식의 종 '{s}'이 species 목록에 없음")
    observable = {
        "name": str(obs.get("name", "A340")),
        "coefficients": {k: float(v) for k, v in obs["coefficients"].items()},
    }
    signals = []
    names = {observable["name"].lower()}
    for i, sg in enumerate(net.get("signals") or [], start=1):
        if not isinstance(sg, dict) or not sg.get("name") or not sg.get("coefficients") or sg.get("sigma") is None:
            raise ProjectError(f"network.signals {i}번째: name, coefficients, sigma가 모두 있어야 함")
        nm = str(sg["name"])
        if nm.lower() in names or nm.lower() in RESERVED_SIGNALS:
            raise ProjectError(f"network.signals: 신호 이름 '{nm}'이 겹치거나 쓸 수 없는 이름")
        names.add(nm.lower())
        for s_ in sg["coefficients"]:
            if s_ not in network.species:
                raise ProjectError(f"신호 '{nm}'의 종 '{s_}'이 species 목록에 없음")
        signals.append(
            {
                "name": nm,
                "coefficients": {k: float(v) for k, v in sg["coefficients"].items()},
                "sigma": _num(sg["sigma"], f"network.signals.{nm}.sigma"),
            }
        )

    thermo = dict(net.get("thermodynamics") or {})
    if thermo.get("reactions") is not None:  # F-13: several overall reactions [{overall, Kp}]
        rx = thermo["reactions"]
        if not isinstance(rx, list) or not all(isinstance(r, dict) and r.get("overall") for r in rx):
            raise ProjectError("network.thermodynamics.reactions: [{overall: 전체 반응식, Kp: K'}, ...] 형식이어야 함")
        reactions = [
            {
                "overall": str(r["overall"]),
                "Kp": None if r.get("Kp") is None else _num(r["Kp"], f"thermodynamics.reactions '{r['overall']}'.Kp"),
            }
            for r in rx
        ]
        if reactions:
            thermo.setdefault("overall", reactions[0]["overall"])
            if reactions[0]["Kp"] is not None:
                thermo.setdefault("Kp", reactions[0]["Kp"])
    else:
        if thermo.get("Kp") is not None:
            thermo["Kp"] = _num(thermo["Kp"], "network.thermodynamics.Kp")
        reactions = [{"overall": str(thermo["overall"]), "Kp": thermo.get("Kp")}] if thermo.get("overall") else []
    thermo["reactions"] = reactions
    if thermo.get("Kp") is not None:
        thermo["Kp"] = _num(thermo["Kp"], "network.thermodynamics.Kp")
    hs = thermo.get("haldane_step")
    if hs is not None:
        hs = int(hs)
        if not (1 <= hs <= len(network.steps)) or not network.steps[hs - 1].reversible:
            raise ProjectError(f"haldane_step {hs}: 가역 단계 번호여야 함")
        thermo["haldane_step"] = hs
    from .thermo import ThermoError, layout

    try:
        lay = layout(network, [r["overall"] for r in reactions], [hs] if hs else None)
    except ThermoError as e:
        raise ProjectError(f"network.thermodynamics: {e}") from None
    for o in lay.overall_used:
        if reactions[o]["Kp"] is None:
            raise ProjectError(f"전체 반응 '{reactions[o]['overall']}'의 K'(Kp)가 없음 — 가역 순환이 이 반응을 이룸")
    if not reactions and thermo.get("Kp") is None and any(st.reversible for st in network.steps):
        raise ProjectError("network.thermodynamics.Kp(평형 상수 K')가 없음")

    prior = dict(net.get("prior") or {})
    n = len(network.steps)
    if prior:
        if len(prior.get("kf_log10", [])) != n:
            listed = "; ".join(f"{i}. {st}" for i, st in enumerate(network.steps, start=1))
            raise ProjectError(f"prior.kf_log10 개수가 단계 수({n})와 다름 — 단계: {listed}")
        n_rev = sum(1 for st in network.steps if st.reversible)
        n_kr = lay.n_free_kr
        if len(prior.get("kr_log10", [])) not in (n_kr, n_rev):
            dep = ", ".join(str(i + 1) for i in lay.dependent) or "없음"
            raise ProjectError(
                f"prior.kr_log10 개수가 {n_kr}이어야 함(가역 단계 {n_rev}개 중 열역학 종속 단계 {dep} 제외; "
                f"가역 단계마다 {n_rev}개를 적어도 됨)"
            )
        kp = prior.get("Kp_log10")
        if isinstance(kp, list | tuple) and len(kp) < len(lay.overall_used):
            raise ProjectError(f"prior.Kp_log10 개수가 K'를 쓰는 전체 반응 수({len(lay.overall_used)})보다 적음")
        if kp is not None and not isinstance(kp, list | tuple) and len(lay.overall_used) > 1:
            raise ProjectError("prior.Kp_log10: 전체 반응이 여럿이면 목록으로 적음")
        prior.setdefault("sd_decades", 1.5)

    control = raw.get("control")
    if control:
        for s in control.get("one_way_steps", []):
            if not (1 <= int(s) <= n):
                raise ProjectError(f"control.one_way_steps: 단계 번호 {s}가 범위 밖")

    exps: dict[str, Experiment] = {}
    for name, e in (raw.get("experiments") or {}).items():
        if not isinstance(e, dict):
            raise ProjectError(f"experiments.{name}: 사전이 아님")
        role = str(e.get("role", "train"))
        if role not in ROLES:
            raise ProjectError(f"experiments.{name}.role '{role}': {ROLES} 중 하나")
        if "Et_nM" not in e:
            raise ProjectError(f"experiments.{name}: Et_nM(효소량, nM)이 없음")
        initial = {}
        for s in network.species:
            if s in e:
                initial[s] = _num(e[s], f"experiments.{name}.{s}")
        for k in e:
            if k not in EXPERIMENT_KEYS and k not in network.species:
                raise ProjectError(f"experiments.{name}: 알 수 없는 항목 '{k}'(종 이름이 아님)")
        exps[str(name)] = Experiment(
            name=str(name),
            initial=initial,
            Et_nM=_num(e["Et_nM"], f"experiments.{name}.Et_nM"),
            role=role,
            file=(root / e["file"]) if e.get("file") else None,
            temp_file=(root / e["temp_file"]) if e.get("temp_file") else None,
            format=e.get("format"),
            time_offset_s=_num(e.get("time_offset_s", 0.0), f"experiments.{name}.time_offset_s"),
            signal_files={str(k): root / v for k, v in (e.get("signal_files") or {}).items()},
        )
        for k in exps[str(name)].signal_files:
            if k not in [sg["name"] for sg in signals]:
                raise ProjectError(f"experiments.{name}.signal_files: '{k}'는 network.signals에 없는 신호")
    if not any(x.role == "train" for x in exps.values()):
        raise ProjectError("학습(role: train) 실험이 하나도 없음")

    ident = _merge(DEFAULTS["identification"], raw.get("identification") or {})
    sim = _merge(DEFAULTS["simulation"], raw.get("simulation") or {})
    des = _merge(DEFAULTS["design"], raw.get("design") or {})
    if des.get("Kp") is None:
        des["Kp"] = thermo.get("Kp")
    if des.get("dead_end"):
        de = des["dead_end"]
        if "step" not in de or "kf" not in de or "Ki_mM" not in de:
            raise ProjectError("design.dead_end: step, kf, Ki_mM이 모두 있어야 함")
    quick = _merge(DEFAULTS["quick"], raw.get("quick") or {})
    if des.get("enabled", True):
        names = [des["analyte"], des.get("partner"), des["free_enzyme"], des.get("product"), des.get("side_product")]
        for s in [x for x in names if x] + list(des.get("fixed_ratio") or {}):
            if s not in network.species:
                raise ProjectError(f"design: 종 '{s}'이 species 목록에 없음")

    return Project(
        root=root,
        name=str(raw.get("name") or root.name),
        description=str(raw.get("description", "")),
        raw=raw,
        network=network,
        observable=observable,
        thermodynamics=thermo,
        prior=prior,
        control=control,
        experiments=exps,
        identification=ident,
        simulation=sim,
        design=des,
        quick_overrides=quick,
        mechanism=mech,
        signals=signals,
        layout=lay,
    )


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_hash(obj) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str, ensure_ascii=False).encode("utf-8")).hexdigest()
