"""Realistic synthetic data (``evals/synthetic.yaml``): truth mechanism + instrument effects → standard CSV files.

Effects per experiment, drawn in this order from one seeded generator: enzyme amount factor U(1±e),
start-time offset U(±δ), linear baseline drift slope U(±d)/900 s, Gaussian noise, temperature record.
The identification side only knows the nominal enzyme amount and nominal times.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .core.massaction import MassAction, SolverSettings, atol_vector, haldane_kr, simulate
from .core.network import Network, x0_vector
from .core.yamlio import load_yaml


def truth_network(cfg: dict) -> Network:
    t = cfg["truth"]
    return Network.from_strings(t.get("network", "truth"), t["species"], t["steps"], t["enzyme_species"])


def truth_rates(cfg: dict, network: Network) -> tuple[np.ndarray, np.ndarray]:
    t = cfg["truth"]
    kf = np.array(t["kf"], float)
    hs = int(t["haldane_step"])
    kr = np.ones(len(kf))
    others = [i for i in range(len(kf)) if i != hs - 1]
    kr[others] = np.array(t["kr"], float)
    kr = haldane_kr(kf, kr, float(t["Kp"]), hs)
    return kf, kr


def _sim(network: Network, k: np.ndarray, ic: dict, Et: float, t: np.ndarray) -> np.ndarray:
    model = MassAction.from_network(network)
    st = SolverSettings(rtol=1e-8, atol_enzyme_rel=1e-8, atol_other=1e-11)
    x0 = x0_vector(network, ic, Et)
    return simulate(model, k, x0, t, atol=atol_vector(network.species, network.enzyme_species, Et, st), settings=st)


def choose_Et(network: Network, k: np.ndarray, ic: dict, target_half: float = 120.0, analyte: str = "NADH") -> float:
    """Et (mM) such that the experiment goes half way to equilibrium in ``target_half`` s (CADKIMM rule)."""
    Et = 1e-6
    t = np.geomspace(1e-3, 1e6, 4000)
    X = _sim(network, k, ic, Et, t)
    y = X[:, network.index()[analyte]]
    half = y[0] - 0.5 * (y[0] - y[-1])
    th = t[np.argmax(y <= half)] if y[0] > y[-1] else t[np.argmax(y >= half)]
    return float(Et * th / target_half)


def observable(network: Network, X: np.ndarray, coefficients: dict) -> np.ndarray:
    idx = network.index()
    return sum(c * X[:, idx[s]] for s, c in coefficients.items())


def generate(cfg: dict) -> dict:
    """Return {set name: dict(t, a340_obs, a340_clean, temp_t, temp, meta)} and the nominal enzyme amount."""
    net = truth_network(cfg)
    kf, kr = truth_rates(cfg, net)
    k = net.rate_vector(kf, kr)
    exps = dict(cfg["experiments"])
    rule = cfg["truth"]["enzyme"]
    first = next(iter(exps))
    ic_first = {s: v for s, v in exps[first].items() if s in net.species}
    Et_nom = choose_Et(net, k, ic_first, float(rule["half_life_s"]))
    Et_nom_nM = float(f"{Et_nom * 1e6:.4g}")
    eff = cfg["effects"]
    obs = cfg["observation"]
    coeffs = obs["A340_coefficients"]
    rng = np.random.default_rng(int(cfg["seed"]))
    sets = {name: (dict(v), float(obs["t_end_s"]), Et_nom_nM) for name, v in exps.items()}
    dc = cfg.get("design_check") or {}
    for name, v in (dc.get("sets") or {}).items():
        sets[name] = (dict(v), float(dc.get("t_end_s", obs["t_end_s"])), float(v["Et_nM"]))
    out = {}
    for name, (spec, t_end, et_nm) in sets.items():
        ic = {s: float(v) for s, v in spec.items() if s in net.species}
        f_E = rng.uniform(1 - eff["enzyme_error_frac"], 1 + eff["enzyme_error_frac"])
        dt0 = rng.uniform(-eff["start_time_error_s"], eff["start_time_error_s"])
        slope = rng.uniform(-eff["baseline_drift_A340_per_900s"], eff["baseline_drift_A340_per_900s"]) / 900.0
        t_nom = np.arange(float(obs["t_first_s"]), t_end + 1e-9, float(obs["dt_s"]))
        X = _sim(net, k, ic, et_nm * 1e-6 * f_E, t_nom + dt0)
        clean = observable(net, X, coeffs)
        noise = rng.normal(0.0, eff["noise_sigma_A340"], len(t_nom))
        a_obs = clean + slope * t_nom + noise
        tc = eff["temperature"]
        temp_t = np.arange(0.0, t_end + 1e-9, float(tc["dt_s"]))
        temp = tc["mean_C"] + rng.normal(0.0, tc["sd_C"], len(temp_t))
        out[name] = dict(
            t=t_nom,
            a340_obs=a_obs,
            a340_clean=clean,
            temp_t=temp_t,
            temp=temp,
            spec=spec,
            Et_nM=et_nm,
            meta=dict(enzyme_factor=float(f_E), start_offset_s=float(dt0), drift_per_s=float(slope)),
        )
    return {"sets": out, "Et_nominal_nM": Et_nom_nM, "kf": kf.tolist(), "kr": kr.tolist()}


def write_csv(path: Path, header: str, cols: list[np.ndarray], fmts: list[str]) -> None:
    with open(path, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(header + "\n")
        for row in zip(*cols, strict=True):
            fh.write(",".join(f % v for f, v in zip(fmts, row, strict=True)) + "\n")


def write_project_data(cfg: dict, project_dir: Path) -> dict:
    """Write data/<set>.csv, data/<set>_temp.csv and data/truth.json into ``project_dir``."""
    res = generate(cfg)
    data = Path(project_dir) / "data"
    data.mkdir(parents=True, exist_ok=True)
    truth = {
        "note": "합성 정답(파이프라인은 읽지 않음)",
        "Et_nominal_nM": res["Et_nominal_nM"],
        "kf": res["kf"],
        "kr": res["kr"],
        "sets": {},
    }
    for name, s in res["sets"].items():
        write_csv(data / f"set{name}.csv", "time_s,A340", [s["t"], s["a340_obs"]], ["%.1f", "%.6f"])
        write_csv(data / f"set{name}_temp.csv", "time_s,T_C", [s["temp_t"], s["temp"]], ["%.1f", "%.3f"])
        truth["sets"][name] = dict(s["meta"], Et_nM_nominal=s["Et_nM"], clean_A340=[float(x) for x in s["a340_clean"]])
    with open(data / "truth.json", "w", encoding="utf-8") as fh:
        json.dump(truth, fh, indent=1, ensure_ascii=False)
    return res


def load_config(path: str | Path) -> dict:
    return load_yaml(path)


# ------------------------------------------------------------------------------------------------ F-13 (version 2)
def _round_prior(values) -> list[int]:
    return [int(round(float(np.log10(v)))) for v in values]


def general_truth(cfg: dict, key: str) -> dict:
    """Network, rate constants, overall reactions and θ layout of one network of ``evals/synthetic_general.yaml``."""
    from .core.thermo import layout

    n = cfg["networks"][key]
    net = Network.from_strings(key, n["species"], n["steps"], n["enzyme_species"])
    reactions = [dict(r) for r in n.get("reactions") or []]
    overall = [str(r["overall"]) for r in reactions]
    Kp = [float(r["Kp"]) if r.get("Kp") is not None else None for r in reactions]
    lay = layout(net, overall)
    kf = np.array(n["kf"], float)
    kr = np.array(n["kr"], float)
    return dict(network=net, kf=kf, kr=kr, overall=overall, Kp=Kp, layout=lay, spec=n)


def generate_general(cfg: dict, key: str) -> dict:
    """Synthetic data of one network: every signal of every experiment with the instrument effects."""
    tr = general_truth(cfg, key)
    net, n = tr["network"], tr["spec"]
    k = net.rate_vector(tr["kf"], tr["kr"])
    com = cfg["common"]
    eff, obs = com["effects"], com["observation"]
    exps = dict(n["experiments"])
    first = next(name for name, v in exps.items() if v.get("role") == "train")
    ic_first = {s: float(v) for s, v in exps[first].items() if s in net.species}
    Et_nom = choose_Et(net, k, ic_first, float(com["enzyme"]["half_life_s"]), analyte=str(n["analyte"]))
    Et_nom_nM = float(f"{Et_nom * 1e6:.4g}")
    rng = np.random.default_rng(int(cfg["seed"]) + list(cfg["networks"]).index(key))
    out = {}
    for name, spec in exps.items():
        ic = {s: float(v) for s, v in spec.items() if s in net.species}
        f_E = rng.uniform(1 - eff["enzyme_error_frac"], 1 + eff["enzyme_error_frac"])
        dt0 = rng.uniform(-eff["start_time_error_s"], eff["start_time_error_s"])
        t_end = float(obs["t_end_s"])
        signals = {}
        for sg in n["signals"]:
            smp = sg.get("sampling") or {}
            t_nom = np.arange(
                float(smp.get("t_first_s", obs["t_first_s"])), t_end + 1e-9, float(smp.get("dt_s", obs["dt_s"]))
            )
            drift = float(sg.get("drift_per_900s", 0.0))
            slope = rng.uniform(-drift, drift) / 900.0 if drift > 0 else 0.0
            X = _sim(net, k, ic, Et_nom_nM * 1e-6 * f_E, t_nom + dt0)
            clean = observable(net, X, sg["coefficients"])
            y = clean + slope * t_nom + rng.normal(0.0, float(sg["sigma"]), len(t_nom))
            signals[str(sg["name"])] = dict(t=t_nom, y=y, clean=clean, slope=float(slope), sampled=bool(smp))
        tc = eff["temperature"]
        temp_t = np.arange(0.0, t_end + 1e-9, float(tc["dt_s"]))
        temp = tc["mean_C"] + rng.normal(0.0, tc["sd_C"], len(temp_t))
        out[name] = dict(
            spec=spec,
            signals=signals,
            temp_t=temp_t,
            temp=temp,
            meta=dict(enzyme_factor=float(f_E), start_offset_s=float(dt0)),
        )
    return {"sets": out, "Et_nominal_nM": Et_nom_nM, "truth": tr}


def _yaml_flow(d: dict) -> str:
    def v(x):
        if isinstance(x, float):
            return f"{x:.10g}"
        return str(x)

    return "{" + ", ".join(f"{k}: {v(x)}" for k, x in d.items()) + "}"


def general_project_yaml(cfg: dict, key: str, res: dict) -> str:
    """project.yaml text for one generated network (design settings are added per network in M6-4)."""
    tr = res["truth"]
    net, n, lay = tr["network"], tr["spec"], tr["layout"]
    prim, extra = n["signals"][0], n["signals"][1:]
    pr = cfg["common"]["prior"]
    kf_p = _round_prior(tr["kf"])
    kr_p = _round_prior([tr["kr"][i] for i in lay.free_kr])
    kp_p = _round_prior([tr["Kp"][o] for o in lay.overall_used])
    note = str(n.get("note", key)).replace('"', "'")
    L = [
        f"# F-13 합성 시험 — evals/synthetic_general.yaml의 {key}에서 생성(enzopt synth). data/truth.json은 정답",
        f"name: gen_{key}",
        f'description: "{note}"',
        "",
        "network:",
        f"  name: {key}",
        f"  species: [{', '.join(net.species)}]",
        f"  enzyme_species: [{', '.join(net.enzyme_species)}]",
        "  steps:",
    ]
    L += [f"    - {st}" for st in n["steps"]]
    L += [f"  observable: {{name: {prim['name']}, coefficients: {_yaml_flow(prim['coefficients'])}}}"]
    if extra:
        L += ["  signals:                    # 주 신호 밖의 신호(같은 시점이면 CSV 열, 다른 시점이면 signal_files)"]
        for sg in extra:
            L += [f"    - {{name: {sg['name']}, coefficients: {_yaml_flow(sg['coefficients'])}, sigma: {sg['sigma']}}}"]
    L += ["  thermodynamics:", "    reactions:"]
    for o, kp in zip(tr["overall"], tr["Kp"], strict=True):
        L += [f"      - {{overall: {o}" + (f", Kp: {kp:g}" if kp is not None else "") + "}"]
    dep = [i + 1 for i in lay.dependent]
    L += [
        f"  prior:                      # 사전분포 = 정답을 10의 거듭제곱으로 반올림, {pr['sd_decades']} 자릿수",
        f"    kf_log10: [{', '.join(map(str, kf_p))}]",
        f"    kr_log10: [{', '.join(map(str, kr_p))}]   # 가역 단계 중 종속 단계 {dep} 제외",
    ]
    if kp_p:
        L += [f"    Kp_log10: [{', '.join(map(str, kp_p))}]"]
    L += [f"    sd_decades: {pr['sd_decades']}", ""]
    if lay.dependent:
        L += [
            "control:                      # 비약가역 대조: 첫 종속 단계를 한 방향으로",
            f"  one_way_steps: [{dep[0]}]",
            "",
        ]
    L += ["experiments:"]
    for name, s in res["sets"].items():
        spec = {k: float(v) for k, v in s["spec"].items() if k != "role"}
        item = {"file": f"data/set{name}.csv", "temp_file": f"data/set{name}_temp.csv", **spec}
        item["Et_nM"] = res["Et_nominal_nM"]
        item["role"] = s["spec"]["role"]
        line = f"  {name}: " + _yaml_flow(item)
        sampled = [sg for sg, v in s["signals"].items() if v["sampled"]]
        if sampled:
            files = ", ".join(f"{sg}: data/set{name}_{sg}.csv" for sg in sampled)
            line = line[:-1] + ", signal_files: {" + files + "}}"
        L += [line]
    L += [
        "",
        "identification:",
        "  seeds: 4",
        f"  noise_sigma: {prim['sigma']}",
        "",
        "simulation:",
        "  hopsan: {enabled: false}",
        "",
        "design:",
        "  enabled: false              # ⑤ 설계 목표는 M6-4에서",
    ]
    return "\n".join(L) + "\n"


def write_general_projects(cfg: dict, root: Path, only: list[str] | None = None) -> dict:
    """Write ``root/<network>/`` (project.yaml + data/) for every network of a version-2 config."""
    out = {}
    for key in cfg["networks"]:
        if only and key not in only:
            continue
        res = generate_general(cfg, key)
        d = Path(root) / key
        data = d / "data"
        data.mkdir(parents=True, exist_ok=True)
        tr = res["truth"]
        truth = {
            "note": "합성 정답(파이프라인은 읽지 않음)",
            "Et_nominal_nM": res["Et_nominal_nM"],
            "kf": tr["kf"].tolist(),
            "kr": tr["kr"].tolist(),
            "overall": tr["overall"],
            "Kp": tr["Kp"],
            "sets": {},
        }
        for name, s in res["sets"].items():
            names = [k for k, v in s["signals"].items() if not v["sampled"]]
            main = [s["signals"][k] for k in names]
            cols = [main[0]["t"]] + [v["y"] for v in main]
            write_csv(data / f"set{name}.csv", ",".join(["time_s", *names]), cols, ["%.1f"] + ["%.6f"] * len(main))
            for sg, v in s["signals"].items():
                if v["sampled"]:
                    write_csv(data / f"set{name}_{sg}.csv", f"time_s,{sg}", [v["t"], v["y"]], ["%.1f", "%.6f"])
            write_csv(data / f"set{name}_temp.csv", "time_s,T_C", [s["temp_t"], s["temp"]], ["%.1f", "%.3f"])
            truth["sets"][name] = dict(
                s["meta"],
                Et_nM_nominal=res["Et_nominal_nM"],
                clean={k: [float(x) for x in v["clean"]] for k, v in s["signals"].items()},
                drift_per_s={k: v["slope"] for k, v in s["signals"].items()},
            )
        with open(data / "truth.json", "w", encoding="utf-8") as fh:
            json.dump(truth, fh, indent=1, ensure_ascii=False)
        (d / "project.yaml").write_text(general_project_yaml(cfg, key, res), encoding="utf-8")
        out[key] = res["Et_nominal_nM"]
    return out
