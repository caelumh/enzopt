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
