"""AC-05 (realistic synthetic data: held-out extrapolation of the WR PINN samples) and AC-08 (GPU float32 vs CPU
float64, automatic CPU fallback). Long tier."""

import json
import os
import time

import numpy as np
import pytest
from conftest import EVALS, EXAMPLE, manifest, run_cli, run_dir_from

from enzopt.core.yamlio import load_yaml

pytestmark = pytest.mark.long


def test_realistic_holdout(full_run):
    syn = load_yaml(EVALS / "synthetic.yaml")
    limit = float(syn["holdout_criterion"]["rmse_max_A340"])  # 0.006 = 2σ
    f = json.loads((full_run / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    assert f["seeds"] == [0, 1, 2, 3]  # seed 4
    per_seed = f["holdout_rmse"]["per_seed"]
    assert per_seed, "no kept WR samples"
    report = {}
    for exp in ("C", "Dp"):
        vals = [v[exp] for v in per_seed.values()]
        report[exp] = {"mean": float(np.mean(vals)), "worst": float(np.max(vals))}
    print("holdout RMSE (A340):", report, "limit", limit)
    for exp, r in report.items():
        assert r["mean"] <= limit, f"{exp} mean {r['mean']:.4f} > {limit}"
        assert r["worst"] <= limit, f"{exp} worst {r['worst']:.4f} > {limit}"


def _train(device, dtype):
    from enzopt.core.massaction import MassAction
    from enzopt.core.project import load_project
    from enzopt.pinn.ldh import PinnConfig, PinnExperiment, train_pinn
    from enzopt.pipeline.common import curve, rmse

    p = load_project(EXAMPLE)
    data = {n: np.loadtxt(e.file, delimiter=",", skiprows=1) for n, e in p.experiments.items()}
    exps = [
        PinnExperiment(n, e.initial, e.Et, data[n][:, 0], data[n][:, 1])
        for n, e in p.experiments.items()
        if e.role == "train"
    ]
    pr = p.prior
    prior = np.array(pr["kf_log10"] + pr["kr_log10"] + [pr["Kp_log10"]], float)
    cfg = PinnConfig(seeds=[0, 1, 2, 3], device=device, dtype=dtype)
    t0 = time.time()
    res = train_pinn(exps, prior, cfg)
    wall = time.time() - t0
    model = MassAction.from_network(p.network)
    hold = []
    for r in res:
        k = p.network.rate_vector(r.kf, r.kr)
        for n, e in p.experiments.items():
            if e.role == "holdout":
                a = curve(p.network, k, e.initial, e.Et, data[n][:, 0], p.observable_weights(), model=model)
                hold.append(rmse(a, data[n][:, 1]))
    return wall, float(np.mean(hold)), hold


def test_gpu_vs_cpu():
    import torch

    assert torch.cuda.is_available(), "GPU가 없어 GPU/CPU 비교를 할 수 없음"
    _train("cuda", "float32")  # warm-up (CUDA context, kernels)
    t32, r32, h32 = _train("cuda", "float32")
    t64, r64, h64 = _train("cpu", "float64")
    print(f"GPU float32 {t32:.1f} s, holdout mean {r32:.5f} | CPU float64 {t64:.1f} s, holdout mean {r64:.5f}")
    assert abs(r32 - r64) / r64 <= 0.10
    assert t32 < t64


def test_cpu_fallback_recorded(tmp_path):
    """Without a usable GPU the run switches to CPU float64 and the manifest says so."""
    r = run_cli(
        "run",
        EXAMPLE,
        "--quick",
        "--no-cache",
        "--runs-root",
        tmp_path / "runs",
        "--stop-after",
        "3_fit",
        env={"CUDA_VISIBLE_DEVICES": ""},
    )
    rd = run_dir_from(r.stdout)
    m = manifest(rd)
    assert m["stages"]["3_fit"]["status"] == "done", m.get("error")
    assert m["device"]["used"] == "cpu" and m["device"]["precision"] == "float64"
    assert "CPU" in (m["device"]["fallback"] or "")
    assert any("CPU" in w for w in m["warnings"])
    assert os.path.exists(rd / "3_fit" / "fit_summary.json")
