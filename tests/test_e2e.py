"""AC-06 (end to end, short mode) and AC-13 (a non-weakly-reversible project also runs to the end)."""

import csv
import json
import re
import shutil

from conftest import EXAMPLE, manifest, run_cli, run_dir_from


def test_e2e_quick_runs_all_stages(quick_run):
    m = manifest(quick_run)
    assert m["status"] == "completed", m.get("error")
    for st in ("1_verdict", "2_data", "3_fit", "4_sim", "5_design"):
        assert m["stages"][st]["status"] in ("done", "cached"), (st, m["stages"][st])
    assert m["mode"] == "quick"


def test_e2e_outputs(quick_run):
    rd = quick_run
    # ① verdict report
    rep = (rd / "1_verdict" / "report.txt").read_text(encoding="utf-8")
    assert "약가역 ✔" in rep and "결핍 0" in rep
    assert json.loads((rd / "1_verdict" / "verdict.json").read_text(encoding="utf-8"))["networks"]["main"][
        "weakly_reversible"
    ]
    # ③ samples
    samples = sorted((rd / "3_fit" / "samples").glob("sample_seed*.json"))
    assert len(samples) >= 2
    kept = [json.loads(f.read_text(encoding="utf-8")) for f in samples]
    assert sum(s["kept"] for s in kept) >= 2
    for s in kept:
        assert len(s["kf"]) == 5 and len(s["kr"]) == 5
    # ④ validation results
    with open(rd / "4_sim" / "validation.csv", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert rows and all(r["ok"] == "True" for r in rows)
    # ⑤ design table and pipette table
    with open(rd / "5_design" / "design_table.csv", encoding="utf-8") as fh:
        table = list(csv.DictReader(fh))
    assert len(table) >= 4
    with open(rd / "5_design" / "pipette.csv", encoding="utf-8") as fh:
        pip = list(csv.DictReader(fh))
    assert {r["set"] for r in pip} == {"L1", "L2", "D1", "D0", "Z"}
    d1 = next(r for r in pip if r["set"] == "D1")
    assert float(d1["Et_final_nM"]) > 0 and float(d1["buffer_uL"]) > 0
    s = json.loads((rd / "5_design" / "design_summary.json").read_text(encoding="utf-8"))
    assert s["robust_fail"] == 0 and s["rho_star"] > s["rho_min"]


def non_wr_project(tmp_path):
    """The example project with the last step one-way, and data generated from that non-WR truth (σ = 0.003),
    as when a truly irreversible reaction (e.g. the pNP control) is run on purpose."""
    import numpy as np

    from enzopt.core.massaction import haldane_kr
    from enzopt.core.project import load_project
    from enzopt.synth import _sim, observable

    proj = tmp_path / "projects" / "example_ldh_irreversible"
    shutil.copytree(EXAMPLE, proj)
    y = (proj / "project.yaml").read_text(encoding="utf-8")
    y = y.replace("name: example_ldh_synth", "name: example_ldh_irreversible")
    y = y.replace("    - EQ <=> E + NAD", "    - EQ -> E + NAD")
    y = re.sub(r"    haldane_step: 5 .*\n", "", y)
    y = re.sub(r"control: .*\n  one_way_steps: \[5\]\n", "", y)
    (proj / "project.yaml").write_text(y, encoding="utf-8")
    p = load_project(proj)
    kf = np.array([1e4, 5e3, 300.0, 1000.0, 200.0])
    kr = haldane_kr(kf, np.array([50.0, 500.0, 1000.0, 200.0, 1.0]), 23.0, 5)
    kr[4] = 0.0
    k = p.network.rate_vector(kf, kr)
    rng = np.random.default_rng(1)
    for e in p.experiments.values():
        t = np.loadtxt(e.file, delimiter=",", skiprows=1)[:, 0]
        a = observable(p.network, _sim(p.network, k, e.initial, e.Et, t), p.observable_weights())
        a = np.clip(a + rng.normal(0, 0.003, len(t)), 0.0, None)
        with open(e.file, "w", encoding="utf-8") as fh:
            fh.write("time_s,A340\n" + "".join(f"{x:.1f},{v:.6f}\n" for x, v in zip(t, a, strict=True)))
    return proj


def test_non_wr(tmp_path):
    proj = non_wr_project(tmp_path)
    r = run_cli("run", proj, "--quick", "--runs-root", tmp_path / "runs")
    rd = run_dir_from(r.stdout)
    m = manifest(rd)
    # does not stop because of the verdict: every stage after ① ran
    assert m["stages"]["1_verdict"]["status"] == "done"
    assert m["stages"]["2_data"]["status"] == "done"
    assert m["stages"]["3_fit"]["status"] == "done", m.get("error")
    assert m["status"] == "completed", m.get("error")
    assert m["flags"]["guarantee_149"] is False
    assert "149 보장 없음" in m["flags"]["banner"]
    rep = (rd / "1_verdict" / "report.txt").read_text(encoding="utf-8")
    assert "149 보장 없음" in rep
