"""AC-11: Hopsan backend — apparent one-step class batch and one representative sample of the full mechanism, each
compared with scipy (max difference recorded); without Hopsan the run uses scipy and records "Hopsan 미사용"."""

import json
import shutil

import pytest
from conftest import EXAMPLE, manifest, run_cli, run_dir_from


@pytest.mark.long
def test_hopsan_runs_and_differences_recorded(full_run):
    s = json.loads((full_run / "4_sim" / "sim_summary.json").read_text(encoding="utf-8"))
    h = s["hopsan"]
    assert h["used"] is True, h
    one = h["one_step"]
    assert one["n_pass"] == len(one["runs"]) >= 13
    assert all(r["backend"] == "hopsan" for r in one["runs"])
    assert one["max_abs_diff_vs_scipy_mM"] is not None and one["max_abs_diff_vs_scipy_mM"] < 1e-3
    full = h["full_representative"]
    assert full["max_abs_diff_analyte_mM"] is not None and full["max_abs_diff_A340"] is not None
    assert full["timestep"] > 0 and full["t_end"] > 0
    assert (full_run / "4_sim" / "hopsan" / "ldh_one_step.hmf").exists()
    assert (full_run / "4_sim" / "hopsan" / full["hmf"]).exists()
    m = manifest(full_run)
    assert m["flags"]["hopsan_used"] is True
    assert "Hopsan 사용" in (full_run / "1_verdict" / "report.txt").read_text(encoding="utf-8")


def test_without_hopsan_falls_back_to_scipy(tmp_path):
    proj = tmp_path / "projects" / "nohopsan"
    shutil.copytree(EXAMPLE, proj)
    y = (proj / "project.yaml").read_text(encoding="utf-8")
    (proj / "project.yaml").write_text(y + "\nquick:\n  hopsan: true\n", encoding="utf-8")
    r = run_cli(
        "run",
        proj,
        "--quick",
        "--runs-root",
        tmp_path / "runs",
        env={"HOPSANCLI": str(tmp_path / "missing_hopsancli.exe")},
    )
    rd = run_dir_from(r.stdout)
    m = manifest(rd)
    assert m["status"] == "completed", m.get("error")
    assert m["flags"]["hopsan_used"] is False
    assert "Hopsan 미사용" in m["flags"]["hopsan_note"]
    assert "Hopsan 미사용" in m["flags"]["hopsan_status"]
    assert any("Hopsan 미사용" in w for w in m["warnings"])
    rep = (rd / "1_verdict" / "report.txt").read_text(encoding="utf-8")
    assert "Hopsan 미사용" in rep
    s = json.loads((rd / "4_sim" / "sim_summary.json").read_text(encoding="utf-8"))
    assert all(r["backend"] == "scipy" for r in s["hopsan"]["one_step"]["runs"])
