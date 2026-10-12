"""AC-09: re-run → new folder, old folder untouched, unchanged inputs reuse the PINN stage from cache; an interrupted
run resumes from the interrupted stage."""

import hashlib
import json
import os

from conftest import EXAMPLE, manifest, run_cli, run_dir_from


def tree_hash(d):
    h = hashlib.sha256()
    for f in sorted(p for p in d.rglob("*") if p.is_file() and p.name != "progress.json"):
        h.update(str(f.relative_to(d)).encode())
        h.update(f.read_bytes())
    return h.hexdigest()


def test_rerun_new_folder_and_cache(quick_run, runs_root):
    before = tree_hash(quick_run)
    r = run_cli("run", EXAMPLE, "--quick", "--runs-root", runs_root)
    assert r.returncode == 0, r.stdout[-2000:]
    rd2 = run_dir_from(r.stdout)
    assert rd2 != quick_run and rd2.parent == quick_run.parent
    assert tree_hash(quick_run) == before  # previous run untouched
    m2 = manifest(rd2)
    assert m2["status"] == "completed"
    st = m2["stages"]["3_fit"]
    assert st["status"] == "cached" and st["cached_from"]  # PINN not re-trained
    assert "③ 식별: 캐시 사용" in r.stdout
    # the copied samples are identical
    a = json.loads((quick_run / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    b = json.loads((rd2 / "3_fit" / "fit_summary.json").read_text(encoding="utf-8"))
    assert a["kept"] == b["kept"] and a["holdout_rmse"] == b["holdout_rmse"]
    # warnings are recorded per stage, and a cached stage brings only its own (not those of later stages of the old run)
    m1 = manifest(quick_run)
    per_stage = [w for st_ in m1["stages"].values() for w in st_.get("warnings", [])]
    assert sorted(per_stage) == sorted(m1["warnings"])
    for name, st_ in m2["stages"].items():
        if st_["status"] == "cached":
            assert st_.get("warnings", []) == m1["stages"][name].get("warnings", [])


def test_resume_after_interruption(tmp_path):
    root = tmp_path / "runs"
    r = run_cli("run", EXAMPLE, "--quick", "--no-cache", "--runs-root", root, "--stop-after", "3_fit")
    rd = run_dir_from(r.stdout)
    m = manifest(rd)
    assert m["status"] == "interrupted"
    assert [m["stages"][s]["status"] for s in ("1_verdict", "2_data", "3_fit")] == ["done"] * 3
    assert m["stages"]["4_sim"]["status"] == "pending"
    # simulate a crash in the middle of stage ④ (power loss / sleep): partial folder, status running, dead process
    (rd / "4_sim").mkdir()
    (rd / "4_sim" / "partial.csv").write_text("half written", encoding="utf-8")
    m["stages"]["4_sim"]["status"] = "running"
    m["status"] = "running"
    m["pid"] = 999999
    (rd / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")
    from enzopt.pipeline.runner import run_state

    assert run_state(rd) == "interrupted"
    started = {s: m["stages"][s]["started"] for s in ("1_verdict", "2_data", "3_fit")}
    fit_mtime = os.path.getmtime(rd / "3_fit" / "fit_summary.json")
    r2 = run_cli("run", "--resume", rd)
    assert r2.returncode == 0, r2.stdout[-2000:]
    m2 = manifest(rd)
    assert m2["status"] == "completed"
    for s, t in started.items():
        assert m2["stages"][s]["started"] == t  # finished stages were skipped
        assert "이미 완료(건너뜀)" in r2.stdout
    assert os.path.getmtime(rd / "3_fit" / "fit_summary.json") == fit_mtime
    assert m2["stages"]["4_sim"]["status"] == "done" and m2["stages"]["5_design"]["status"] == "done"
    assert not (rd / "4_sim" / "partial.csv").exists()  # the interrupted stage was redone from its start
    assert m2.get("resumed")
