"""AC-19 (dashboard shows projects/runs, tabs ①–⑤, running/failed/non-WR states) and AC-20 ([새 실행] starts a
detached engine process that finishes even after the dashboard process has ended)."""

import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import pytest
from conftest import EXAMPLE, ROOT, manifest

from enzopt.pipeline.runner import now_iso, run_state

APP = ROOT / "src" / "enzopt" / "dashboard" / "app.py"
TAB_LABELS = ["① 반응망", "② 데이터", "③ 식별", "④ 해석", "⑤ 설계"]


def _edit(run_dir: Path, created: str, **changes):
    m = manifest(run_dir)
    m["created"] = created
    for k, v in changes.items():
        if k == "stage_status":
            for s, st in v.items():
                m["stages"][s]["status"] = st
        else:
            m[k] = v
    (run_dir / "manifest.json").write_text(json.dumps(m, ensure_ascii=False), encoding="utf-8")


@pytest.fixture
def dash(tmp_path, quick_run, monkeypatch):
    projects = tmp_path / "projects"
    projects.mkdir()
    shutil.copytree(EXAMPLE, projects / "example_ldh_synth")
    runs = tmp_path / "runs" / "example_ldh_synth"
    runs.mkdir(parents=True)
    names = {}
    for i, kind in enumerate(["done", "failed", "running", "nonwr"]):
        d = runs / f"2026-10-0{i + 1}_1200"
        shutil.copytree(quick_run, d)
        names[kind] = d
    _edit(names["done"], "2026-10-01T12:00:00+09:00")
    _edit(
        names["failed"],
        "2026-10-02T12:00:00+09:00",
        status="failed",
        error={"stage": "3_fit", "message": "PINN 발산(시험)", "type": "StageFailed"},
        stage_status={"3_fit": "failed", "4_sim": "pending", "5_design": "pending"},
    )
    shutil.rmtree(names["failed"] / "4_sim")
    _edit(
        names["running"],
        "2026-10-03T12:00:00+09:00",
        status="running",
        pid=os.getpid(),
        stage_status={"4_sim": "running", "5_design": "pending"},
    )
    (names["running"] / "progress.json").write_text(
        json.dumps(
            {
                "stage": "4_sim",
                "stage_title": "④ 해석",
                "stage_index": 4,
                "n_stages": 5,
                "fraction": 0.5,
                "overall_fraction": 0.7,
                "message": "표본 1 · same03",
                "eta_s": 42.0,
                "updated": now_iso(),
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    m = manifest(names["nonwr"])
    flags = dict(m["flags"], guarantee_149=False, banner="149 보장 없음: 멸종 가능")
    _edit(names["nonwr"], "2026-10-04T12:00:00+09:00", flags=flags)
    monkeypatch.setenv("ENZOPT_PROJECTS", str(projects))
    monkeypatch.setenv("ENZOPT_RUNS", str(tmp_path / "runs"))
    return names


def _app():
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(APP), default_timeout=120)
    at.run()
    assert not at.exception, at.exception
    return at


def _select(at, run_dir):
    at.sidebar.radio[0].set_value(run_dir.name).run()
    assert not at.exception, at.exception
    return at


def test_dashboard_lists_and_tabs(dash):
    at = _app()
    assert at.sidebar.selectbox[0].value == "example_ldh_synth"
    opts = at.sidebar.radio[0].options
    assert len(opts) == 4
    assert opts[0].startswith(dash["nonwr"].name) and "완료" in opts[0]  # newest first, with state
    _select(at, dash["done"])
    assert [t.label for t in at.tabs] == TAB_LABELS
    assert any("약가역 ✔" in s.value for s in at.subheader)
    assert any("보장하는 것 / 전제" in i.value for i in at.info)
    assert len(at.get("imgs")) >= 10 or len(at.image) >= 10
    assert any(b.label == "새 실행" for b in at.sidebar.button)
    assert any("피펫 표" in m.value for m in at.markdown)


def test_dashboard_running_state(dash):
    at = _select(_app(), dash["running"])
    prog = at.get("progress")
    assert prog, "no progress bar"
    txt = json.dumps([p.proto.text for p in prog], ensure_ascii=False) if hasattr(prog[0], "proto") else str(prog)
    assert "④ 해석" in txt and "남은 시간" in txt
    assert any(b.label == "취소" for b in at.button)


def test_dashboard_failed_state(dash):
    at = _select(_app(), dash["failed"])
    errs = [e.value for e in at.error]
    assert any("③ 식별 단계 실패" in e and "PINN 발산(시험)" in e for e in errs), errs
    assert any(b.label == "③ 식별부터 다시" for b in at.button)
    assert any("실패" in c.value or "실행되지" in c.value for c in at.caption)


def test_dashboard_non_wr_banner_on_every_tab(dash):
    at = _select(_app(), dash["nonwr"])
    banners = [e for e in at.error if "149 보장 없음" in e.value]
    assert len(banners) >= len(TAB_LABELS)


def test_dashboard_design_tab_for_the_peak_goal(dash):
    """Goal ``peak`` writes a different design summary (stop time, worst-sample yield): the tab shows it."""
    peak = {
        "goal": "peak",
        "rho_star": 3.0,
        "Et_nM": 200.0,
        "robust_stop_s": 1234.0,
        "robust_yield": 0.567,
        "stop_window_s": [900.0, 1500.0],
        "window_fraction": 0.95,
    }
    (dash["done"] / "5_design" / "design_summary.json").write_text(json.dumps(peak), encoding="utf-8")
    at = _select(_app(), dash["done"])
    labels = {m.label: m.value for m in at.metric}
    assert labels.get("멈출 시각") == "1234 s" and labels.get("최악 표본 수율") == "56.7 %"
    assert not any("결과를 표시하지 못함" in c.value for c in at.caption)
    assert any("피펫 표" in m.value for m in at.markdown)


def _wait_completed(rd: Path, timeout=300.0) -> str:
    t0 = time.time()
    while time.time() - t0 < timeout:
        st = run_state(rd)
        if st in ("completed", "failed", "cancelled") or (st == "interrupted" and not manifest(rd).get("starting")):
            return st
        time.sleep(1.0)
    return run_state(rd)


def test_detached_run_survives_dashboard_exit(runs_root, quick_run):
    code = (
        "import sys; from pathlib import Path; from enzopt.dashboard.launch import start_run; "
        f"rd, pid = start_run(Path(r'{EXAMPLE}'), Path(r'{runs_root}'), quick=True); print(rd); print(pid)"
    )
    t0 = time.time()
    r = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60, env={**os.environ, "PYTHONUTF8": "1"}
    )
    assert r.returncode == 0, r.stderr
    launcher_exit = time.time() - t0
    rd, pid = Path(r.stdout.split()[0]), int(r.stdout.split()[1])
    assert run_state(rd) == "running"  # the dashboard (launcher) is gone, the engine still works
    st = _wait_completed(rd)
    assert st == "completed", (rd / "run.log").read_text(encoding="utf-8")[-2000:]
    m = manifest(rd)
    # the engine wrote its own pid (on Windows the venv python.exe launcher may sit between, so pids can differ)
    assert m["pid"] and m["pid"] != os.getpid() and pid > 0
    assert m["stages"]["5_design"]["status"] in ("done", "cached")
    assert launcher_exit < 30


def test_detached_run_from_button(dash, tmp_path):
    at = _app()
    btn = next(b for b in at.sidebar.button if b.label == "새 실행")
    at.sidebar.checkbox[0].check()
    btn.click().run()
    assert not at.exception, at.exception
    runs = sorted((tmp_path / "runs" / "example_ldh_synth").iterdir())
    new = [d for d in runs if d.name not in {v.name for v in dash.values()}]
    assert len(new) == 1
    assert any("새 실행 시작" in s.value for s in at.success)
    del at  # the dashboard session ends here; the engine keeps running in its own process
    assert _wait_completed(new[0]) == "completed", (new[0] / "run.log").read_text(encoding="utf-8")[-2000:]


def test_peak_figure_listed_for_old_runs(tmp_path):
    """Peak-goal runs before 2026-10-13 have no 5_design/figures.json: the figure is found in the design summary."""
    from enzopt.dashboard import data as D

    rd = tmp_path / "2026-10-11_1200"
    (rd / "5_design").mkdir(parents=True)
    (rd / "manifest.json").write_text(json.dumps({"status": "completed", "stages": {}}), encoding="utf-8")
    fig = {"name": "fig5a_peak", "pdf": "fig5a_peak.pdf", "png": "fig5a_peak.png", "csv": "fig5a_peak.csv"}
    summary = json.dumps({"goal": "peak", "figures": [fig]})
    (rd / "5_design" / "design_summary.json").write_text(summary, encoding="utf-8")
    (rd / "5_design" / "fig5a_peak.png").write_bytes(b"png")
    assert [f["name"] for f in D.stage_figures(D.load_run(rd), "5_design")] == ["fig5a_peak"]
