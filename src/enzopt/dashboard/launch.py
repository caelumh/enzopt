"""Start the engine as a separate, detached process (SC5-10·11): closing the dashboard does not stop a run."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

from ..pipeline.runner import FORMAT_VERSION, STAGE_TITLES, STAGES, new_run_dir, now_iso, write_json

REPO = Path(__file__).resolve().parents[3]


def _popen(cmd: list[str], log: Path) -> subprocess.Popen:
    fh = open(log, "a", encoding="utf-8")  # noqa: SIM115 — handed to the child process
    env = dict(os.environ, PYTHONUTF8="1", PYTHONIOENCODING="utf-8")
    kw = dict(stdout=fh, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL, cwd=str(REPO), env=env, close_fds=True)
    if os.name == "nt":
        DETACHED_PROCESS = 0x00000008
        CREATE_NEW_PROCESS_GROUP = 0x00000200
        CREATE_NO_WINDOW = 0x08000000
        CREATE_BREAKAWAY_FROM_JOB = 0x01000000
        base = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
        try:  # leave the dashboard's job object too, when the job allows it
            return subprocess.Popen(cmd, creationflags=base | CREATE_BREAKAWAY_FROM_JOB, **kw)
        except OSError:
            return subprocess.Popen(cmd, creationflags=base, **kw)
    return subprocess.Popen(cmd, start_new_session=True, **kw)


def start_run(
    project_dir: Path, runs_root: Path, quick: bool = False, project_name: str | None = None, hopsan: bool = True
) -> tuple[Path, int]:
    """Create the run folder now (so the dashboard can show it at once) and start ``enzopt run`` detached."""
    from .data import project_name as pname

    name = project_name or pname(project_dir)
    rd = new_run_dir(Path(runs_root), name)
    placeholder = {
        "format_version": FORMAT_VERSION,
        "project": {"name": name, "dir": str(project_dir)},
        "mode": "quick" if quick else "full",
        "created": now_iso(),
        "status": "running",
        "starting": True,
        "pid": None,
        "stages": {s: {"status": "pending", "title": STAGE_TITLES[s]} for s in STAGES},
        "flags": {},
        "warnings": [],
    }
    write_json(rd / "manifest.json", placeholder)
    cmd = [sys.executable, "-m", "enzopt", "run", str(project_dir), "--runs-root", str(runs_root), "--run-dir", str(rd)]
    if quick:
        cmd.append("--quick")
    if not hopsan:
        cmd.append("--no-hopsan")
    p = _popen(cmd, rd / "run.log")
    from ..pipeline.runner import read_json

    m = read_json(rd / "manifest.json", {}) or {}
    if m.get("starting"):  # the engine has not replaced the placeholder yet: record its process id
        m["pid"] = p.pid
        write_json(rd / "manifest.json", m)
    return rd, p.pid


def resume_run(run_dir: Path, from_stage: str | None = None) -> int:
    """Continue an interrupted/failed run in the same folder (optionally redo from a stage)."""
    cmd = [sys.executable, "-m", "enzopt", "run", "--resume", str(run_dir)]
    if from_stage:
        cmd += ["--from-stage", from_stage]
    return _popen(cmd, Path(run_dir) / "run.log").pid


def cancel_run(run_dir: Path) -> None:
    """Ask the engine to stop at its next progress check (it writes status 'cancelled')."""
    (Path(run_dir) / "CANCEL").write_text("cancel", encoding="utf-8")
