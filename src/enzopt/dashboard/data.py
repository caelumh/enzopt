"""Read-only access to projects and run folders for the dashboard (no Streamlit import, testable on its own)."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, field
from pathlib import Path

from ..pipeline.runner import FORMAT_VERSION, STAGE_TITLES, STAGES, list_runs, read_json, run_state

REPO = Path(__file__).resolve().parents[3]
STATE_LABEL = {
    "completed": "완료",
    "failed": "실패",
    "running": "진행 중",
    "interrupted": "중단됨",
    "cancelled": "취소",
    "unknown": "알 수 없음",
}


def projects_root() -> Path:
    return Path(os.environ.get("ENZOPT_PROJECTS") or REPO / "projects")


def runs_root() -> Path:
    return Path(os.environ.get("ENZOPT_RUNS") or REPO / "runs")


def list_projects(root: Path | None = None) -> list[Path]:
    root = Path(root or projects_root())
    if not root.exists():
        return []
    return sorted(d for d in root.iterdir() if (d / "project.yaml").exists())


def project_name(project_dir: Path) -> str:
    """The name the runs are filed under (``name:`` in project.yaml, else the folder name)."""
    try:
        from ..core.yamlio import load_yaml

        return str((load_yaml(Path(project_dir) / "project.yaml") or {}).get("name") or Path(project_dir).name)
    except Exception:  # noqa: BLE001 — unreadable yaml: fall back to the folder name
        return Path(project_dir).name


@dataclass
class RunView:
    path: Path
    manifest: dict
    state: str
    progress: dict | None
    old_format: bool
    stages: dict = field(default_factory=dict)

    @property
    def name(self) -> str:
        return self.path.name

    @property
    def label(self) -> str:
        mode = " · 짧은" if self.manifest.get("mode") == "quick" else ""
        return f"{self.name} · {STATE_LABEL.get(self.state, self.state)}{mode}"

    @property
    def banner(self) -> str | None:
        f = self.manifest.get("flags") or {}
        if f.get("guarantee_149") is False:
            return f.get("banner") or "149 보장 없음: 멸종 가능"
        return None

    @property
    def failed_stage(self) -> str | None:
        err = self.manifest.get("error") or {}
        if err.get("stage"):
            return err["stage"]
        for s in STAGES:
            if (self.manifest.get("stages") or {}).get(s, {}).get("status") == "failed":
                return s
        return None

    def failure_line(self) -> str | None:
        if self.state != "failed":
            return None
        st = self.failed_stage
        err = self.manifest.get("error") or {}
        msg = err.get("message") or (self.manifest.get("stages") or {}).get(st, {}).get("error") or "이유 기록 없음"
        return f"{STAGE_TITLES.get(st, st)} 단계 실패: {msg}"

    def stage_status(self, stage: str) -> str:
        return ((self.manifest.get("stages") or {}).get(stage) or {}).get("status", "pending")


def load_run(path: Path) -> RunView:
    path = Path(path)
    m = read_json(path / "manifest.json", {}) or {}
    state = run_state(path) if m else "unknown"
    prog = read_json(path / "progress.json") if state in ("running",) else None
    return RunView(path, m, state, prog, m.get("format_version") != FORMAT_VERSION, m.get("stages") or {})


def project_runs(project_dir: Path, root: Path | None = None) -> list[RunView]:
    d = Path(root or runs_root()) / project_name(project_dir)
    return [load_run(r) for r in list_runs(d)]


def stage_file(run: RunView, stage: str, name: str) -> tuple[Path | None, str | None]:
    """(path, None) when the file is there, else (None, one-line reason)."""
    st = run.stage_status(stage)
    p = run.path / stage / name
    if p.exists() and p.stat().st_size > 0:
        return p, None
    if st in ("pending",):
        return None, "아직 실행되지 않은 단계"
    if st == "running":
        return None, "계산 중"
    if st == "failed":
        return None, "이 단계가 실패해 결과가 없음"
    if st == "cancelled":
        return None, "취소된 단계"
    return None, f"결과 파일이 없거나 깨짐: {stage}/{name}"


def stage_json(run: RunView, stage: str, name: str) -> tuple[dict | None, str | None]:
    p, why = stage_file(run, stage, name)
    if p is None:
        return None, why
    d = read_json(p)
    if d is None:
        return None, f"결과 파일을 읽지 못함(깨짐): {stage}/{name}"
    return d, None


def stage_figures(run: RunView, stage: str) -> list[dict]:
    figs = read_json(run.path / stage / "figures.json", []) or []
    if not figs and stage == "5_design":  # peak-goal runs before 2026-10-13 listed their figure in the summary only
        figs = (read_json(run.path / stage / "design_summary.json", {}) or {}).get("figures") or []
    return [f for f in figs if (run.path / stage / f.get("png", "")).exists()]


def read_table(path: Path) -> list[dict]:
    with open(path, encoding="utf-8") as fh:
        return list(csv.DictReader(fh))
