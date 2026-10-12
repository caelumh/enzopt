"""Run orchestration: run folder, manifest, stages ①–⑤, progress, cancel, cache and resume.

Run folder ``runs/<project>/<YYYY-MM-DD_HHMM>/`` holds ``manifest.json`` (config copy, input hashes, code version,
stage states), ``progress.json`` (for the dashboard), ``project.yaml`` (copy), ``inputs/`` (input data copies)
and one folder per stage (``1_verdict`` … ``5_design``). A stage whose key (config + inputs + upstream keys +
code hash) matches a finished stage of an earlier run of the same project is copied instead of recomputed.
"""

from __future__ import annotations

import datetime as _dt
import json
import os
import shutil
import socket
import subprocess
import sys
import time
import traceback
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

from .. import __version__
from ..core.project import Project, file_sha256, load_project, stable_hash

FORMAT_VERSION = 1
STAGES = ["1_verdict", "2_data", "3_fit", "4_sim", "5_design"]
STAGE_TITLES = {
    "1_verdict": "① 반응망",
    "2_data": "② 데이터",
    "3_fit": "③ 식별",
    "4_sim": "④ 해석",
    "5_design": "⑤ 설계",
}
REPO_ROOT = Path(__file__).resolve().parents[3]


class Cancelled(Exception):
    pass


class StageFailed(Exception):
    pass


# ------------------------------------------------------------------------------------------------ helpers
def now_iso() -> str:
    return _dt.datetime.now().astimezone().isoformat(timespec="seconds")


def write_json(path: Path, obj, best_effort: bool = False) -> None:
    """Atomic JSON write (the dashboard may read while we write).

    On Windows the final rename fails with PermissionError while another process (dashboard, virus scanner, indexer)
    holds the target open, so it is retried for up to ~2 s. ``best_effort`` writes (progress) give up silently.
    """
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=1, ensure_ascii=False, default=_json_default)
    for attempt in range(40):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 39:
                try:
                    tmp.unlink()
                except OSError:
                    pass
                if best_effort:
                    return
                raise
            time.sleep(0.05)


def _json_default(o):
    try:
        import numpy as np

        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, np.bool_):
            return bool(o)
    except ImportError:  # pragma: no cover
        pass
    if isinstance(o, Path):
        return str(o)
    return str(o)


def read_json(path: Path, default=None):
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        return default


_CODE_HASH: str | None = None


def code_hash() -> str:
    """Hash of the enzopt source files: a code change invalidates cached stages."""
    global _CODE_HASH
    if _CODE_HASH is None:
        import hashlib

        h = hashlib.sha256()
        pkg = Path(__file__).resolve().parents[1]
        for f in sorted(pkg.rglob("*.py")):
            if "_vendor" in f.parts or "dashboard" in f.parts:
                continue
            h.update(str(f.relative_to(pkg)).encode())
            h.update(f.read_bytes())
        _CODE_HASH = h.hexdigest()[:16]
    return _CODE_HASH


def git_info() -> dict:
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
        ).stdout.strip()
        dirty = bool(
            subprocess.run(
                ["git", "status", "--porcelain", "--", "src"], cwd=REPO_ROOT, capture_output=True, text=True, timeout=10
            ).stdout.strip()
        )
        return {"commit": commit or None, "dirty_src": dirty}
    except (OSError, subprocess.SubprocessError):
        return {"commit": None, "dirty_src": None}


def lock_hash() -> str | None:
    lock = REPO_ROOT / "uv.lock"
    return file_sha256(lock) if lock.exists() else None


def default_runs_root(project: Project) -> Path:
    env = os.environ.get("ENZOPT_RUNS")
    if env:
        return Path(env)
    if project.root.parent.name == "projects":
        return project.root.parent.parent / "runs"
    return Path.cwd() / "runs"


def new_run_dir(runs_root: Path, project_name: str) -> Path:
    base = Path(runs_root) / project_name
    base.mkdir(parents=True, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y-%m-%d_%H%M")
    d, i = base / stamp, 2
    while True:
        try:
            d.mkdir()
            return d
        except FileExistsError:
            d = base / f"{stamp}_{i}"
            i += 1


def list_runs(project_runs_dir: Path) -> list[Path]:
    """Run folders of one project, newest first."""
    p = Path(project_runs_dir)
    if not p.exists():
        return []
    runs = [d for d in p.iterdir() if d.is_dir() and (d / "manifest.json").exists()]
    return sorted(runs, key=lambda d: (read_json(d / "manifest.json", {}) or {}).get("created", d.name), reverse=True)


def pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    if os.name == "nt":
        import ctypes

        PROCESS_QUERY_LIMITED_INFORMATION = 0x1000
        h = ctypes.windll.kernel32.OpenProcess(PROCESS_QUERY_LIMITED_INFORMATION, False, int(pid))
        if not h:
            return False
        code = ctypes.c_ulong()
        ok = ctypes.windll.kernel32.GetExitCodeProcess(h, ctypes.byref(code))
        ctypes.windll.kernel32.CloseHandle(h)
        return bool(ok) and code.value == 259  # STILL_ACTIVE
    try:
        os.kill(int(pid), 0)
        return True
    except OSError:
        return False


def run_state(run_dir: Path) -> str:
    """completed | failed | cancelled | running | interrupted (running but the process is gone)."""
    m = read_json(Path(run_dir) / "manifest.json", {}) or {}
    st = m.get("status", "unknown")
    if st == "running" and m.get("starting"):  # placeholder written by the dashboard before the engine started
        try:
            age = (_dt.datetime.now().astimezone() - _dt.datetime.fromisoformat(m["created"])).total_seconds()
        except (KeyError, ValueError):
            age = 1e9
        return "running" if age < 120 or pid_alive(m.get("pid")) else "interrupted"
    if st == "running" and not pid_alive(m.get("pid")):
        return "interrupted"
    return st


class _KeepAwake:
    """Prevent Windows sleep while a run is active (ES_CONTINUOUS | ES_SYSTEM_REQUIRED)."""

    def __enter__(self):
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
            except (OSError, AttributeError):
                pass
        return self

    def __exit__(self, *exc):
        if os.name == "nt":
            try:
                import ctypes

                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            except (OSError, AttributeError):
                pass
        return False


# ------------------------------------------------------------------------------------------------ context
@dataclass
class RunContext:
    project: Project
    run_dir: Path
    quick: bool
    settings: dict
    manifest: dict
    options: dict = field(default_factory=dict)
    _stage: str | None = None
    _stage_t0: float = 0.0

    def stage_dir(self, name: str | None = None) -> Path:
        d = self.run_dir / (name or self._stage)
        d.mkdir(parents=True, exist_ok=True)
        return d

    def save_manifest(self) -> None:
        write_json(self.run_dir / "manifest.json", self.manifest)

    def warn(self, msg: str) -> None:
        if msg not in self.manifest["warnings"]:
            self.manifest["warnings"].append(msg)
            if self._stage and self._stage in self.manifest.get("stages", {}):
                self.manifest["stages"][self._stage].setdefault("warnings", []).append(msg)
            self.save_manifest()
        print(f"[경고] {msg}", flush=True)

    def flag(self, key: str, value) -> None:
        self.manifest["flags"][key] = value
        self.save_manifest()

    def check_cancel(self) -> None:
        if (self.run_dir / "CANCEL").exists():
            raise Cancelled("사용자 취소")

    def progress(self, fraction: float, message: str = "") -> None:
        """Report progress inside the current stage (0..1) with a simple remaining-time estimate."""
        self.check_cancel()
        fraction = max(0.0, min(1.0, float(fraction)))
        el = time.time() - self._stage_t0
        eta = el * (1 - fraction) / fraction if fraction > 0.02 else None
        i = STAGES.index(self._stage) if self._stage in STAGES else 0
        write_json(
            self.run_dir / "progress.json",
            {
                "stage": self._stage,
                "stage_title": STAGE_TITLES.get(self._stage, self._stage),
                "stage_index": i + 1,
                "n_stages": len(STAGES),
                "fraction": fraction,
                "overall_fraction": (i + fraction) / len(STAGES),
                "message": message,
                "elapsed_s": el,
                "eta_s": eta,
                "updated": now_iso(),
                "pid": os.getpid(),
            },
            best_effort=True,
        )

    def read_stage_json(self, stage: str, name: str):
        p = self.run_dir / stage / name
        d = read_json(p)
        if d is None:
            raise StageFailed(f"이전 단계 결과를 읽지 못함: {stage}/{name}")
        return d


# ------------------------------------------------------------------------------------------------ keys
def stage_key(ctx: RunContext, stage: str, upstream: dict) -> str:
    p, s = ctx.project, ctx.settings
    common = {"stage": stage, "code": code_hash(), "quick": ctx.quick, "upstream": upstream}
    if stage == "1_verdict":
        cfg = {
            "network": p.raw.get("network"),
            "control": p.raw.get("control"),
            "design_dead_end": s["design"].get("dead_end"),
            "hopsan": [s["simulation"]["hopsan"].get("enabled", True), _hopsan_env_signature()],
        }
    elif stage == "2_data":
        cfg = {
            "experiments": p.raw.get("experiments"),
            "files": {str(f.name): file_sha256(f) for f in p.input_files()[1:] if f.exists()},
            "network": p.raw.get("network"),
        }
    elif stage == "3_fit":
        cfg = {
            "identification": s["identification"],
            "options": {k: ctx.options.get(k) for k in ("device", "precision")},
        }
    elif stage == "4_sim":
        cfg = {"simulation": s["simulation"], "hopsan_env": _hopsan_env_signature()}
    else:
        cfg = {"design": s["design"]}
    return stable_hash({**common, "cfg": cfg})[:20]


def _settings(project: Project, quick: bool, options: dict) -> dict:
    """project.yaml settings with the run options applied (``--no-hopsan``)."""
    s = project.settings(quick)
    if options.get("hopsan") is False:
        s["simulation"]["hopsan"]["enabled"] = False
    return s


def _hopsan_env_signature() -> str:
    try:
        from ..hopsan.cli import find_cli

        return " ".join(find_cli())
    except Exception:  # noqa: BLE001 — any failure means "not found"
        return "none"


def _find_cached(ctx: RunContext, stage: str, key: str) -> Path | None:
    for d in list_runs(ctx.run_dir.parent):
        if d == ctx.run_dir:
            continue
        m = read_json(d / "manifest.json", {}) or {}
        st = (m.get("stages") or {}).get(stage) or {}
        if st.get("key") == key and st.get("status") in ("done", "cached") and (d / stage).is_dir():
            return d
    return None


# ------------------------------------------------------------------------------------------------ runner
def _stage_functions() -> dict[str, Callable[[RunContext], None]]:
    from . import stages

    return {
        "1_verdict": stages.stage_verdict,
        "2_data": stages.stage_data,
        "3_fit": stages.stage_fit,
        "4_sim": stages.stage_sim,
        "5_design": stages.stage_design,
    }


def _init_manifest(project: Project, run_dir: Path, quick: bool, options: dict) -> dict:
    inputs = {}
    for f in project.input_files():
        if f.exists():
            inputs[str(f.relative_to(project.root)).replace("\\", "/")] = file_sha256(f)
    return {
        "format_version": FORMAT_VERSION,
        "enzopt_version": __version__,
        "code_hash": code_hash(),
        "git": git_info(),
        "uv_lock_sha256": lock_hash(),
        "python": sys.version.split()[0],
        "project": {"name": project.name, "dir": str(project.root), "description": project.description},
        "inputs": inputs,
        "mode": "quick" if quick else "full",
        "options": options,
        "created": now_iso(),
        "finished": None,
        "status": "running",
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "stages": {s: {"status": "pending", "title": STAGE_TITLES[s]} for s in STAGES},
        "device": {},
        "seeds": [],
        "flags": {},
        "warnings": [],
        "error": None,
    }


def _copy_inputs(project: Project, run_dir: Path) -> None:
    """Settings copy at the top and a self-contained project copy in inputs/ (used by resume and pack)."""
    shutil.copy2(project.root / "project.yaml", run_dir / "project.yaml")
    dst = run_dir / "inputs"
    dst.mkdir(parents=True, exist_ok=True)
    shutil.copy2(project.root / "project.yaml", dst / "project.yaml")
    for f in project.input_files()[1:]:
        if f.exists():
            rel = f.relative_to(project.root)
            (dst / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dst / rel)


def run_project(
    project_dir: str | Path,
    quick: bool = False,
    runs_root: str | Path | None = None,
    use_cache: bool = True,
    resume: str | Path | None = None,
    from_stage: str | None = None,
    stop_after: str | None = None,
    device: str | None = None,
    precision: str | None = None,
    run_dir: str | Path | None = None,
    hopsan: bool | None = None,
) -> Path:
    """Run stages ①–⑤ for a project and return the run folder.

    ``resume``: continue an earlier run folder (skip its finished stages; with ``from_stage`` redo from there).
    ``stop_after``: stop after that stage with status ``interrupted`` (used to test resume).
    ``run_dir``: use this pre-created empty folder (the dashboard creates it so it can show it at once).
    ``hopsan``: False turns the Hopsan comparison of stage ④ off for this run (``enzopt run --no-hopsan``); None follows
    project.yaml (``simulation.hopsan.enabled``, on by default).
    """
    options = {"device": device, "precision": precision, "hopsan": hopsan}
    if resume:
        rd = Path(resume)
        man = read_json(rd / "manifest.json")
        if man is None:
            raise FileNotFoundError(f"manifest.json이 없는 실행 폴더: {rd}")
        project = load_project(rd / "inputs")  # the run's own copy of project.yaml + data
        quick = man.get("mode") == "quick"
        options = man.get("options") or options
        man["status"] = "running"
        man["pid"] = os.getpid()
        man["error"] = None
        man.setdefault("resumed", []).append(now_iso())
        ctx = RunContext(project, rd, quick, _settings(project, quick, options), man, options)
        if from_stage:
            for s in STAGES[STAGES.index(from_stage) :]:
                man["stages"][s]["status"] = "pending"
                man["stages"][s].pop("key", None)
    else:
        project = load_project(project_dir)
        root = Path(runs_root) if runs_root else default_runs_root(project)
        if run_dir:
            rd = Path(run_dir)
            rd.mkdir(parents=True, exist_ok=True)
        else:
            rd = new_run_dir(root, project.name)
        man = _init_manifest(project, rd, quick, options)
        ctx = RunContext(project, rd, quick, _settings(project, quick, options), man, options)
        _copy_inputs(project, rd)
        man["settings"] = ctx.settings
    ctx.save_manifest()
    (rd / "CANCEL").unlink(missing_ok=True)
    funcs = _stage_functions()
    upstream: dict = {}
    status = "completed"
    print(f"실행 폴더: {rd}", flush=True)
    with _KeepAwake():
        for stage in STAGES:
            st = man["stages"][stage]
            ctx._stage = stage
            ctx._stage_t0 = time.time()
            key = stage_key(ctx, stage, dict(upstream))
            if st.get("status") in ("done", "cached") and st.get("key") == key and (rd / stage).is_dir():
                print(f"{STAGE_TITLES[stage]}: 이미 완료(건너뜀)", flush=True)
                upstream[stage] = key
                if stop_after == stage:
                    status = "interrupted"
                    break
                continue
            st.update({"key": key, "started": now_iso(), "finished": None, "error": None, "cached_from": None})
            st.pop("warnings", None)
            cached = _find_cached(ctx, stage, key) if use_cache else None
            if cached is not None:
                if (rd / stage).exists():
                    shutil.rmtree(rd / stage)
                shutil.copytree(cached / stage, rd / stage)
                old = read_json(cached / "manifest.json", {}) or {}
                for k, v in (old.get("flags") or {}).items():
                    man["flags"].setdefault(k, v)
                if stage == "3_fit":
                    man["device"] = old.get("device", {})
                    man["seeds"] = old.get("seeds", [])
                # only this stage's warnings (runs before 2026-10-12 did not record them per stage: all of them)
                old_st = (old.get("stages") or {}).get(stage) or {}
                carried = old_st.get("warnings") if "warnings" in old_st else old.get("warnings") or []
                for w in carried:
                    if w not in man["warnings"]:
                        man["warnings"].append(w)
                if carried:
                    st["warnings"] = list(carried)
                st.update(
                    {"status": "cached", "cached_from": str(cached.name), "finished": now_iso(), "elapsed_s": 0.0}
                )
                ctx.save_manifest()
                print(f"{STAGE_TITLES[stage]}: 캐시 사용({cached.name})", flush=True)
                upstream[stage] = key
                if stop_after == stage:
                    status = "interrupted"
                    break
                continue
            if (rd / stage).exists():
                shutil.rmtree(rd / stage)
            st["status"] = "running"
            ctx.save_manifest()
            print(f"{STAGE_TITLES[stage]}: 시작", flush=True)
            ctx.progress(0.0, "시작")
            try:
                funcs[stage](ctx)
            except Cancelled:
                st.update({"status": "cancelled", "finished": now_iso()})
                status = "cancelled"
                ctx.save_manifest()
                print(f"{STAGE_TITLES[stage]}: 취소", flush=True)
                break
            except Exception as e:  # noqa: BLE001 — record any stage failure and stop
                st.update({"status": "failed", "finished": now_iso(), "error": f"{type(e).__name__}: {e}"})
                man["error"] = {
                    "stage": stage,
                    "message": str(e),
                    "type": type(e).__name__,
                    "traceback": traceback.format_exc()[-4000:],
                }
                status = "failed"
                ctx.save_manifest()
                print(f"{STAGE_TITLES[stage]}: 실패 — {e}", flush=True)
                break
            st.update({"status": "done", "finished": now_iso(), "elapsed_s": round(time.time() - ctx._stage_t0, 2)})
            upstream[stage] = key
            ctx.save_manifest()
            ctx.progress(1.0, "완료")
            print(f"{STAGE_TITLES[stage]}: 완료 ({st['elapsed_s']:.1f} s)", flush=True)
            if stop_after == stage:
                status = "interrupted"
                break
    man["status"] = status
    man["finished"] = now_iso()
    man["size_bytes"] = sum(f.stat().st_size for f in rd.rglob("*") if f.is_file())
    ctx.save_manifest()
    print(f"상태: {status} · 실행 폴더 크기 {man['size_bytes'] / 1e6:.1f} MB", flush=True)
    return rd
