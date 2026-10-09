import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE = ROOT / "projects" / "example_ldh_synth"
EVALS = ROOT / "evals"

LDH_SPECIES = ["E", "NADH", "EN", "Pyr", "ENP", "EQL", "Lac", "EQ", "NAD"]
LDH_STEPS = ["E + NADH <=> EN", "EN + Pyr <=> ENP", "ENP <=> EQL", "EQL <=> EQ + Lac", "EQ <=> E + NAD"]
LDH_ENZYME = ["E", "EN", "ENP", "EQL", "EQ"]


def baselines() -> dict:
    with open(EVALS / "baselines.json", encoding="utf-8") as fh:
        return json.load(fh)


def run_cli(*args, cwd=ROOT, env=None, timeout=1800) -> subprocess.CompletedProcess:
    e = dict(os.environ)
    e["PYTHONUTF8"] = "1"
    if env:
        e.update(env)
    return subprocess.run(
        [sys.executable, "-m", "enzopt", *map(str, args)],
        cwd=cwd,
        env=e,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        timeout=timeout,
    )


@pytest.fixture(scope="session")
def example_dir() -> Path:
    return EXAMPLE


@pytest.fixture(scope="session")
def base() -> dict:
    return baselines()


def run_dir_from(stdout: str) -> Path:
    for line in stdout.splitlines():
        if line.startswith("실행 폴더:"):
            return Path(line.split(":", 1)[1].strip())
    raise AssertionError("실행 폴더 줄이 없음:\n" + stdout[-2000:])


def manifest(run_dir: Path) -> dict:
    with open(Path(run_dir) / "manifest.json", encoding="utf-8") as fh:
        return json.load(fh)


@pytest.fixture(scope="session")
def runs_root(tmp_path_factory) -> Path:
    return tmp_path_factory.mktemp("runs")


@pytest.fixture(scope="session")
def quick_run(runs_root) -> Path:
    """One `enzopt run projects/example_ldh_synth --quick` through the command line, shared by the quick tests."""
    r = run_cli("run", EXAMPLE, "--quick", "--runs-root", runs_root)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return run_dir_from(r.stdout)


@pytest.fixture(scope="session")
def full_run(tmp_path_factory) -> Path:
    """Full-mode run of the example (long tier). The harness passes its timed Q1 run via ENZOPT_LONG_RUN_DIR."""
    env = os.environ.get("ENZOPT_LONG_RUN_DIR")
    if env and (Path(env) / "manifest.json").exists():
        return Path(env)
    root = tmp_path_factory.mktemp("runs_full")
    r = run_cli("run", EXAMPLE, "--no-cache", "--runs-root", root, timeout=3600)
    assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
    return run_dir_from(r.stdout)
