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
