"""HopsanCLI helper (ported from CADKIMM ``hopsan/cli.py``).

Search order for the executable:
  1. env ``HOPSANCLI`` — full command (split on spaces), e.g. ``flatpak run --command=hopsancli <app id>``
  2. env ``HOPSAN_HOME`` → ``bin/hopsancli(.exe)``
  3. the enzopt config file (``ENZOPT_CONFIG`` or ``<repo>/enzopt.config.yaml``), key ``hopsan.cli``
  4. ``hopsancli`` / ``hopsan.cli`` on PATH
  5. portable copies under ``~/tools/Hopsan-*/bin``
"""

from __future__ import annotations

import os
import shlex
import shutil
import subprocess
from pathlib import Path

from .results import read_species

REPO_ROOT = Path(__file__).resolve().parents[3]


class HopsanNotFound(FileNotFoundError):
    pass


class HopsanError(RuntimeError):
    pass


def _config_cli() -> str | None:
    cfg = os.environ.get("ENZOPT_CONFIG") or str(REPO_ROOT / "enzopt.config.yaml")
    if not Path(cfg).exists():
        return None
    try:
        from ..core.yamlio import load_yaml

        data = load_yaml(cfg) or {}
        return (data.get("hopsan") or {}).get("cli")
    except Exception:  # noqa: BLE001 — a broken config file just means "not configured"
        return None


def find_cli() -> list[str]:
    env = os.environ.get("HOPSANCLI")
    if env is not None:
        cmd = shlex.split(env, posix=(os.name != "nt"))
        if cmd and (Path(cmd[0]).exists() or shutil.which(cmd[0])):
            return cmd
        raise HopsanNotFound(f"HOPSANCLI가 가리키는 실행 파일이 없음: {env}")
    cands = []
    home = os.environ.get("HOPSAN_HOME")
    if home:
        cands += [os.path.join(home, "bin", "hopsancli.exe"), os.path.join(home, "bin", "hopsancli")]
    c = _config_cli()
    if c:
        cands.append(c)
    for name in ("hopsancli", "hopsan.cli", "hopsancli.exe"):
        w = shutil.which(name)
        if w:
            cands.append(w)
    tools = Path.home() / "tools"
    if tools.exists():
        cands += [str(p) for p in sorted(tools.glob("Hopsan-*/bin/hopsancli*"), reverse=True)]
    for c in cands:
        if c and os.path.exists(c):
            return [c]
    raise HopsanNotFound(
        "hopsancli를 찾지 못함. HOPSANCLI 또는 HOPSAN_HOME을 지정하거나 enzopt.config.yaml의 hopsan.cli를 고치세요."
    )


def version() -> str | None:
    try:
        r = subprocess.run(find_cli() + ["--version"], capture_output=True, text=True, timeout=60)
        return (r.stdout or r.stderr).strip().splitlines()[-1] if (r.stdout or r.stderr) else None
    except (HopsanNotFound, OSError, subprocess.SubprocessError):
        return None


def run_model(hmf, out_csv, param_csv=None, sim="hmf", nsamples=None, timeout=3600):
    """Run HopsanCLI on ``hmf`` and return ``(t, {species: array})`` from the full-results CSV."""
    cmd = find_cli() + ["-m", str(hmf), "-s", sim, "--resultsFullCSV", str(out_csv), "--silent"]
    if param_csv:
        cmd += ["--parameterImport", str(param_csv)]
    if nsamples:
        cmd += ["-l", str(int(nsamples))]
    r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout, encoding="utf-8", errors="replace")
    if r.returncode != 0 or "Error" in (r.stdout or ""):
        raise HopsanError("HopsanCLI 실패:\n" + (r.stdout or "")[-1500:] + (r.stderr or "")[-500:])
    return read_species(out_csv)
