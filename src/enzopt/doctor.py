"""``enzopt doctor``: check Python, packages, CUDA, Hopsan and disk space; name whatever is missing."""

from __future__ import annotations

import importlib
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path

REQUIRED = ["numpy", "scipy", "matplotlib"]
OPTIONAL = {"torch": "PINN 식별(③)", "yaml": "PyYAML(없으면 동봉 순수 Python 판)", "streamlit": "대시보드"}
MIN_DISK_GB = 2.0


@dataclass
class Check:
    name: str
    ok: bool
    detail: str
    required: bool = True

    def line(self) -> str:
        mark = "OK  " if self.ok else ("없음" if self.required else "주의")
        return f"[{mark}] {self.name:<12} {self.detail}"


def _version(mod: str) -> str | None:
    try:
        m = importlib.import_module(mod)
        return getattr(m, "__version__", "?")
    except Exception:  # noqa: BLE001 — any import failure means "missing"
        return None


def run_checks(
    need_torch: bool = True, need_hopsan: bool = False, need_streamlit: bool = False, path: Path | None = None
):
    checks = []
    py = sys.version.split()[0]
    checks.append(Check("python", sys.version_info[:2] == (3, 12), f"{py} (3.12 필요)"))
    for m in REQUIRED:
        v = _version(m)
        checks.append(Check(m, v is not None, v or "설치되지 않음"))
    for m, why in OPTIONAL.items():
        v = _version(m)
        req = (m == "torch" and need_torch) or (m == "streamlit" and need_streamlit)
        checks.append(Check(m, v is not None, (v or "설치되지 않음") + f" — {why}", required=req))
    cuda_ok, cuda_detail = False, "torch 없음"
    if _version("torch"):
        import torch

        cuda_ok = torch.cuda.is_available()
        cuda_detail = (
            f"{torch.cuda.get_device_name(0)} (CUDA {torch.version.cuda})"
            if cuda_ok
            else "사용 불가 → CPU(float64)로 전환"
        )
    checks.append(Check("cuda", cuda_ok, cuda_detail, required=False))
    try:
        from .hopsan.cli import find_cli, version

        cli = find_cli()
        checks.append(Check("hopsan", True, f"{' '.join(cli)} ({version() or '버전 확인 실패'})", required=need_hopsan))
    except Exception as e:  # noqa: BLE001
        checks.append(Check("hopsan", False, f"찾지 못함 → scipy로 대신(Hopsan 미사용). {e}", required=need_hopsan))
    target = Path(path) if path else Path.cwd()
    try:
        free = shutil.disk_usage(target).free / 1e9
        checks.append(Check("disk", free >= MIN_DISK_GB, f"여유 {free:.1f} GB ({target.anchor or target})"))
    except OSError as e:
        checks.append(Check("disk", False, f"확인 실패: {e}", required=False))
    return checks


def missing(checks) -> list[str]:
    return [c.name for c in checks if c.required and not c.ok]


def main(need_hopsan: bool = False) -> int:
    checks = run_checks(need_hopsan=need_hopsan)
    print("enzopt doctor")
    for c in checks:
        print(" " + c.line())
    miss = missing(checks)
    if miss:
        print(f"빠진 구성 요소: {', '.join(miss)}")
        return 1
    print("점검 통과")
    return 0
