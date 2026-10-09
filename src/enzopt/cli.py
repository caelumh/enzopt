"""Command line: ``enzopt run | pack | doctor | synth | dashboard``."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def _utf8_stdio() -> None:
    for s in (sys.stdout, sys.stderr):
        try:
            s.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass


def cmd_run(a) -> int:
    from .core.project import ProjectError, load_project
    from .doctor import missing, run_checks
    from .pipeline.runner import STAGES, read_json, run_project

    if not a.resume:
        try:
            proj = load_project(a.project)
        except (ProjectError, FileNotFoundError) as e:
            print(f"설정 오류: {e}")
            return 2
        need_torch = proj.identification.get("method", "pinn") == "pinn"
        miss = missing(run_checks(need_torch=need_torch))
        if miss:
            print(f"실행 중단 — 빠진 구성 요소: {', '.join(miss)} (enzopt doctor로 확인)")
            return 2
    if a.from_stage and a.from_stage not in STAGES:
        print(f"--from-stage는 {STAGES} 중 하나")
        return 2
    rd = run_project(
        a.project,
        quick=a.quick,
        runs_root=a.runs_root,
        use_cache=not a.no_cache,
        resume=a.resume,
        from_stage=a.from_stage,
        stop_after=a.stop_after,
        device=a.device,
        precision=a.precision,
        run_dir=a.run_dir,
    )
    st = (read_json(rd / "manifest.json", {}) or {}).get("status")
    return 0 if st in ("completed", "interrupted") else 1


def cmd_pack(a) -> int:
    from .pack import pack_run

    out = pack_run(Path(a.run_dir), Path(a.output) if a.output else None)
    print(f"묶음: {out}")
    return 0


def cmd_doctor(a) -> int:
    from .doctor import main

    return main(need_hopsan=a.hopsan)


def cmd_synth(a) -> int:
    from .synth import load_config, write_project_data

    res = write_project_data(load_config(a.config), Path(a.project))
    print(f"합성 데이터 작성: {a.project}/data (명목 효소량 {res['Et_nominal_nM']} nM)")
    return 0


def cmd_dashboard(a) -> int:
    import subprocess

    app = Path(__file__).resolve().parent / "dashboard" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app), "--server.headless", "true"]
    if a.port:
        cmd += ["--server.port", str(a.port)]
    return subprocess.call(cmd)


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="enzopt", description="CRNT 판정 → PINN 식별 → 해석·검증 → 강건 설계")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="프로젝트를 한 번 실행(새 실행 폴더)")
    r.add_argument("project", nargs="?", help="프로젝트 폴더(project.yaml이 있는 곳)")
    r.add_argument("--quick", action="store_true", help="짧은 모드(시드 2, 반복 축소)")
    r.add_argument("--runs-root", help="실행 폴더 상위(기본: <저장소>/runs)")
    r.add_argument("--no-cache", action="store_true", help="이전 실행 결과를 재사용하지 않음")
    r.add_argument("--resume", metavar="RUN_DIR", help="중단된 실행 폴더를 이어서 실행")
    r.add_argument("--from-stage", metavar="STAGE", help="--resume과 함께: 이 단계부터 다시(예: 3_fit)")
    r.add_argument("--stop-after", metavar="STAGE", help=argparse.SUPPRESS)
    r.add_argument("--run-dir", help=argparse.SUPPRESS)
    r.add_argument("--device", choices=["auto", "cuda", "cpu"], help="PINN 장치(기본: project.yaml)")
    r.add_argument("--precision", choices=["auto", "float32", "float64"], help="PINN 정밀도(기본: project.yaml)")
    r.set_defaults(func=cmd_run)
    p = sub.add_parser("pack", help="실행 폴더를 제출용 zip 하나로 묶기")
    p.add_argument("run_dir")
    p.add_argument("-o", "--output", help="zip 경로(기본: <실행 폴더>.zip)")
    p.set_defaults(func=cmd_pack)
    d = sub.add_parser("doctor", help="환경 점검")
    d.add_argument("--hopsan", action="store_true", help="Hopsan을 필수로 점검")
    d.set_defaults(func=cmd_doctor)
    s = sub.add_parser("synth", help="현실적 합성 데이터로 프로젝트 data/ 작성")
    s.add_argument("config", help="evals/synthetic.yaml")
    s.add_argument("project", help="프로젝트 폴더")
    s.set_defaults(func=cmd_synth)
    b = sub.add_parser("dashboard", help="Streamlit 대시보드")
    b.add_argument("--port", type=int)
    b.set_defaults(func=cmd_dashboard)
    return ap


def main(argv=None) -> int:
    _utf8_stdio()
    a = build_parser().parse_args(argv)
    if a.cmd == "run" and not a.project and not a.resume:
        print("프로젝트 폴더 또는 --resume <실행 폴더>가 필요함")
        return 2
    return int(a.func(a) or 0)


if __name__ == "__main__":
    raise SystemExit(main())
