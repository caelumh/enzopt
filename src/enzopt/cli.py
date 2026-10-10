"""Command line: ``enzopt run | check | suggest | pack | doctor | synth | dashboard``."""

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


def cmd_check(a) -> int:
    """Stages ①② only, in a temporary folder: verdict + data checks right after measuring (nothing kept in runs/)."""
    import tempfile

    from .core.project import ProjectError, load_project
    from .pipeline.runner import read_json, run_project

    try:
        p = load_project(a.project)
    except (ProjectError, FileNotFoundError) as e:
        print(f"설정 오류: {e}")
        return 2
    missing = [f"{e.name}: {e.file}" for e in p.experiments.values() if e.file is not None and not e.file.exists()]
    with tempfile.TemporaryDirectory(prefix="enzopt_check_") as tmp:
        rd = run_project(a.project, quick=True, runs_root=tmp, use_cache=False, stop_after="2_data")
        m = read_json(rd / "manifest.json", {}) or {}
        print()
        if m.get("error"):
            print(f"검사 실패 — {m['error']['message']}")
            if missing:
                print("아직 없는 파일: " + ", ".join(missing))
            return 1
        v = read_json(rd / "1_verdict" / "verdict.json", {})["networks"]["main"]
        d = read_json(rd / "2_data" / "data.json", {})
        coeff = sum(p.observable_weights().values()) / max(1, len(p.observable_weights()))
        print(f"판정: {v['headline']}")
        from .core.dataio import equilibrium_observable, final_slope

        obs = p.observable["name"]
        print(
            f"{'실험':<8}{'역할':<14}{'점':>5}{'시간(s)':>12}{obs + ' 처음':>12}{obs + ' 끝':>11}{'예상 평형':>10}"
            f"{'차이':>8}{'끝 기울기/분':>12}{'온도 평균±(℃)':>16}"
        )
        for name, x in d["experiments"].items():
            e = p.experiments[name]
            t = x.get("temperature")
            temp = f"{t['mean_C']:.2f}±{t['range_C'] / 2:.2f}" if t else "기록 없음"
            span = f"{x['t_first_s']:g}–{x['t_last_s']:g}"
            a_eq = equilibrium_observable(p, e.initial)
            diff = f"{(x['A_last'] - a_eq) / a_eq * 100:+.1f}%" if a_eq is not None and a_eq > 0.05 * coeff else "—"
            slope = final_slope(rd / "2_data" / f"exp_{name}.csv")
            a_eq_txt = f"{a_eq:.4f}" if a_eq is not None else "—"
            print(
                f"{name:<8}{x['role']:<14}{x['n']:>5}{span:>12}{x['A_first']:>12.4f}{x['A_last']:>11.4f}"
                f"{a_eq_txt:>10}{diff:>8}{slope * 60:>12.4f}{temp:>16}"
            )
        for w in m.get("warnings") or []:
            print(f"[경고] {w}")
        print(
            f"예상 평형 = K' {p.Kp:g}로 푼 전체 반응의 평형(효소에 붙은 양 무시). "
            "같은 적합류 실험의 끝값이 서로 5% 안이고 예상 근처면 정상, 한쪽으로 치우치면 pH·온도부터 의심. "
            "끝 기울기가 0에 가깝지 않으면 아직 평형 전."
        )
    return 0


def cmd_suggest(a) -> int:
    from .pipeline.suggest import run_suggest

    out = run_suggest(Path(a.project), quick=a.quick, verify_top=a.verify, runs_root=a.runs_root)
    print(f"결과: {out}")
    return 0


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
    c = sub.add_parser("check", help="측정 직후 점검: ①판정·②데이터 검사만(임시 폴더, 결과를 남기지 않음)")
    c.add_argument("project", help="프로젝트 폴더")
    c.set_defaults(func=cmd_check)
    g = sub.add_parser("suggest", help="빠진 반응 단계 제안: PINN 보정 항 → 후보 단계 대조 → 고전 피팅 검증")
    g.add_argument("project", help="프로젝트 폴더")
    g.add_argument("--quick", action="store_true", help="짧은 모드(시드 2 × 재시작 2, 검증 축소)")
    g.add_argument("--verify", type=int, default=1, help="고전 피팅으로 확인할 상위 후보 수(0 = 생략)")
    g.add_argument("--runs-root", help="결과 폴더 상위(기본: <저장소>/runs)")
    g.set_defaults(func=cmd_suggest)
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
