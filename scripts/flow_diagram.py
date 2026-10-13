"""System-flow diagram of docs/flow/README.md: stage computation → run-folder files → dashboard tab, and the files each
stage reads from earlier stages. Writes docs/flow/system.svg (no third-party packages).

    uv run python scripts/flow_diagram.py
"""

from __future__ import annotations

from html import escape
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "flow" / "system.svg"

W, H = 1100, 1170
FONT = "Pretendard, 'Noto Sans KR', 'Malgun Gothic', 'Apple SD Gothic Neo', sans-serif"
MONO = "Consolas, 'D2Coding', monospace"
BLUE, GREEN, GREY, PURPLE = "#0072B2", "#009E73", "#555555", "#A0527F"
COLS = {"calc": (50, 350), "files": (440, 290), "tab": (770, 310)}  # x, width
ROW0, PITCH, BOX_H = 222, 175, 150

ROWS = [
    {
        "calc": (
            "① 반응망 판정 (CRNT)",
            [
                "복합체 · 연결 성분 · 강연결 성분",
                "약가역 · 결핍 = 복합체 − 연결 성분 − 랭크",
                "보존 법칙(정수 기저) · 등급 A / B / C",
                "식별 · 대조 · 설계 반응망을 각각 판정",
            ],
            "읽음: project.yaml의 network · control · design",
        ),
        "files": (
            "1_verdict/",
            ["verdict.json · report.txt", "fig1a · fig1b · fig1c", "→ manifest 플래그 guarantee_149"],
        ),
        "tab": (
            "① 반응망",
            [
                "판정 한 줄 · 보장/전제 상자",
                "판정 표 (반응망마다 한 줄)",
                "그림 1a · 1c · 1b · 펼침: 판정 보고",
                "약가역 아니면 모든 탭에 빨간 띠",
            ],
        ),
    },
    {
        "calc": (
            "② 데이터 검사",
            [
                "실험마다 CSV 읽기 (format이면 변환)",
                "검사: 머리글 · 숫자 · 시간 역행 · A340 ≤ 2",
                "처음 값 예상 · 온도 변동 ≥ 0.2 ℃면 경고",
                "학습 실험이 하나도 없으면 실패",
            ],
            "읽음: project.yaml의 experiments + data/",
        ),
        "files": (
            "2_data/",
            ["exp_<실험>.csv (time_s, 관측값)", "temp_<실험>.csv", "sig_<실험>__<신호>.csv", "data.json · fig2_data"],
        ),
        "tab": ("② 데이터", ["실험 표 (역할·점 수·처음/끝·Et·온도)", "그림 2: (a) 진행 곡선 (b) 온도"]),
    },
    {
        "calc": (
            "③ 식별",
            [
                "PINN: 시드 × 재시작 8, GPU에서 함께 학습",
                "재시작마다 ODE로 다시 풀어 최선 → 표본",
                "학습 RMSE > 3σ → 제외 (남은 < 2 → ⑤ 멈춤)",
                "비교 모델: 고전 피팅(최소제곱) · 대조 PINN",
            ],
            "읽음: 2_data의 exp_·sig_*.csv",
        ),
        "files": (
            "3_fit/",
            [
                "samples/sample_seed<N>.json",
                "baseline/ · control/",
                "fit_summary.json",
                "holdout_comparison.csv",
                "predictions.csv · fig3a~3f",
            ],
        ),
        "tab": (
            "③ 식별",
            [
                "지표: 표본(남음/시드) · 장치",
                "      학습 시간 · 홀드아웃 RMSE",
                "제외 사유 · 펼침: 홀드아웃 비교 표",
                "그림 3a~3d (+3e 실활, 3f 신호 여럿)",
            ],
        ),
    },
    {
        "calc": (
            "④ 해석 · CRNT 자동 검증",
            [
                "표본 × (같은 적합류 12 + 다른 1) · scipy",
                "검증: 보존 · 양수 · |ln Q/K| · 리아푸노프",
                "걸리면 Δt/10으로 한 번 더 (수치 오류 의심)",
                "Hopsan: 1단계 배치 + 대표 완전 메커니즘",
            ],
            "읽음: 3_fit/samples · 플래그 guarantee_149",
        ),
        "files": (
            "4_sim/",
            [
                "validation.csv · trajectories.csv",
                "sim_summary.json",
                "hopsan/ (*.hmf 모델 · *.csv 결과)",
                "fig4a · fig4b · fig4c",
            ],
        ),
        "tab": (
            "④ 해석",
            [
                "지표: 검증 통과 · 재실행 · Hopsan 차이",
                "Hopsan을 못 쓰면 노란 경고",
                "그림 4a · 4b · 4c",
                "펼침: 검증 표 (실행마다 한 줄)",
            ],
        ),
    },
    {
        "calc": (
            "⑤ 강건 설계",
            [
                "평형 한계 ρ_min (약가역 · 결핍 0 · K′)",
                "ρ마다 최소 효소량: 공칭(표본 1) · 강건(전부)",
                "ρ* = 강건 최소 → 두 설계를 모든 표본에 확인",
                "피펫 표 · 설계 조건 예측 곡선 (+ Hopsan)",
            ],
            "읽음: fit_summary · samples · design-check 실험",
        ),
        "files": (
            "5_design/",
            [
                "design_summary.json",
                "pipette.csv · pipette.txt",
                "design_table.csv · design_check.csv",
                "equilibrium_limit.csv · hopsan/",
                "fig5a~5d (peak: fig5a_peak)",
            ],
        ),
        "tab": (
            "⑤ 설계",
            [
                "지표: ρ* · Et* · E₀ 배수 · 공칭 Et(미달)",
                "평형 한계 ρ_min 한 줄",
                "피펫 표 + CSV 내려받기",
                "그림 5a~5d (목표 peak: 5a_peak)",
            ],
        ),
    },
]

STYLE = {  # stroke, header fill, body fill
    "calc": (BLUE, "#d9e8f5", "#ffffff"),
    "files": ("#999999", "#ebebeb", "#f9f9f9"),
    "tab": (GREEN, "#dff3ec", "#ffffff"),
}


def text(x, y, s, size=12, weight=None, fill="#222", anchor=None, family=None, extra="") -> str:
    a = f' font-size="{size}" fill="{fill}"'
    if weight:
        a += f' font-weight="{weight}"'
    if anchor:
        a += f' text-anchor="{anchor}"'
    if family:
        a += f' font-family="{family}"'
    return f'<text x="{x}" y="{y}"{a}{extra}>{escape(s)}</text>'


def box(kind: str, y: float, title: str, lines: list[str], note: str | None = None) -> list[str]:
    x, w = COLS[kind]
    stroke, head, body = STYLE[kind]
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{BOX_H}" rx="8" fill="{body}" '
        f'stroke="{stroke}" stroke-width="1.5"/>',
        f'<path d="M{x},{y + 8} Q{x},{y} {x + 8},{y} H{x + w - 8} Q{x + w},{y} {x + w},{y + 8} V{y + 28} H{x} z" '
        f'fill="{head}"/>',
    ]
    mono = kind == "files"
    out.append(text(x + 14, y + 20, title, 14 if not mono else 13, "700", family=MONO if mono else None))
    for i, ln in enumerate(lines):
        out.append(text(x + 14, y + 47 + 18 * i, ln, 11.5 if mono else 12, family=MONO if mono else None))
    if note:
        out.append(text(x + 14, y + BOX_H - 12, note, 11, fill="#666"))
    return out


def arrow(d: str, color: str, marker: str, dash: str | None = None, width: float = 1.6) -> str:
    da = f' stroke-dasharray="{dash}"' if dash else ""
    return f'<path d="{d}" stroke="{color}" stroke-width="{width}" fill="none"{da} marker-end="url(#{marker})"/>'


def main() -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" role="img" '
        'aria-label="enzopt 시스템 흐름도: 단계 계산, 실행 폴더 파일, 대시보드 탭">',
        "<defs>",
    ]
    for mid, col in (("ah", GREY), ("ahb", BLUE), ("ahg", GREEN), ("ahp", PURPLE)):
        parts.append(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        )
    parts += ["</defs>", f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    # input and runner
    parts += [
        '<rect x="330" y="14" width="440" height="56" rx="28" fill="#f2f2f2" stroke="#888" stroke-width="1.4"/>',
        text(550, 37, "입력", 15, "700", anchor="middle"),
        '<text x="550" y="58" text-anchor="middle" font-size="12.5" fill="#333">'
        f'<tspan font-family="{MONO}">project.yaml</tspan> (반응식·실험·설정) + '
        f'<tspan font-family="{MONO}">data/*.csv</tspan> (진행 곡선)</text>',
        arrow("M550,70 V94", GREY, "ah"),
        text(560, 87, "enzopt run  또는  대시보드 [새 실행]", 11, fill=GREY, family=MONO),
        '<rect x="50" y="96" width="1030" height="90" rx="8" fill="#f7f7f7" stroke="#999" stroke-width="1.3"/>',
        text(66, 117, "실행기 (pipeline/runner.py)", 14, "700"),
        text(
            66,
            139,
            "새 실행 폴더 runs/<프로젝트>/<날짜_시각>/ · "
            "manifest.json(설정·입력 해시·코드 버전·단계 상태·플래그·경고) · "
            "inputs/(입력 사본)",
            12,
        ),
        text(
            66,
            158,
            "단계마다 ① → ⑤ 순서로: 캐시 키 계산 → 이 폴더에서 이미 끝났으면 건너뜀 / "
            "앞 실행에 같은 키가 있으면 복사 / "
            "아니면 계산",
            12,
        ),
        text(
            66,
            177,
            "진행은 progress.json, 취소는 실행 폴더의 CANCEL 파일, 실패하면 그 단계에서 멈추고 manifest에 오류를 적음",
            11,
            fill="#666",
        ),
    ]
    # column headers
    for kind, label in (
        ("calc", "계산 (pipeline의 단계 함수)"),
        ("files", "실행 폴더의 파일"),
        ("tab", "대시보드 탭 (읽기만)"),
    ):
        x, w = COLS[kind]
        parts.append(text(x + w / 2, 208, label, 12.5, "700", fill="#444", anchor="middle"))
    parts.append(arrow("M75,186 V220", GREY, "ah"))
    # launch arrow: the dashboard starts the engine as a separate process
    parts.append(arrow("M1062,204 V188", PURPLE, "ahp", "5 3"))
    parts.append(text(1054, 202, "[새 실행]", 10.5, fill=PURPLE, anchor="end"))
    # rows
    cx, cw = COLS["calc"]
    fx, fw = COLS["files"]
    tx, _tw = COLS["tab"]
    for i, r in enumerate(ROWS):
        y = ROW0 + i * PITCH
        parts += box("calc", y, *r["calc"])
        parts += box("files", y, *r["files"])
        parts += box("tab", y, *r["tab"])
        ym = y + 75
        parts.append(arrow(f"M{cx + cw},{ym} H{fx - 2}", GREY, "ah"))
        parts.append(arrow(f"M{fx + fw},{ym} H{tx - 2}", GREEN, "ahg"))
    # files read by later stages (blue dashed)
    yb = lambda i: ROW0 + i * PITCH + BOX_H  # noqa: E731 — bottom of row i
    yt = lambda i: ROW0 + i * PITCH  # noqa: E731 — top of row i
    # ② → ③
    g = yb(1) + 12
    parts.append(arrow(f"M{fx + 40},{yb(1)} V{g} H{cx + 230} V{yt(2) - 2}", BLUE, "ahb", "6 3"))
    parts.append(text(fx + 50, g + 4, "exp_·sig_*.csv", 10.5, fill=BLUE, family=MONO))
    # ③ → ④ and ③ → ⑤ (one trunk; the ⑤ branch runs down the left margin)
    g = yb(2) + 12
    parts.append(arrow(f"M{fx + 40},{yb(2)} V{g} H{cx + 230} V{yt(3) - 2}", BLUE, "ahb", "6 3"))
    parts.append(arrow(f"M{cx + 230},{g} H16 V{yt(4) + 75} H{cx - 2}", BLUE, "ahb", "6 3"))
    parts.append(text(fx + 50, g + 4, "samples/*.json · fit_summary.json", 10.5, fill=BLUE, family=MONO))
    ylab = (g + yt(4) + 75) / 2
    parts.append(
        text(
            34,
            ylab,
            "⑤도 읽음: 표본 · design_blocked",
            10.5,
            fill=BLUE,
            anchor="middle",
            extra=f' transform="rotate(-90 34 {ylab})"',
        )
    )
    # legend
    ly = H - 40
    items = [
        (GREY, "ah", None, "계산 결과를 파일로 씀"),
        (BLUE, "ahb", "6 3", "다음 단계가 그 파일을 읽음"),
        (GREEN, "ahg", None, "탭이 파일을 읽어 표시"),
        (PURPLE, "ahp", "5 3", "대시보드가 엔진을 별도 프로세스로 띄움"),
    ]
    x = 60
    for col, mk, dash, label in items:
        parts.append(arrow(f"M{x},{ly} H{x + 34}", col, mk, dash))
        parts.append(text(x + 42, ly + 4, label, 11.5, fill="#333"))
        x += 64 + 12 * len(label) * 0.95
    parts.append(
        text(
            60,
            H - 14,
            "단계는 앞 단계의 결과를 메모리가 아니라 실행 폴더의 파일로만 읽습니다. "
            "그래서 캐시·이어서 실행·특정 단계부터 다시가 됩니다.",
            11,
            fill="#666",
        )
    )
    parts.append("</svg>")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
