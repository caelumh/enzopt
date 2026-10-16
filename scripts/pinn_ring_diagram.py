"""Diagram of one PINN evaluation for LDH (docs/flow, section 5.3): network curve, stoichiometry, the quasi-steady
enzyme ring, the two loss terms, and where the gradients go. Writes docs/flow/pinn_ring.svg (no third-party packages).

    uv run python scripts/pinn_ring_diagram.py
"""

from __future__ import annotations

import math
from pathlib import Path

from flow_diagram import BLUE, FONT, GREY, MONO, arrow, text

OUT = Path(__file__).resolve().parents[1] / "docs" / "flow" / "pinn_ring.svg"
W, H = 1290, 650
RED = "#C0392B"
STYLE = {  # stroke, fill
    "net": (BLUE, "#e8f1f9"),
    "ring": (GREEN := "#009E73", "#e6f5ef"),
    "plain": ("#999999", "#f7f7f7"),
    "loss": ("#D55E00", "#fbece3"),
}


def rbox(x, y, w, h, lines: list[str], kind: str, title_size: float = 13, size: float = 11.5) -> list[str]:
    """Rounded box with a bold first line and plain lines below, centred."""
    stroke, fill = STYLE[kind]
    out = [
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="8" fill="{fill}" stroke="{stroke}" stroke-width="1.5"/>'
    ]
    y0 = y + h / 2 - 9 * (len(lines) - 1) + 5
    for i, ln in enumerate(lines):
        out.append(
            text(x + w / 2, y0 + 18 * i, ln, title_size if i == 0 else size, "700" if i == 0 else None, anchor="middle")
        )
    return out


def ring(cx: float, cy: float, r: float) -> list[str]:
    """The five enzyme forms of the ordered bi-bi cycle, each step a pair of arrows with its pseudo-first-order rate."""
    names = ["E", "EN", "ENP", "EQL", "EQ"]
    pos = [(cx + r * math.cos(math.radians(a)), cy + r * math.sin(math.radians(a))) for a in (-90, -18, 54, 126, 198)]
    fwd = ["k₊₁[NADH]", "k₊₂[Pyr]", "k₊₃", "k₊₄", "k₊₅"]
    rev = ["k₋₁", "k₋₂", "k₋₃", "k₋₄[Lac]", "k₋₅[NAD]"]
    out = []
    for i in range(5):
        (x0, y0), (x1, y1) = pos[i], pos[(i + 1) % 5]
        L = math.hypot(x1 - x0, y1 - y0)
        ux, uy = (x1 - x0) / L, (y1 - y0) / L
        nx, ny = uy, -ux  # outward normal of the clockwise (forward) direction
        for sgn, (a, b) in ((1, ((x0, y0), (x1, y1))), (-1, ((x1, y1), (x0, y0)))):
            ox, oy = 4 * sgn * nx, 4 * sgn * ny
            dx, dy = (b[0] - a[0]) / L, (b[1] - a[1]) / L
            p0 = (a[0] + 27 * dx + ox, a[1] + 27 * dy + oy)
            p1 = (b[0] - 27 * dx + ox, b[1] - 27 * dy + oy)
            col = "#444444" if sgn > 0 else "#999999"
            out.append(arrow(f"M{p0[0]:.1f},{p0[1]:.1f} L{p1[0]:.1f},{p1[1]:.1f}", col, "ah", width=1.3))
        mx, my = (x0 + x1) / 2, (y0 + y1) / 2
        # labels clear the arrows: the offset grows with the half text width on near-vertical edges
        of = 14 + abs(nx) * 3.4 * len(fwd[i]) + abs(ny) * 4
        orv = 12 + abs(nx) * 3.2 * len(rev[i]) + abs(ny) * 4
        out.append(text(f"{mx + of * nx:.1f}", f"{my + of * ny + 4:.1f}", fwd[i], 11.5, anchor="middle"))
        out.append(text(f"{mx - orv * nx:.1f}", f"{my - orv * ny + 4:.1f}", rev[i], 11, fill="#777", anchor="middle"))
    for (x, y), nm in zip(pos, names, strict=True):
        out.append(
            f'<rect x="{x - 24:.1f}" y="{y - 14:.1f}" width="48" height="28" rx="12" fill="#ffffff" '
            f'stroke="{BLUE}" stroke-width="1.4"/>'
        )
        out.append(text(f"{x:.1f}", f"{y + 5:.1f}", nm, 12.5, "700", anchor="middle"))
    return out


def main() -> None:
    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" font-family="{FONT}" role="img" '
        'aria-label="PINN 한 번의 계산: 신경망 곡선, 화학량, 효소 고리, 손실, 경사">',
        "<defs>",
    ]
    for mid, col in (("ah", GREY), ("ahr", RED)):
        parts.append(
            f'<marker id="{mid}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
            f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{col}"/></marker>'
        )
    parts += ["</defs>", f'<rect width="{W}" height="{H}" fill="#ffffff"/>']
    # network and its two outputs
    parts += rbox(30, 150, 180, 70, ["신경망", "시드 × 웰마다 하나", "폭 32, 깊이 3"], "net", 13.5, 12)
    parts.append(arrow("M180,104 V148", GREY, "ah"))
    parts.append(text(188, 128, "τ = t / T", 12, family=MONO))
    parts.append(arrow("M120,220 V238 H72 V258", GREY, "ah"))
    parts.append(arrow("M120,238 H167 V258", GREY, "ah"))
    parts += rbox(30, 260, 85, 46, ["기울기", "dNADH/dt"], "net", 12)
    parts += rbox(125, 260, 85, 46, ["곡선", "NADH(t)"], "net", 12)
    parts.append(text(30, 326, "기울기는 신경망을 미분해 정확히", 10.5, fill="#666"))
    # stoichiometry
    parts += rbox(245, 245, 170, 80, ["나머지 농도", "Pyr = Pyr₀ − 쓴 양", "Lac = NAD = 쓴 양"], "plain")
    parts.append(arrow("M210,283 H243", GREY, "ah"))
    parts.append(arrow("M415,290 H438", GREY, "ah"))
    # rate constants and the ring
    parts += rbox(
        440,
        30,
        320,
        62,
        ["속도 상수 θ (학습하는 값)", "log10 [k₊₁~k₊₅, k₋₁~k₋₄, K′]", "k₋₅는 K′에서 (Haldane)"],
        "loss",
    )
    parts.append(arrow("M600,92 V110", GREY, "ah"))
    parts.append(
        f'<rect x="440" y="112" width="320" height="330" rx="10" fill="#fafcff" stroke="{BLUE}" '
        'stroke-width="1.5" stroke-dasharray="6 3"/>'
    )
    parts.append(text(600, 136, "효소 고리 (준정상)", 13.5, "700", anchor="middle"))
    parts.append(text(600, 432, "화살표 속도 = 상수 × 그때 농도", 11, fill="#555", anchor="middle"))
    parts += ring(600, 292, 92)
    # enzyme distribution → prediction (up) and rate (down)
    parts += rbox(790, 110, 220, 60, ["예측 관측값", "NADH + Et(φ_EN + φ_ENP)"], "ring")
    parts += rbox(790, 255, 220, 70, ["효소 분포 φ", "형태마다 들어옴 = 나감, 합 1", "(미분 가능한 풀이)"], "ring")
    parts += rbox(
        790, 395, 220, 66, ["반응 속도 v", "Et(k₊₁φ_E[NADH] − k₋₁φ_EN)", "= 고리를 도는 흐름"], "ring", 13, 11
    )
    parts.append(arrow("M760,290 H788", GREY, "ah"))
    parts.append(arrow("M900,255 V172", GREY, "ah"))
    parts.append(arrow("M900,325 V393", GREY, "ah"))
    # data term, measurement, total loss
    parts += rbox(1040, 24, 210, 54, ["측정값 y", "학습 웰 108개 × 50점"], "plain")
    parts += rbox(1040, 110, 210, 60, ["데이터 항", "((y − 예측) / σ)², σ = 9 µM"], "loss")
    parts.append(arrow("M1010,140 H1038", GREY, "ah"))
    parts.append(arrow("M1145,78 V108", GREY, "ah"))
    parts += rbox(
        1040,
        488,
        210,
        92,
        ["손실 (시드마다)", "데이터 + w × 물리", "+ 처음 조건 + 0.01 × 사전분포", "w: 1 → 10⁴로 올림"],
        "loss",
    )
    parts.append(arrow("M1145,170 V486", GREY, "ah"))
    # physics term under the ring
    parts += rbox(250, 500, 350, 60, ["물리 항", "(dNADH/dt + v)²: 곡선 기울기 = 고리 속도?"], "loss")
    parts.append(arrow("M72,306 V530 H248", GREY, "ah"))
    parts.append(arrow("M900,461 V478 H425 V498", GREY, "ah"))
    parts.append(arrow("M600,530 H1038", GREY, "ah"))
    # gradients (backward)
    parts.append(arrow("M1250,534 H1272 V12 H14 V185 H28", RED, "ahr", "6 4", 1.5))
    parts.append(arrow("M600,12 V28", RED, "ahr", "6 4", 1.5))
    parts.append(text(245, 44, "경사 → 속도 상수", 11.5, fill=RED))
    parts.append(text(245, 61, "(주로 물리 항에서, 고리를 거꾸로)", 11, fill=RED))
    parts.append(text(24, 142, "경사 → 신경망 가중치", 11.5, fill=RED))
    # legend
    ly = H - 34
    parts.append(arrow(f"M40,{ly} H80", GREY, "ah"))
    parts.append(text(88, ly + 4, "계산(순전파): 시각마다 매번", 11.5, fill="#333"))
    parts.append(arrow(f"M320,{ly} H360", RED, "ahr", "6 4", 1.5))
    parts.append(
        text(368, ly + 4, "경사(역전파): 손실이 줄어드는 쪽으로 신경망과 속도 상수를 함께 고침", 11.5, fill="#333")
    )
    parts.append(
        text(
            40,
            H - 10,
            "예: ldh_wang2025. 학습 웰 108개 × 시드 × 재시작마다 이 계산을 Adam 2000번 + L-BFGS 1000번 되풀이합니다.",
            11,
            fill="#666",
        )
    )
    parts.append("</svg>")
    OUT.write_text("\n".join(parts) + "\n", encoding="utf-8", newline="\n")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
