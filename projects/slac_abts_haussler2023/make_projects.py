"""Write projects/slac_abts_haussler2023_25C … _45C/project.yaml from the experiment lines that make_data.py printed
(experiments.txt in each).

    uv run python -I projects/slac_abts_haussler2023/make_projects.py

One project per temperature: rate constants depend on temperature and every enzopt project is isothermal (the chamber
temperature of each read is in the data files' T_C column, so stage ② checks it). Training: ABTS 5, 15, 50, 100 µM;
held out: 10, 25, 75 µM (inside) and 150 µM (outside the training range); three replicate wells each.
"""

from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
PROJECTS = HERE.parent
TEMPS = [25, 30, 35, 40, 45]
TRAIN = {"S5", "S15", "S50", "S100"}

TEMPLATE = """# 공개 문헌 데이터 프로젝트: Häußler (2023) 작은 라카아제(SLAC, S. coelicolor)의 ABTS 산화, pH 3.0, {T} ℃
# 데이터: DaRUS doi:10.18419/DARUS-3337 (University of Stuttgart, CC BY 4.0).
# slac_abts_haussler2023/make_data.py가 플레이트 리더 파일에서 340 nm ABTS 흡광을 같은 온도의 표준 곡선으로
# 농도로 바꾸고, 측정마다의 플레이트 온도를 T_C 열로 넣음
name: slac_abts_haussler2023_{T}C
description: 공개 문헌 데이터 — SLAC + ABTS, pH 3.0, {T} ℃, 웰 24개(ABTS 5~150 µM × 3반복), 15분 11점

network:
  name: slac_mm_inactivation
  # 저자 모델(비가역 Michaelis–Menten + 시간에 따른 효소 실활)을 질량 작용으로. O₂는 재지 않았고 넉넉하다고 봄
  # (ABTS 150 µM을 다 산화해도 O₂는 약 40 µM만 씀). P = ABTS 라디칼, ED = 망가진 효소
  species: [E, ABTS, ES, P, ED]
  enzyme_species: [E, ES, ED]
  steps:
    - E + ABTS <=> ES               # 1 ABTS 결합
    - ES -> E + P                   # 2 산화와 방출
    - E -> ED                       # 3 자유 효소의 실활
    - ES -> ED                      # 4 ABTS가 붙은 효소의 실활(저자처럼 효소 전체가 망가짐; 35 ℃에서 3만 둘 때보다
                                    #   ΔAIC −40, enzopt suggest --candidate "ES -> ED", README)
  observable:
    name: ABTS_mM                   # 340 nm 표준 곡선으로 잰 ABTS(효소에 붙은 몫 포함)
    coefficients: {{ABTS: 1, ES: 1}}
  thermodynamics:
    reactions:                      # 한 방향 반응이라 K'를 쓰지 않음
      - {{overall: ABTS <=> P}}
  prior:                            # 일반 자릿수(저자 값이 아님), 표준편차 1.5 자릿수
    kf_log10: [3, 2, -2, -2]        # 결합 1/(mM·s), 산화 1/s, 실활 둘 1/s
    kr_log10: [2]                   # 해리 1/s(해리 상수 0.1 mM 근처)
    sd_decades: 1.5

control:                            # 대조: ABTS 결합을 한 방향으로(ABTS가 다시 떨어지지 않음)
  one_way_steps: [1]

identification:
  method: pinn
  seeds: 4
  noise_sigma: 0.0005               # mM = 0.5 µM: 표준 곡선 직선의 잔차(0.4 µM) 근처, 대조 웰의 읽기 잡음보다 큼
  baseline_fit: {{starts: 4, max_nfev: 200}}

simulation:
  t_long_s: 1800

design:                             # 참고용: ABTS 0.1 mM의 90 %를 15분 안에 산화하는 최소 효소량
  target: assay
  analyte: ABTS
  partner: null                     # 기질 1개(O₂는 넉넉): 효소량만 찾음
  free_enzyme: E
  analyte0_mM: 0.1
  conversion: 0.9
  time_s: 900
  Et_bounds_mM: [1.0e-6, 1.0e-2]
  check_t_end_s: 1800
  pipette:
    volume_uL: 200
    enzyme_uL: 20
    stocks_mM: {{ABTS: 2}}
    limit_test_rhos: []
    limit_test_enzyme_multiples: []

# 처음 ABTS = 그 웰의 첫 읽기(효소 0.0739 µM = 73.94 nM, 저자 템플릿). 학습: 5, 15, 50, 100 µM(3반복씩),
# 홀드아웃: 10, 25, 75 µM(학습 범위 안), 150 µM(학습 범위 바깥)
experiments:
"""


def main() -> None:
    for T in TEMPS:
        d = PROJECTS / f"slac_abts_haussler2023_{T}C"
        lines = []
        for ln in (d / "experiments.txt").read_text(encoding="utf-8").splitlines():
            name = ln.split(":")[0].strip()
            role = "train" if name.split("_")[0] in TRAIN else "holdout"
            lines.append(ln.replace("role: ROLE", f"role: {role}"))
        text = TEMPLATE.format(T=T) + "\n".join(lines) + "\n"
        (d / "project.yaml").write_text(text, encoding="utf-8", newline="\n")
        print(
            f"slac_abts_haussler2023_{T}C/project.yaml: {sum('role: train' in x for x in lines)} train, "
            f"{sum('role: holdout' in x for x in lines)} held out"
        )


if __name__ == "__main__":
    main()
