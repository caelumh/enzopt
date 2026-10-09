# ldh_holmes1988 — 실측 문헌 데이터로 식별·외삽 미리 시험하기

## 출처

Holmes, L. D. (1988). *Kinetics of Two-Substrate Reactions Using Integrated Rate Equations*. PhD dissertation, Utah State University. https://digitalcommons.usu.edu/etd/7167 (원문 `raw/holmes1988_two_substrate_integrated_rate.pdf`). 관련 논문: Holmes, Schiller & Boeker (1993), *Experientia* 49:893–901, doi:10.1007/BF01952605.

- 효소: 토끼 근육 LDH M4(Sigma, 투석), 비활성도 330 units/mg(pH 7.0, 25 ℃)
- 조건: 0.1 M 인산 pH 7.0, 25 ℃, BSA 0.1 mg/ml, Cary 219 분광광도계, 340 nm(1 cm·0.2 cm 큐벳)
- 측정: 초기 농도 21조건(표 II, 생성물 NAD⁺ 최대 7 mM·젖산 최대 40 mM 포함) × 6회 반복, 진행 곡선 전체

## 이 폴더의 데이터가 무엇인가 (중요)

논문에 원시 측정값 표는 없다. 대신 다음을 옮겨 데이터를 만들었다(`make_data.py`).

| 파일 | 출처 | 성격 |
|---|---|---|
| `raw/holmes1988_tables.csv` | 표 II(초기 농도), 표 III(적분 속도식 계수 평균·표준편차), 표 XII(적합 오차) — 쪽 이미지와 대조해 옮김 | 원문 수치 |
| `data/set<번호>.csv` (21개) | 식 6 e₀t = −C_f ln(1−ΔP/A₀) − C_s ln(1−ΔP/B₀) + C₁ΔP + ½C₂ΔP²에 표 III 계수를 넣어 복원, 0~90% 전환 | **측정을 매끈하게 근사한 곡선**(6회 반복 적합의 평균, 적합 오차 1% 미만). 잡음 없음 |
| `data/set408fig.csv`, `set172fig.csv` | 그림 4·5의 측정점(●)을 축 눈금으로 읽음(±0.005) | **원시 측정점** |

시간 축: 원 데이터는 e₀t(units·min/ml)라서 모든 세트를 명목 효소 0.05 units/ml(= 4.14 nM 활성 자리, 36.6 kDa 기준)로 환산했다. 피펫 표의 "E₀의 몇 배"는 이 환산값과 무관하다.

## 무엇을 시험하나

- 학습 15세트로 속도 상수 표본을 만들고, 학습에 쓰지 않은 6세트(젖산 많은 172, 묽은 408·465, NAD⁺ 많은 422, 고농도 251·457)와 원시 점 2세트의 외삽 오차를 본다.
- 세트 172는 Holmes의 전역 상수 예측이 33% 빗나간 조건(그림 5, 표 XII)이다.

## 한계

- pH 7.0에서는 K'(≈ 2×10⁴)가 커서 반응이 거의 끝까지 간다. 정리 149의 "같은 적합류 → 같은 평형" 시험은 이 데이터로는 약하다(우리 실험이 pH 10을 쓰는 이유).
- 복원 곡선은 Holmes 적분식으로 맞춘 곡선이라 실제 측정의 잡음·기기 효과가 없다. 원시 점은 2세트뿐이다.
- 설계(⑤)는 참고용이다. pH 7.0 인산 조건이고 막다른 복합체 Ki = 5 mM은 가정값이다.

```
uv run python projects/ldh_holmes1988/make_data.py     # data/ 다시 만들기
uv run enzopt check projects/ldh_holmes1988
uv run enzopt run projects/ldh_holmes1988
```
