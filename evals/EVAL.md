# 평가 기준

> 최종 갱신: 2026-10-10(보완 인터뷰: F-13 기준 AC-21~32, 긴 평가 60분 — 브랜치 `feature/general-network`) · 평가 1회 예산: 빠른 5분 / 긴 60분 · 점수: 0~100, 높을수록 좋음 · 명세: `../PROJECT_SPEC.md`

## 지침 (사람이 편집)
개선 루프에 주는 방향. 비어 있으면 아래 우선순위대로 돈다.
- 이번 사이클의 집중: F-13 일반화를 M6-1 → M6-5 순서로(명세 "F-13 범위" G8). 단계마다 AC-01~20이 그대로 통과해야 한다. (전 사이클: ① 10/12 → ① 10/14 → ② 10/21 순서, 모두 통과)
- 하지 말 것: 아래 "변경 금지" 목록. 합성 데이터 점수를 올리려고 PINN 사전분포를 합성 정답 쪽으로 옮기지 않는다. 사전분포 값, 수렴 기준, 판정 기준 같은 과학적 판단은 바꾸지 말고 `evals/QUESTIONS.md`에 남긴다(명세 W3).

## 평가 단계 (E4)
| 단계 | 예산 | 언제 | 무엇을 |
|---|---|---|---|
| 빠른 | ≤ 5분 | 모든 변경마다 | 관문 + 평가 열이 "빠른"인 기준. PINN은 짧은 모드(시드 2, 반복 축소) |
| 긴 | ≤ 60분 (2026-10-10, 전에는 30분) | 마일스톤 끝, 그리고 개선 후보를 채택하기 전 | 관문 + 모든 기준 + 지표 |

하네스: `uv run python evals/run.py --tier quick` / `--tier long`. 각 기준의 통과 여부, 지표, 점수를 출력하고 `evals/results.tsv`에 한 줄을 더한다. 하네스와 `evals/baselines.json`·`evals/synthetic.yaml`은 마일스톤 0~1에서 build가 한 번 만들고, 그 뒤로는 바꾸지 않는다. F-13 기준(AC-21~32)을 하네스에 더하는 일과 `evals/synthetic_general.yaml`(부록 C)은 M6-0b에서 build가 한 번 한다.

## 관문 — 하나라도 실패하면 그 변경은 폐기
| ID | 검사 | 명령 |
|---|---|---|
| G1 | 환경 동기화 | `uv sync --locked` |
| G2 | 린트·포맷 | `uv run ruff check .` 그리고 `uv run ruff format --check .` |
| G3 | 빠른 테스트 | `uv run pytest -m "not long" -q` |
| G4 | 이미 통과하던 회귀 기준(AC-01·AC-02)이 실패로 바뀌지 않음 | 하네스가 직전 `results.tsv`와 비교 |

## 수용 기준 — 가중치: 필수 3 · 중요 2 · 선택 1
점수 범위는 AC-01~AC-32다(AC-21~32는 F-13 기준, 2026-10-10 보완 인터뷰). 아직 만들지 않은 기능의 기준은 실패로 센다. F-14·F-15 기준은 보완 인터뷰로 명세가 바뀐 뒤에 더한다.

| ID | 단계 | 근거 | 기준 | 가중치 | 평가 | 검사 방법 |
|---|---|---|---|---|---|---|
| AC-01 | ① 10/12 | T1-회귀, F-5 | 겉보기 1단계 LDH(k₁ = 2.0 /mM/s, K' = 23)의 149 배치를 enzopt로 다시 돌리면: 같은 류 12개 실행 모두 CRNT 검증 통과, 마지막 NADH 0.1485 mM(±0.0005), 퍼짐 < 1e-4. 다른 류(D = −0.3) 마지막 NADH 0.1019 mM(±0.0005). 합계 13/13 통과 | 3 | 빠른(scipy 백엔드), 긴(Hopsan 백엔드) | `tests/test_regression.py::test_batch149` |
| AC-02 | ① 10/12 | T1-회귀, F-6 | 합성 표본 9개의 분석법 설계가 CADKIMM과 같다: ρ_min = 5.251(K' = 23), 강건 최적 ρ* = 9.00, 강건 Et* = 180.54 nM(±2%), 같은 ρ*의 공칭 Et = 31.35 nM(±2%), 공칭 설계는 표본 9개 중 4개 미달·최악 전환율 0.760(±0.01), 강건 설계는 0개 미달 | 3 | 긴(전체). 빠른 평가는 `baselines.json`의 `--quick` 기준값과 비교 | `tests/test_regression.py::test_design` |
| AC-03 | ① 10/12 | F-3 | 판정: LDH 완전 메커니즘(9종·10반응)은 약가역·결핍 0·연결 성분 3·정수 보존 법칙 4개. 설계 메커니즘(EQP 포함 10종)은 약가역·결핍 0·연결 성분 4·보존 법칙 4개. pNP(E + S ⇌ ES → E + P)와 비약가역 대조(마지막 단계 한 방향)는 약가역 아님 | 3 | 빠른 | `tests/test_crnt.py` |
| AC-04 | ① 10/12 | F-1, F-2, SC5-6 | 예제 프로젝트 `projects/example_ldh_synth`의 `project.yaml`과 표준 CSV를 읽는다. 시간 역행·빈 값·A340 > 2가 든 CSV는 파일 이름과 줄 번호를 담은 오류로 거부한다 | 3 | 빠른 | `tests/test_io.py` |
| AC-05 | ① 10/12 | RK1·RK2, F-4 | 현실적 합성 데이터(`evals/synthetic.yaml`)에서 약가역 PINN 표본(시드 4)의 홀드아웃 세트 C·D' 외삽 RMSE(관측 데이터 대비) ≤ max(2σ = 0.006 A340, 기준 모델 RMSE × 1.5), 세트마다 표본 평균과 최악 모두. 기준 모델 = 합성 정답 메커니즘·속도 상수 + 학습 세트(A·B)에 최소제곱으로 맞춘 효소량 배율 하나(평가 전용, 2026-10-09 보완 인터뷰 Q1·Q-01) | 3 | 긴 | `tests/test_pinn.py::test_realistic_holdout` (long) |
| AC-06 | ① 10/12 | F1, F-6 | 끝-끝: `enzopt run projects/example_ldh_synth`가 ①~⑤를 끝까지 돌고, 실행 폴더에 판정 보고, 표본, 검증 결과, 설계 표, 피펫 표(L1·L2·D1·D0·Z)가 생긴다 | 3 | 빠른(짧은 모드) | `tests/test_e2e.py` |
| AC-07 | ① 10/12 | F-5, SC5-5 | 모든 해석 결과가 CRNT 검증을 통과한다: 보존 표류 < 1e-6, 출발 직후부터 최소 농도 > 0, 마지막 상태의 \|ln(Q/K)\| < 0.01, 리아푸노프 함수 비증가. 실패한 해석은 "수치 오류 의심"으로 표시되고 Δt를 1/10로 줄여 한 번 재실행된다 | 2 | 빠른 | `tests/test_validate.py` |
| AC-08 | ① 10/12 | F6 | 같은 현실적 합성 문제에서 GPU float32 PINN과 CPU float64 PINN의 홀드아웃 RMSE 차이 ≤ 10%, GPU 쪽 학습 벽시계 시간이 더 짧다. GPU를 쓸 수 없으면 CPU로 자동 전환되고 그 사실이 manifest에 기록된다 | 2 | 긴 | `tests/test_pinn.py::test_gpu_vs_cpu` (long) |
| AC-09 | ① 10/12 | F-7, SC5-7~9 | 재실행하면 새 실행 폴더가 생기고 이전 폴더는 그대로다. 입력이 같으면 두 번째 실행은 PINN 단계를 캐시로 건너뛴다. 단계 도중 중단된 실행을 다시 시작하면 끝난 단계는 건너뛰고 끊긴 단계부터 잇는다 | 2 | 빠른 | `tests/test_runs.py` |
| AC-10 | ① 10/12 | I1-KIMM-CAD | 핵심 모듈(판정·scipy 해석·검증·설계·YAML 읽기) 테스트가 KIMM-CAD 호환 환경(Python 3.12, numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.1, torch·Streamlit 없음)에서 통과한다 | 2 | 빠른 | 호환 환경에서 `pytest tests/core` |
| AC-11 | ① 10/12 | I1-Hopsan, SC5-4 | Hopsan 백엔드로 겉보기 1단계 적합류 배치와 대표 표본 하나의 완전 메커니즘을 풀어 scipy 결과와의 최대 차이를 기록한다. Hopsan을 찾지 못하면 scipy로 대신하고 manifest와 판정 보고에 "Hopsan 미사용"이 남는다 | 2 | 긴 | `tests/test_hopsan.py` (long) |
| AC-12 | ① 10/12 | F-8, 영역 13 확인 사항 | 표준 그림 세트(명세 13)가 그림마다 PDF + PNG(300 dpi) + 같은 이름의 CSV로 생기고, 그림 CSV의 값이 실행 폴더의 결과 파일 값과 같다 | 2 | 빠른 | `tests/test_figures.py` |
| AC-13 | ① 10/12 | SC5-1 | 비약가역 반응망 프로젝트도 멈추지 않고 끝까지 돌며, 판정 보고와 manifest에 "149 보장 없음"이 기록된다 | 2 | 빠른 | `tests/test_e2e.py::test_non_wr` |
| AC-14 | ① 10/12 | F7, 영역 5 확인 사항 ① | `enzopt pack <실행 폴더>`가 입력 데이터·설정 사본·그림·CSV·Hopsan `.hmf`·검증 보고를 zip 하나로 묶는다 | 1 | 빠른 | `tests/test_pack.py` |
| AC-15 | ① 10/12 | 영역 14 확인 사항 ① | `enzopt doctor`가 Python·패키지 버전, CUDA, Hopsan 경로를 점검하고 빠진 구성 요소를 이름으로 알린다 | 1 | 빠른 | `tests/test_doctor.py` |
| AC-16 | ① 10/14 | F-9 | 같은 현실적 합성 문제에서 고전 피팅(같은 ODE, 다중 시작점 최소제곱) 표본과 PINN 표본의 홀드아웃 외삽 오차를 한 표(CSV + 그림)로 낸다 | 3 | 긴 | `tests/test_baseline_fit.py` (long) |
| AC-17 | ① 10/14 | F-10 | 비약가역 대조 모델을 같은 흐름으로 학습해, 약가역 모델과의 홀드아웃 외삽 비교 표와 그림 ③(b)를 낸다 | 3 | 긴 | `tests/test_control.py` (long) |
| AC-18 | ① 10/14 | F-11 | 설계 조건의 예측 곡선(scipy 표본 전체 + Hopsan 대표 표본)을 한 그림에 그리고, 두 곡선의 최대 차이를 기록한다. 확인 실험 CSV가 있으면 점으로 겹친다 | 2 | 긴 | `tests/test_design_check.py` (long) |
| AC-19 | ② 10/21 | F-12, SC2, SC4 | 대시보드가 실행 폴더를 읽어 왼쪽 프로젝트·실행 목록과 탭 ①~⑤를 보여 주고, 진행 중(진행률)·실패(단계와 이유)·판정 실패("149 보장 없음" 띠) 상태를 표시한다 | 3 | 빠른 | Streamlit `AppTest`로 `tests/test_dashboard.py` |
| AC-20 | ② 10/21 | F-12, SC5-10·11 | 대시보드 [새 실행]이 별도 프로세스로 엔진을 시작하고, 대시보드 프로세스를 끝내도 계산이 계속되어 실행 폴더가 완료 상태가 된다 | 2 | 빠른 | `tests/test_dashboard.py::test_detached_run` |
| AC-21 | ③ F-13 M6-1 | G1ⓐ, G2 | 이름·관측식: LDH와 같은 모양을 다른 이름(A·B·P·Q)과 다른 관측식으로 적은 프로젝트가 ①~⑤를 끝까지 돈다(짧은 모드). ③ 학습 RMSE와 ⑤ ρ*·Et*가 LDH 이름판과 1% 안에서 같다(같은 장치·시드) | 3 | 빠른 | `tests/test_general.py::test_renamed_ldh` |
| AC-22 | ③ F-13 M6-1 | G6 | 여러 신호: 같은 시점의 CSV 열과 다른 시점의 파일을 읽어 신호마다 σ로 나눈 손실로 학습한다. 관측식에 없는 신호, 빈 값, 시간 역행은 파일 이름·줄 번호를 담은 오류로 거부한다 | 2 | 빠른 | `tests/test_general.py::test_multi_signal` |
| AC-23 | ③ F-13 M6-2 | G4 | 열역학 자동: 부록 C의 반응망 11개와 LDH에서, 도구가 만든 속도 상수가 모든 순환 조건을 만족한다(전체 반응을 이루는 순환의 곱 = 그 반응의 K′, 내부 순환 = 1, 상대 오차 1e-9). LDH는 `haldane_step`으로 만든 상수와 같다 | 3 | 빠른 | `tests/test_general.py::test_cycle_constraints` |
| AC-24 | ③ F-13 M6-2 | G1ⓑ, RK1 | 고리형 7종 식별: `uni_irreversible`·`uni`·`uni_iso`·`uni_bi_ordered`·`theorell_chance`·`ping_pong`·`ordered_terter`의 현실적 합성 데이터(부록 C)에서 일반 PINN 표본(시드 4)이 반응망마다 학습 재현과 홀드아웃 AC-05 틀(아래 정의)을 모두 통과한다 | 3 | 긴 | `tests/test_general.py::test_ring_identification` (long) |
| AC-25 | ③ F-13 M6-2 | G1ⓑ | 갈래 식별: `random_bibi`와 LDH + 막다른 복합체(EQ + Pyr ⇌ EQP를 정식 단계로)에서 AC-24와 같은 기준 | 3 | 긴 | `tests/test_general.py::test_branched_identification` (long) |
| AC-26 | ③ F-13 M6-3 | G3 | 비교 모델: AC-24·25·31의 반응망마다 고전 피팅 기준선과 비약가역 대조(`control.one_way_steps`로 지정한 단계를 한 방향으로)를 같은 흐름으로 돌려, AC-16·17과 같은 형식의 홀드아웃 비교 표(CSV + 그림)를 낸다 | 2 | 긴 | `tests/test_general.py::test_comparisons` (long) |
| AC-27 | ③ F-13 M6-4 | G5 | ⑤ 분석법 일반화: 기질 1개(`uni`)·2개(`ping_pong`)·3개(`ordered_terter`)에서 강건 설계와 피펫 표가 나온다. 평형 한계의 수치 계산이 LDH 대수식(ρ_min = 5.251, K′ = 23)과 ±0.1%로 같다. 결핍이 0보다 크면 평형 한계를 쓰지 않고 이유를 남긴다 | 3 | 빠른 | `tests/test_general_design.py::test_assay` |
| AC-28 | ③ F-13 M6-4 | G5 | 수율 목표: 주어진 표본 모두에서 시간 안에 목표 수율 이상인 최소 효소량이 촘촘한 격자 전수 탐색과 ±2%로 같다 | 2 | 빠른 | `tests/test_general_design.py::test_yield` |
| AC-29 | ③ F-13 M6-4 | G5 | 선택도 목표: 경쟁 기질 반응망에서 주어진 표본 모두가 선택도·수율 조건을 만족하는 조건(최소 효소량)이 전수 탐색과 ±2%로 같다 | 2 | 빠른 | `tests/test_general_design.py::test_selectivity` |
| AC-30 | ③ F-13 M6-4 | G5 | 도달 시간 목표: 효소량 상한에서, 표본 중 가장 느린 도달 시간을 최소로 만드는 ρ가 전수 탐색과 같은 격자점이다 | 1 | 빠른 | `tests/test_general_design.py::test_time` |
| AC-31 | ③ F-13 M6-5 | G1ⓒ | 반응 둘 이상 식별: LDH 경쟁 기질(피루브산 + 2-옥소부티르산)과 LDH + 효소 없는 NADH 분해(한 방향)에서 AC-24와 같은 기준(여러 신호를 써도 됨) | 3 | 긴 | `tests/test_general.py::test_multi_reaction_identification` (long) |
| AC-32 | ③ F-13 M6-5 | G2, F1 | 끝-끝: 부록 C의 반응망 11개와 LDH 이름판의 프로젝트가 `enzopt run --quick`으로 ①~⑤를 끝까지 돈다(⑤는 설계 목표를 적은 경우). 비약가역(등급 C)은 "149 보장 없음" 띠와 함께 돈다 | 3 | 긴 | `tests/test_general.py::test_e2e_all` (long) |

F-13 기준의 두 판정(2026-10-10 보완 인터뷰):
- 학습 재현: 표본마다 학습된 속도 상수로 ODE를 다시 풀었을 때, 모든 학습 세트·신호에서 RMSE ÷ σ ≤ 3인 표본이 절반 이상
- 홀드아웃 AC-05 틀: 홀드아웃 세트(·신호)마다 RMSE ≤ max(2σ, 기준 모델 RMSE × 1.5)를 표본 평균과 최악 모두에서. 기준 모델 = 합성 정답 메커니즘·속도 상수 + 학습 세트에 최소제곱으로 맞춘 효소량 배율 하나(평가 전용)

## 품질 지표 (E3)
| ID | 지표 | 목표 | 방향 | 측정 |
|---|---|---|---|---|
| Q1 | 끝-끝 실행 시간: `projects/example_ldh_synth` 전체 모드(시드 4, Hopsan 포함, 캐시 없이) | 15분 | 낮을수록 좋음 | 긴 평가에서 하네스가 잰 벽시계 시간 |

## 점수 (E2)
- 기준 점수 = 통과한 수용 기준의 가중치 합 ÷ 전체 가중치 합(AC-01~AC-32, 합계 77. F-13 전에는 AC-01~20, 합계 47)
- 지표 달성도 = min(1, 15분 ÷ Q1 실측)
- 점수 = 70 × 기준 점수 + 30 × 지표 달성도
- 빠른 평가에서는 "긴" 기준과 Q1에 직전 긴 평가의 결과를 그대로 쓴다

## 채택 규칙
- 관문을 모두 통과하고, 점수가 지금까지의 최고보다 높고, 전에 통과하던 수용 기준이 실패로 바뀌지 않았으면 채택한다. 채택 전에 긴 평가를 한 번 돌린다
- 그 외에는 폐기한다. 점수가 같으면 폐기한다(단순한 쪽을 남긴다)

## 종료 (E5·E6)
- AC-01~AC-32가 모두 통과하고 Q1 ≤ 15분이면 멈춘다
- 기준이 틀렸거나 서로 부딪혀 진행할 수 없으면 코드를 억지로 맞추지 않고 `evals/QUESTIONS.md`에 질문을 남기고 멈춘다
- 사람은 사이클이 끝날 때(전부 통과 또는 막힘) 결과를 확인한다

## 변경 금지
- `evals/` 폴더 전체: 이 파일, 하네스, `baselines.json`, `synthetic.yaml`, `synthetic_general.yaml`(M6-0b에서 만든 뒤). 수용 기준의 추가·수정은 보완 인터뷰로 명세가 바뀐 뒤에만 한다
- CRNT 판정·검증 기준과 허용 오차(E7): 약가역성(강연결 성분)·결핍·보존 법칙 판정 로직, 검증 허용 오차(보존 표류 1e-6, |ln Q/K| 0.01, 리아푸노프 비증가), AC-05의 문턱, F-13 기준의 문턱(학습 재현 3σ·절반, 홀드아웃 AC-05 틀, 순환 조건 1e-9, 설계 ±2%·±0.1%)

## 부록 A — 회귀 기준값 (`evals/baselines.json`으로 옮긴다)
출처는 CADKIMM(`C:\Users\cael\Desktop\CADKIMM`)의 2026-10-09 실행 결과다.
| 항목 | 값 | 출처 |
|---|---|---|
| 149 배치, 같은 류 12개 | 마지막 [NADH, NAD, Pyr, Lac] = [0.1485, 0.0515, 0.1485, 9.8515] mM, 퍼짐 0 | `hopsan/results/batch149/summary.txt` |
| 149 배치, 다른 류 (D = −0.3) | 마지막 [0.1019, 0.0981, 0.4019, 9.5981] mM | 같은 파일 |
| 149 배치 자동 검증 | 13/13 통과 | `hopsan/README.md`, `analysis/crnt_validate.py` |
| 비약가역 대조(역반응 제거) | NADH·Pyr → 8.3×10⁻⁴ mM으로 감소(멸종 방향) | `hopsan/results/batch149/summary.txt` |
| 설계 판정(EQP 포함 10종) | 복합체 10, 연결 성분 4, rank 6, 결핍 0, 약가역, 보존 법칙 4개 | `analysis/results/design/summary.txt` |
| 평형 한계 | ρ_min = 5.251 (K' = 23, 99%) | 같은 파일 |
| 강건 설계(합성 표본 9) | ρ* = 9.00, 강건 Et* = 180.54 nM, 공칭 Et = 31.35 nM, 공칭 미달 4/9·최악 전환율 0.7602, 강건 미달 0/9·최악 0.9900 | 같은 파일 |
| 설계 `--quick` | 마일스톤 0에서 CADKIMM `design_optimize.py --quick`을 돌려 생성 | — |
| CSTR + PID | q₀ = 0.00767 1/s, 개루프 출구 NADH 0.0500 → 0.0925 mM, PID 0.0499 → 0.0500 mM, Hopsan vs scipy 최대 오차 1.47×10⁻⁴ mM | `hopsan/results/reactor/summary.txt` |

## 부록 B — 현실적 합성 데이터 (`evals/synthetic.yaml`로 옮긴다)
명세 RK2의 항목을 수치로 정한 것이다. 항목은 사용자가 승인했고 수치는 Claude가 정했다.
- 정답 메커니즘·상수·세트: CADKIMM `analysis/pinn_full_mechanism.py`와 같다(9종 10반응, TRUE_KF·TRUE_KR4, K' = 23, 세트 A·B 학습, C·D' 홀드아웃, A340 = 6.22·(NADH+EN+ENP), 관측 900 s까지 10 s 간격)
- 사전분포 중심: 정답 상수를 10의 거듭제곱으로 반올림한 값(문헌 자릿수만 안다는 가정), 표준편차 1.5 자릿수(log10)
- 잡음: 가우스 σ = 0.003 A340
- 혼합 사점: 첫 관측 5 s, 실제 시작 시각 오차 ±1 s(실험별 균등)
- 기준선 표류: 선형, 900 s당 −0.002~+0.002 A340(실험별 균등)
- 효소량 오차: 실험별 곱셈 오차 ±10%(균등). 식별 쪽은 명목 효소량만 안다
- 난수 시드는 고정해 둔다

## 부록 C — F-13 현실적 합성 데이터 (`evals/synthetic_general.yaml`로 옮긴다)
2026-10-10 보완 인터뷰에서 항목을 승인했고 수치는 Claude가 정한다(부록 B와 같은 방식). M6-0b에서 build가 한 번 만들고 그 뒤로는 바꾸지 않는다.
- 반응망 11개(LDH 순서형은 부록 B 그대로): 메커니즘 목록 8종(`uni_irreversible`·`uni`·`uni_iso`·`uni_bi_ordered`·`theorell_chance`·`random_bibi`·`ping_pong`·`ordered_terter`), LDH + 막다른 복합체 EQ·Pyr(정식 단계), LDH 경쟁 기질(피루브산 + 2-옥소부티르산, 전체 반응 둘), LDH + 효소 없는 NADH 분해(한 방향, 등급 C)
- 정답 속도 상수: 반응망마다 문헌 자릿수 수준에서 고른다. K′는 전체 반응마다
- 사전분포 중심: 정답을 10의 거듭제곱으로 반올림한 값, 표준편차 1.5 자릿수(log10)
- 잡음·혼합 사점·기준선 표류·효소량 오차: 부록 B와 같은 크기(신호가 A340이 아니면 신호 폭에 대한 같은 상대 크기)
- 세트: 반응망마다 학습 2 + 홀드아웃 2(홀드아웃 하나는 다른 적합류). 전체 반응이 둘 이상이면 신호 둘까지
- 난수 시드는 고정한다. 세트 수가 식별에 모자라 보이면 설정을 바꾸지 말고 `evals/QUESTIONS.md`에 남긴다
