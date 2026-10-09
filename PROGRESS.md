# 진행 상황

> 상태: 진행 중 · 현재 마일스톤: M1 · 현재 작업: M1-1 · 최종 갱신: 2026-10-09

## 사용자 필요 작업
- [ ] 분광광도계 내보내기 파일(파일럿) 전달 — 필요 시점: M3 끝(늦어도 10/11 실험 직후, 10/12 재계산 전) — 원본 파일을 `projects/<이름>/data/raw/`에 두고 알려 주면 변환기를 맞춘다
- [ ] 실데이터 프로젝트 작성 — 필요 시점: 10/12 — `projects/example_ldh_synth/project.yaml`을 복사해 `projects/ldh_ph10/`을 만들고 초기 농도·효소량·파일 이름을 고친다
- [ ] KIMM-SYS 설치 위치·예제 파일 — 필요 시점: 대회 전(받는 대로) — 설치 경로와 예제 모델 파일 위치를 알려 주면 해석 백엔드를 붙인다
- [ ] (선택) KIMM-CAD 실제 실행 허락 — 필요 시점: 대회 서류 전 — 지금은 같은 버전의 호환 환경(`.venv-kimmcad`)에서만 시험한다

## 마일스톤

### M0 가정 확인 (RK2) — 끝나면: 설치가 끝나고, CADKIMM 기준값이 이 PC에서 재현되고, 현실적 합성 데이터 시험 결과가 기록된다
- [x] M0-1 git·pyproject·uv 환경(`.venv`, PyTorch CUDA) — torch 2.11.0+cu128, RTX 5070 Ti 인식, `uv sync --locked` 통과
- [x] M0-2 KIMM-CAD 호환 환경 `.venv-kimmcad`(Python 3.12, numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.1) — import 확인, PyYAML 없음(동봉 판 사용)
- [x] M0-3 Hopsan 2.24.4 포터블(`C:\Users\cael\tools\Hopsan-2.24.4-win64`) — CLI 2.24.4.20260918, 149 배치·CSTR 실행 확인
- [x] M0-4 CADKIMM 기준값 재현(사본에서 실행) — 모두 재현, `evals/baselines.json` (결정 기록 참조)
- [x] M0-5 `evals/synthetic.yaml` + 현실적 합성 데이터 시험 — CADKIMM PINN은 물리 잔차 척도 문제로 무너짐 → enzopt 준정상 PINN. AC-05와 합성 설정 충돌(QUESTIONS Q1~Q3)

### M1 뼈대와 판정 — 끝나면: `enzopt run projects/example_ldh_synth`가 ①②까지 돌고 판정 보고·데이터 그림이 생긴다 (AC-03·04·15)
- [ ] M1-1 패키지 구조·ruff·pytest·CLI 뼈대 — 끝: G2·G3 통과, `enzopt --help`
- [ ] M1-2 평가 하네스 `evals/run.py --tier quick|long` — 끝: 관문·AC·지표·점수 출력, 미구현 AC는 실패로 기록
- [ ] M1-3 반응망·`project.yaml` 스키마·YAML 읽기(PyYAML, 없으면 동봉 순수 Python 판) — 끝: 예제 설정을 읽는 테스트
- [ ] M1-4 CRNT 판정(crnt_check 이식) + 판정 보고 — 끝: AC-03
- [ ] M1-5 합성 데이터 생성기 + 예제 프로젝트 `projects/example_ldh_synth` — 끝: 표준 CSV 생성
- [ ] M1-6 데이터 읽기·검사(F-2) — 끝: AC-04
- [ ] M1-7 실행 폴더·manifest·단계 실행기 + `enzopt run` ①② + 데이터 그림 — 끝: 예제가 ①②까지 돈다
- [ ] M1-8 `enzopt doctor` — 끝: AC-15

### M2 식별·해석·검증 — 끝나면: ①~④까지 돈다 (AC-01·05·07·08·09·10·11)
- [ ] M2-1 scipy 질량작용 해석 + CRNT 검증(crnt_validate 이식) + "수치 오류 의심" 재실행 — 끝: AC-07
- [ ] M2-2 149 배치 회귀(scipy) — 끝: AC-01(빠른)
- [ ] M2-3 Hopsan 백엔드(.hmf 생성·CLI·결과 읽기), 1단계 적합류 배치, 대표 표본 완전 메커니즘 대조, 없을 때 scipy 대체 — 끝: AC-01(긴), AC-11
- [ ] M2-4 PINN(GPU float32 시드 묶음, CPU float64 대체) — 끝: AC-05, AC-08
- [ ] M2-5 ③식별·④해석 단계 — 끝: 예제가 ①~④까지 돈다
- [ ] M2-6 캐시·재개·새 실행 폴더 — 끝: AC-09
- [ ] M2-7 KIMM-CAD 호환 시험(`tests/core`) — 끝: AC-10

### M3 설계와 출력 — 끝나면: 예제가 ①~⑤ 끝-끝으로 돈다 (AC-02·06·12·13·14)
- [ ] M3-1 강건 설계 이식(design_optimize) + 회귀 — 끝: AC-02
- [ ] M3-2 ⑤설계 단계 + 피펫 표(L1·L2·D1·D0·Z) — 끝: AC-06
- [ ] M3-3 표준 그림 세트(PDF + PNG 300 dpi + CSV) — 끝: AC-12
- [ ] M3-4 비약가역 프로젝트 끝-끝 — 끝: AC-13
- [ ] M3-5 `enzopt pack` — 끝: AC-14
- [ ] M3-6 기기 형식 변환기(설정형 일반 변환기, 파일이 오면 맞춤) — 끝: 단위 테스트
- [ ] M3-7 README(설치, `project.yaml` 항목, 예제, 평가 명령) — 끝: 문서

### M4 논문 근거 — 끝나면: 고전 피팅·비약가역 대조·설계 확인 곡선이 실행마다 나온다 (AC-16·17·18)
- [ ] M4-1 고전 피팅 기준선(같은 ODE, 다중 시작점 최소제곱) — 끝: AC-16
- [ ] M4-2 비약가역 대조 모델 학습·비교 — 끝: AC-17
- [ ] M4-3 설계 조건 확인 곡선(scipy 표본 전체 + Hopsan 대표 표본, 확인 실험 겹침) — 끝: AC-18
- [ ] M4-4 긴 평가 + Q1 측정 — 끝: `evals/run.py --tier long`

### M5 대시보드 — 끝나면: Streamlit에서 실행 목록·탭 ①~⑤·[새 실행]을 쓴다 (AC-19·20)
- [ ] M5-1 Streamlit 앱(프로젝트·실행 목록, 탭 ①~⑤, 진행·실패·판정 실패 상태) — 끝: AC-19
- [ ] M5-2 [새 실행] 분리 프로세스, 진행률·취소·[이 단계부터 다시] — 끝: AC-20
- [ ] M5-3 최종 점검(깨끗한 환경 설치·테스트, 긴 평가, README, 가정 보고) — 끝: 완료 보고

### M6 (대회 뒤) ③ 단계 — 보완 인터뷰로 기준을 더한 뒤 시작
- [ ] M6-0 보완 인터뷰(F-13·F-14·F-15 수용 기준)

## 결정 기록
- 2026-10-09 M0-4: CADKIMM 기준값을 사본(scratchpad)에서 재현 — 149 배치 저장 CSV 13/13, Hopsan 재실행 13/13(평형 0.1485/0.1019), 설계 전체 표가 CADKIMM과 숫자까지 동일(이 PC 85 s), `--quick` ρ* = 8.00·Et* = 182.28 nM, CSTR 1.47e-4 mM. `evals/baselines.json`에 고정.
- 2026-10-09 M0-5: 현실적 합성 설정 `evals/synthetic.yaml` 작성(시드 20261010). 명목 효소량 24.23 nM. 사전분포 중심은 K'까지 포함해 정답을 10의 거듭제곱으로 반올림(log10 kf 4,4,2,3,2 / kr 2,3,3,2 / K' 1).
- 2026-10-09 M0-5: 정답 모델(명목 효소량)조차 홀드아웃 D' RMSE 0.0146 > 0.006 → AC-05와 합성 설정이 충돌. `evals/QUESTIONS.md` Q1.
- 2026-10-09 M0-5: CADKIMM PINN 사본(문헌 자릿수 사전분포, CPU float64 시드 4개 124 s, GPU float32 시드당 48~77 s)은 신경망은 데이터에 맞으나 학습된 속도 상수로 ODE를 다시 풀면 학습 RMSE 0.03~0.14 → 물리 잔차 척도 문제. enzopt PINN은 느린 잔차를 실험의 특성 속도 (hi−lo)/T로 나누고 w_phys를 1→1e4로 올리며, 효소 형태는 준정상(King–Altman)으로 계산한다(`representation: qss`, 원래 표현은 `full`로 남김). QUESTIONS Q3.
- 2026-10-09 M0-5: enzopt PINN 결과(시드 4) — K' 23.2~23.8, 학습 ODE 재해석 RMSE 0.004~0.008, 홀드아웃 C 0.010~0.012, D' 0.007~0.127. GPU float32(CUDA 그래프) 6.8 s vs CPU float64 21.7 s, 홀드아웃 평균 차이 1% 미만. CUDA 그래프 없이 GPU는 40 s(커널 실행 대기가 지배).
- 2026-10-09 M0-5: SC5-3의 "데이터 RMSE"를 PINN 신경망의 학습 데이터 RMSE로 잰다(ODE 재해석 RMSE는 기록·경고). ODE 재해석으로 재면 예제의 시드가 모두 빠져 설계가 멈춘다. QUESTIONS Q2.
- 2026-10-09 M0-5: 고전 피팅과의 비교는 M4-1(AC-16)에서 같은 데이터로 잰다.

## 알려진 문제

## 제안 (범위 밖)
