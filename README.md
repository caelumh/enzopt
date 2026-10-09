# enzopt

> 속도 상수를 몰라도, 효소 반응 조건은 정할 수 있습니다.

반응식과 진행 곡선을 넣으면 **① CRNT 구조 판정**(정리 149) → **② 데이터 검사** → **③ PINN 식별**(속도 상수 표본) → **④ 해석 + CRNT 자동 검증**(scipy·Hopsan) → **⑤ 강건 설계**(평형 한계, 최소 효소량, 피펫 표)가 한 번에 이어지는 로컬 연구 도구입니다. 명령(`enzopt run`)과 대시보드([새 실행])가 같은 엔진과 같은 결과 폴더를 씁니다.

명세는 `PROJECT_SPEC.md`, 진행 상태는 `PROGRESS.md`, 평가 기준은 `evals/EVAL.md`에 있습니다.

## 설치

필요한 것: Windows 11, [uv](https://docs.astral.sh/uv/), (선택) NVIDIA GPU, (선택) Hopsan 2.24.4.

```
uv sync                       # Python 3.12 + numpy/scipy/matplotlib/PyYAML/Streamlit + PyTorch(CUDA 12.8 빌드)
uv run enzopt doctor          # 버전·CUDA·Hopsan·디스크 점검. 빠진 구성 요소를 이름으로 알려 준다
```

- PyTorch는 `pyproject.toml`의 `pytorch-cu128` 인덱스에서 받습니다(RTX 50 계열 Blackwell 지원). GPU가 없으면 PINN이 CPU(float64)로 자동 전환되고 manifest에 기록됩니다.
- Hopsan 2.24.4 포터블 zip(`Hopsan-2.24.4-win64-with_compiler-zip.zip`, [GitHub 릴리스](https://github.com/Hopsan/hopsan/releases/tag/v2.24.4))을 풀어 둔 위치를 다음 순서로 찾습니다: 환경 변수 `HOPSANCLI`(전체 명령) → `HOPSAN_HOME` → `enzopt.config.yaml`의 `hopsan.cli` → PATH → `~/tools/Hopsan-*/bin`. 이 PC는 `C:\Users\cael\tools\Hopsan-2.24.4-win64`. 못 찾으면 해당 계산을 scipy로 대신하고 "Hopsan 미사용"을 남깁니다.

## 빠른 시작

```
uv run enzopt run projects/example_ldh_synth --quick   # 짧은 모드(시드 2, 반복 축소) ≈ 30 s
uv run enzopt run projects/example_ldh_synth           # 전체 모드(시드 4, Hopsan 포함) ≈ 2~3분
uv run enzopt dashboard                                # 대시보드 (http://localhost:8501)
uv run enzopt pack runs/example_ldh_synth/<실행 폴더>   # 제출용 zip 하나
```

실데이터 프로젝트: 측정용 틀 `projects/ldh_ph10/`(측정 기록지 README 포함)을 채운다. 실측 문헌 데이터로 미리 시험한 예는 `projects/ldh_holmes1988/`(Holmes 1988, 토끼 근육 LDH pH 7.0). 처음부터 만들 때는 `projects/example_ldh_synth`를 복사해 `projects/<이름>/`을 만들고, `data/`에 진행 곡선 CSV를 넣은 뒤 `project.yaml`의 `name`과 `experiments`(초기 농도·효소량·파일·역할)를 고칩니다.

| 명령 | 하는 일 |
|---|---|
| `enzopt run <프로젝트>` | 새 실행 폴더를 만들고 ①~⑤를 돈다. 입력이 같은 단계는 이전 실행 결과를 재사용 |
| `enzopt run <프로젝트> --no-cache` | 재사용 없이 전부 다시 계산 |
| `enzopt run --resume <실행 폴더>` | 중단된 실행을 끊긴 단계부터 이어서 (`--from-stage 3_fit`처럼 특정 단계부터 다시) |
| `enzopt run ... --device cpu --precision float64` | PINN 장치·정밀도 지정(기본은 project.yaml) |
| `enzopt check <프로젝트>` | 측정 직후 점검: 판정·데이터 검사만 임시 폴더에서. 실험별 점 수·처음/끝값·K'로 푼 예상 평형과의 차이·끝 기울기·온도 |
| `enzopt pack <실행 폴더> [-o x.zip]` | 입력 데이터·설정 사본·그림·CSV·Hopsan `.hmf`·검증 보고를 zip 하나로 |
| `enzopt doctor [--hopsan]` | 환경 점검 |
| `enzopt synth evals/synthetic.yaml <프로젝트>` | 현실적 합성 데이터로 `data/` 작성(예제 생성용) |
| `enzopt dashboard [--port N]` | Streamlit 대시보드 |

## 입력: `projects/<이름>/`

```
projects/<이름>/
├─ project.yaml      반응망·실험·식별·해석·설계 설정 (이 파일이 재현의 기록)
└─ data/             진행 곡선·온도 CSV (원본, 읽기만)
```

데이터 표준 형식(UTF-8 CSV, 머리글 필수): 진행 곡선 `time_s,A340`, 온도 `time_s,T_C`(1 s 간격). 온도는 같은 파일의 세 번째 열(`time_s,A340,T_C`)로 넣어도 된다. 관측값 이름이 A340이 아니면(예: `NADH_mM`) 그 이름이 열 이름이고 A340 > 2 검사는 하지 않는다. 기기 시계가 효소 투입이 아닌 첫 측정에서 0이면 실험에 `time_offset_s`(사점 시간)를 적는다. 시간 역행·빈 값·숫자가 아닌 값·A340 > 2인 파일은 **파일 이름과 줄 번호**를 담은 오류로 멈춥니다. 분광광도계 내보내기 형식이 다르면 실험 항목에 `format:`을 적어 일반 변환기로 읽습니다(변환 결과는 실행 폴더에 저장, 원본은 그대로):

```yaml
  A: {file: data/raw/A.txt, format: {delimiter: tab, decimal: ",", skip_rows: 2, time_col: 0, value_col: 1, time_unit: min, encoding: cp949}, ...}
```

### `project.yaml` 항목

| 항목 | 뜻 |
|---|---|
| `name`, `description` | 프로젝트 이름(실행 폴더 `runs/<name>/`), 설명 |
| `network.species` | 종 목록(상태 벡터 순서) |
| `network.enzyme_species` | 효소 형태(합 = Et). 첫째가 자유 효소 |
| `network.steps` | 반응식. 가역은 `A + B <=> C`, 한 방향은 `C -> D`. 계수는 `2X + Y` |
| `network.observable` | 관측식. `coefficients: {NADH: 6.22, EN: 6.22, ENP: 6.22}` → A340 = 6.22·(NADH+EN+ENP) |
| `network.thermodynamics` | `overall`(전체 반응식, 적합류 생성에 씀), `Kp`(K'), `haldane_step`(K'로 k₋를 정하는 단계 번호) |
| `network.prior` | 사전분포: `kf_log10`(모든 단계), `kr_log10`(가역 단계 중 Haldane 단계 제외), `Kp_log10`, `sd_decades`(기본 1.5) |
| `control.one_way_steps` | 비약가역 대조 반응망(F-10): 이 단계들을 한 방향으로 |
| `experiments.<이름>` | 초기 농도(종 이름: mM), `Et_nM`, `role`(train / holdout / design-check), `file`, `temp_file`, (선택) `format`(`temp_col` 포함), `time_offset_s` |
| `identification` | `method: pinn`, `seeds`, `noise_sigma`(σ), `device`, `precision`, `pinn: {adam_iters, lbfgs_iters, restarts, representation: qss, time_scale: per_experiment}`, `baseline_fit: {starts, max_nfev}`(F-9), `control: {seeds}`(F-10), `exclusion_rmse: ode|pinn`, `exclusion_sigma`(기본 3) |
| `simulation` | `t_long_s`, `same_class_runs`, `class_seed`, `hopsan: {enabled, one_step_k1: auto, representative_experiment, max_steps}` |
| `design` | `target: assay`, `analyte`, `partner`(ρ = partner₀/analyte₀), `free_enzyme`, `analyte0_mM`, `conversion`, `time_s`, `Kp`, `dead_end: {step, enzyme_species, kf, Ki_mM}`, `rho_grid: auto`, `Et_bounds_mM`, `bisection_iters`, `check_t_end_s`, `pipette: {volume_uL, enzyme_uL, stocks_mM, limit_test_rhos, limit_test_enzyme_multiples}` |
| `quick` | 짧은 모드에서 바꿀 값(`seeds`, `same_class_runs`, `rho_grid`, `bisection_iters`, `baseline_starts`, `hopsan` …) |

단위: 농도 mM, 시간 s, 효소량은 nM으로 입력(내부 mM).

## 결과: `runs/<프로젝트>/<YYYY-MM-DD_HHMM>/`

```
manifest.json   설정 사본·입력 파일 해시·코드 버전(git 커밋, 소스 해시)·uv.lock 해시·시드·단계 상태·장치·경고
project.yaml    설정 사본          inputs/   입력 사본(이어서 실행·묶음용)
1_verdict/      판정 보고(report.txt, verdict.json), 복합체 그래프, 판정 표
2_data/         검사한 데이터(exp_*.csv, temp_*.csv), data.json, 진행 곡선·온도 그림
3_fit/          표본(samples/sample_seed<N>.json), 고전 피팅(baseline/), 비약가역 대조(control/), 홀드아웃 비교표, 그림 (a)~(d)
4_sim/          표본 × 적합류 해석의 CRNT 검증(validation.csv), Hopsan(.hmf·CSV), 그림 (a)~(c)
5_design/       U자 곡선 표, 설계 확인, 평형 한계, 피펫 표(pipette.csv·txt: L1·L2·D1·D0·Z), 그림 (a)~(d)
```

그림은 모두 PDF(벡터) + PNG(300 dpi) + **그린 값 그대로의 CSV**(같은 이름)로 저장됩니다. 축·범례는 영어, 색각 이상에도 구분되는 팔레트와 선 모양을 씁니다. 실행 결과는 지우지 않습니다(재실행은 새 폴더). 논문·대회에 쓴 실행은 `enzopt pack`으로 따로 보관하세요(`runs/`는 git 밖).

## 단계별 계산 요약

- **① 판정**: 복합체·연결 성분·강연결 성분(Kosaraju)·약가역성·결핍·정수 보존 법칙. 등급 A(약가역·결핍 0: 영속 + 적합류마다 평형 하나), B(약가역: 영속), C(약가역 아님: "149 보장 없음: 멸종 가능" 띠를 모든 결과에 남기고 계속 진행).
- **③ 식별**: LDH 순차 bi-bi PINN(보존 법칙 4개·Haldane 내장, 효소 형태는 준정상 King–Altman 해, 실험마다 NADH(t) 신경망). 표본(시드)마다 8번 재시작해 학습 데이터에 ODE로 가장 잘 맞는 해를 고릅니다(지역 최소에서 외삽이 갈리는 문제 방지). GPU float32 + CUDA 그래프로 시드 × 재시작을 묶어 학습(4 × 8 ≈ 10 s), GPU가 없으면 CPU float64. 학습된 속도 상수로 ODE를 다시 풀어 모든 실험의 RMSE를 기록하고, ODE 재해석 학습 RMSE > 3σ인 시드는 표본에서 뺍니다(남은 표본 < 2이면 ⑤에서 멈춤). 같은 데이터·사전분포·시작점으로 고전 피팅(다중 시작점 최소제곱)과 비약가역 대조 PINN을 함께 돌려 홀드아웃 외삽을 한 표로 비교합니다.
- **④ 해석**: 표본마다 같은 적합류의 무작위 초기 상태와 다른 적합류 하나를 scipy(Radau, 해석적 야코비안)로 풀고 CRNT 검증(보존 표류 < 1e-6, 양수, |ln Q/K| < 0.01, 리아푸노프 비증가, 적합류 일관성). 실패하면 "수치 오류 의심"으로 표시하고 Δt를 1/10로 줄여 한 번 다시 풉니다. Hopsan은 겉보기 1단계 반응의 적합류 배치와 대표 표본 하나의 완전 메커니즘(작은 Δt)을 풀어 scipy와의 최대 차이를 기록합니다.
- **⑤ 설계**: CRNT 평형 한계 ρ_min(약가역·결핍 0일 때만 대수식 사용) → ρ 격자에서 표본 전체를 만족하는 최소 효소량(강건)과 공칭 표본만 만족하는 효소량(공칭) → 강건 최적 ρ*·Et*(학습 실험 효소량 E₀의 배수) → 두 설계를 표본마다 넣은 전환율 → 피펫 표 → 설계 조건 예측 곡선(scipy 표본 전체 + Hopsan 대표 표본 + 확인 실험 점).

## KIMM-CAD 내장 Python 호환

`src/enzopt/core/`(판정·scipy 해석·검증·설계·YAML 읽기)는 numpy·scipy·matplotlib만 씁니다. PyYAML이 없으면 동봉한 순수 Python 판(`enzopt/_vendor/yaml`, PyYAML 6.0.3, MIT)을 씁니다. 같은 버전 환경(Python 3.12, numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.1)에서 시험:

```
uv venv .venv-kimmcad --python 3.12
uv pip install --python .venv-kimmcad/Scripts/python.exe numpy==1.26.4 scipy==1.13.1 matplotlib==3.9.1 pytest
set PYTHONPATH=src && .venv-kimmcad\Scripts\python.exe -m pytest tests/core
```

## 시험과 평가

```
uv run pytest -m "not long" -q              # 빠른 테스트(≈ 2~3분)
uv run pytest -q                            # 긴 테스트 포함
uv run python evals/run.py --tier quick     # 관문 + 빠른 수용 기준 → 점수 (≤ 5분)
uv run python evals/run.py --tier long      # 관문 + 모든 기준 + Q1(전체 모드 실행 시간) → 점수 (≤ 30분)
```

하네스는 결과를 `evals/last.json`, `evals/results.tsv`(git 밖)에 남깁니다. 기준·하네스·기준값(`evals/`)은 고치지 않습니다. 기준에 대한 질문은 `evals/QUESTIONS.md`에 있습니다.

## 알려 둘 것

- PINN 표현은 명세 방법론(4.2·4.3)의 "효소 형태 = 신경망 출력"이 아니라 **준정상 효소 분포**(King–Altman)입니다(2026-10-09 결정). 원래 표현은 `identification.pinn.representation: full`로 남아 있고, 실패 근거는 `docs/pinn_representation/`(`uv run python scripts/compare_pinn_representation.py`로 재현).
- 표본 제외(SC5-3): 학습된 속도 상수로 ODE를 다시 푼 학습 RMSE > 3σ인 시드는 뺍니다(`exclusion_rmse: ode`, `exclusion_sigma: 3`).
- AC-05 문턱은 정답 모델 대비입니다: 홀드아웃 RMSE ≤ max(2σ, 정답 모델 RMSE × 1.5). 효소량 ±10%에서는 정답 속도 상수로도 D' RMSE가 0.0146이기 때문입니다.
- 막다른 복합체 해리 상수 Ki = 5 mM은 가정값입니다(문헌 확인 필요).
- KIMM-SYS 백엔드는 형식을 받으면 붙입니다. 그때까지 Hopsan이 대체 도구입니다.

## 라이선스

MIT. 의존: Hopsan(Apache-2.0, 별도 프로세스로 호출), PyTorch·numpy·scipy(BSD 계열), matplotlib(PSF 계열), Streamlit(Apache-2.0), 동봉 PyYAML(MIT).
