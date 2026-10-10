# enzopt

속도 상수를 데이터로 다 정할 수 없는 효소 반응에서, 반응식과 진행 곡선으로 CRNT 구조 판정 → PINN 식별(속도 상수 표본) → 해석(scipy·Hopsan) + CRNT 자동 검증 → 강건 설계(피펫 표)를 잇는 로컬 연구 도구. 명세는 PROJECT_SPEC.md, 진행 상태는 PROGRESS.md를 따른다.

## 스택
- Python 3.12, uv(`uv.lock`), 패키지 `src/enzopt`, 명령 `enzopt run|pack|doctor|synth|dashboard`
- 계산: numpy, scipy(Radau·최소제곱), PyTorch CUDA 12.8 빌드(PINN, RTX 5070 Ti), matplotlib(논문 그림), PyYAML(없으면 `enzopt/_vendor/yaml`의 순수 Python 판)
- 대시보드: Streamlit(결과 폴더를 읽기만 하고, 계산은 별도 프로세스)
- 해석 백엔드: scipy(완전 메커니즘 표본 × 조건), Hopsan 2.24.4 CLI(`C:\Users\cael\tools\Hopsan-2.24.4-win64`, 1단계 적합류 배치 + 대표 표본 대조). KIMM-SYS는 형식을 받으면 백엔드로 추가
- DB 없음. 결과는 `runs/<프로젝트>/<YYYY-MM-DD_HHMM>/`의 파일(git 밖). 입력은 `projects/<이름>/project.yaml` + `data/`(git 안)
- `src/enzopt/core/`는 KIMM-CAD 내장 Python과 호환(numpy 1.26.4, scipy 1.13.1, matplotlib 3.9.1)해야 한다: torch·streamlit·PyYAML을 import하지 않고, numpy 2 전용 API(`np.trapezoid` 등)를 쓰지 않는다

## 명령
- 설치: `uv sync` (호환 환경은 하네스가 `.venv-kimmcad`에 만든다)
- 실행: `uv run enzopt run projects/example_ldh_synth` (`--quick` 짧은 모드, `--resume <실행 폴더>`)
- 점검: `uv run enzopt doctor`
- 빠진 반응 단계 제안: `uv run enzopt suggest projects/<이름>`
- 반응식 고르기(메커니즘 목록): `uv run enzopt mechanisms`
- 대시보드: `uv run enzopt dashboard`
- 테스트: `uv run pytest -m "not long" -q` (긴 테스트 포함: `uv run pytest -q`)
- 린트·포맷: `uv run ruff check .` / `uv run ruff format --check .`
- 평가(관문 + 수용 기준 + 지표 → 점수): `uv run python evals/run.py --tier quick` / `--tier long`

## 작업 규칙
- 작업을 시작하고 끝낼 때 PROGRESS.md를 갱신한다.
- 완성과 개선의 기준은 evals/EVAL.md다. 이 파일과 evals/ 하네스·`baselines.json`·`synthetic.yaml`·`synthetic_general.yaml`은 고치지 않고, 기준에 문제가 있으면 evals/QUESTIONS.md에 적는다.
- CRNT 판정·검증 기준과 허용 오차(보존 표류 1e-6, |ln Q/K| 0.01, 리아푸노프 비증가, AC-05 문턱)는 바꾸지 않는다.
- 명세에 없는 결정은 추천안으로 진행하고 PROGRESS.md "결정 기록"과 PROJECT_SPEC.md "가정"에 적는다. 단 과학적 판단(사전분포 값, 수렴·판정 기준, 허용 오차)은 evals/QUESTIONS.md에 남긴다. 되돌리기 어려운 결정은 묻는다.
- CADKIMM(`C:\Users\cael\Desktop\CADKIMM`)은 손대지 않는다(대회 기록). 기준값 재현은 사본에서 한다.
- 테스트 수준: CRNT 판정·검증·설계·데이터 읽기 단위 테스트 + CADKIMM 기준값 회귀 + 합성 예제 끝-끝(짧은 모드).
- 커밋: `enzopt/` 로컬 git, 작업 단위 커밋, 원격 없음.
- 코드 규칙: ruff 기본(줄 120), 공개 함수에 타입 힌트 · 주석·docstring은 영어, README·명세·PROGRESS·보고는 한국어. 그림 축·범례는 영어, 대시보드 문구는 간결한 명사형 한국어.
- 단위: 농도 mM, 시간 s, 효소 입력 nM(내부 mM). 파일은 모두 UTF-8로 읽고 쓴다.
- 확인 빈도: 끝까지 진행하고 마일스톤마다 짧게 보고(멈추지 않음).
