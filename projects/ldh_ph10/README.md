# ldh_ph10 — 실데이터 측정 기록지와 실행 순서

프로토콜: CADKIMM `experiments/프로토콜.md` 1절(가역 반응)·4절(설계 확인). 이 폴더의 `project.yaml`이 계산 설정이다.

## 측정마다 적을 것 (실험 노트)

| 항목 | 왜 필요한가 | `project.yaml` 어디에 |
|---|---|---|
| 저장액 실제 농도(NADH·NAD⁺는 340 nm, ε 6.22로 확인) | 초기 농도가 틀리면 식별·외삽이 함께 틀린다 | 실험의 `NADH`, `NAD`, `Pyr`, `Lac` |
| 효소 작업액 농도·희석·넣은 부피 → 최종 효소량 E₀ | 피펫 표의 "E₀의 몇 배"의 기준 | `Et_nM` (단백 농도 ÷ 분자량으로 대략이면 됨) |
| 사점 시간(효소 투입 → 첫 측정) | 기기 시계가 첫 측정에서 0이면 시간을 그만큼 밀어야 함 | `time_offset_s` |
| 큐벳 온도(1 s 간격) | 등온 전제 확인(변동 < 0.2 ℃) | 같은 파일의 `T_C` 열 또는 `temp_file` |
| 효소 추가 투입 뒤 흡광 불변 여부 | 효소 실활(한 방향 단계) 배제 | 노트에만 |
| pH(완충액 실측) | K'가 pH 1당 약 10배 바뀜 | `Kp`(pH 9.5면 72) |

## 파일

- 이름: `data/set<세트>_<반복>.csv` (예: `setA_1.csv`). 열: `time_s,A340` 또는 `time_s,A340,T_C`(UTF-8, 머리글 필수).
- 기기 내보내기가 다르면(탭 구분, 쉼표 소수점, 분 단위, 머리줄 등) 원본을 `data/raw/`에 두고 실험 항목에 `format:`을 적는다. 예:
  `A_1: {file: data/raw/A_1.txt, format: {delimiter: tab, decimal: ",", skip_rows: 2, time_col: 0, value_col: 1, temp_col: 2, time_unit: min, encoding: cp949}, ...}`
- 원본은 지우거나 고치지 않는다(도구는 읽기만 한다).

## 날짜별 순서

| 날짜 | 실험 | 명령 | 확인할 것 |
|---|---|---|---|
| 10/10 | 파일럿: 세트 A 1회, 효소 적정(반감기 2~5분) | `uv run enzopt check projects/ldh_ph10` | 파일이 읽히는지, 끝값이 예상 평형(≈ 0.92) 근처인지, 끝 기울기가 0에 가까운지 |
| 10/11 | 세트 A·B·C·D' × 2회 | `uv run enzopt check projects/ldh_ph10` | A·B·C 끝값이 서로 5% 안이고 예상 근처, D'는 뚜렷이 다름(≈ 0.63). 어긋나면 pH·온도부터 |
| 10/12 | — | `uv run enzopt run projects/ldh_ph10` | ③ 식별: 표본 수·학습 RMSE, ⑤ 설계: ρ*·Et*·피펫 표(`5_design/pipette.txt`) |
| 10/13 | 설계 확인 세트 L1·L2·D1·D0·Z (피펫 표대로) | 측정 파일을 `design-check`로 넣고 `uv run enzopt run projects/ldh_ph10` | ⑤ 그림 (c): 측정점이 예측 곡선 위에 오는지, D1은 300 s에 99% 이상, L1·L2 끝값이 효소량과 무관한지 |
| 제출 전 | — | `uv run enzopt pack runs/ldh_ph10/<실행 폴더>` | 대회 결과 파일 묶음 |

`enzopt check`는 판정과 데이터 검사만 임시 폴더에서 돌린다(결과를 남기지 않음). 아직 없는 파일은 이름을 대고 알려 준다.
