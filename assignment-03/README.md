# Titanic 특성 공학 및 생존 예측

Titanic 공개 데이터 891행으로 결측치 처리, 인코딩, 스케일링, 파생 변수와 변수 선택을 비교합니다. Logistic Regression과 Random Forest를 같은 층화 분할에서 평가합니다.

## 재현

Python 3.12에서 이 폴더로 이동한 뒤 아래 명령을 실행합니다.

```sh
python -m pip install -r requirements.txt
python experiment.py
python report.py --repo-url https://github.com/Sindongyeob/bigdata_homework/tree/main/assignment-03
```

`experiment.py`는 고정된 [공개 CSV 사본](https://github.com/datasciencedojo/datasets/blob/f0ccab6a7ceafdff780052166fb6fab3311398eb/titanic.csv)을 내려받고, `results/`에 평가 지표와 그림을 생성합니다. 기존 CSV를 쓰려면 `--csv`를 지정합니다. `report.py`는 Windows의 맑은 고딕을 기본으로 사용합니다. 다른 환경에서는 한글 TrueType 글꼴 파일을 `--font`로 지정합니다. 저장소 주소는 실제 공개 또는 채점자가 접근 가능한 주소를 사용합니다. `--repo-url`을 생략하면 미등록 상태가 PDF에 표시됩니다.

## 실험 설계

- 80:20 층화 홀드아웃과 학습 데이터 내 층화 5-fold CV를 사용합니다. 난수 시드는 42입니다.
- Base에는 결측이 없는 원본 수치·순서형 4개 열을 그대로 입력합니다. Exp-1부터 Exp-3까지는 7개 원본 특성에 가족 규모와 가족 규모당 요금을 추가합니다.
- `ColumnTransformer`와 `Pipeline` 안에서 대치, 인코딩, 스케일링, SelectKBest 및 분류기를 각 폴드의 학습 데이터로 적합합니다.
- 필수 조합은 홀드아웃에 보고합니다. 개별 요소의 차이는 동일한 검증 폴드에서 한 번에 한 요소씩 바꿔 비교합니다.
- 두 모델별로 4개 필수 조합 중 검증 ROC-AUC가 가장 높은 조합에 `GridSearchCV`를 적용합니다.

## 주요 산출물

| 파일 | 내용 |
| --- | --- |
| `results/data_summary.json` | 표본 수, 범주별 비율, 실행 버전, 원본 해시 |
| `results/holdout.csv` | 4개 필수 조합의 홀드아웃 분류 지표 5종 |
| `results/cv_folds.csv` | 조합·모델·폴드별 학습과 검증 점수 |
| `results/paired.csv` | 한 요소만 바꾼 비교와 학습–검증 격차 |
| `results/selection.csv` | Exp-2·Exp-3의 ANOVA 점수와 선택 여부 |
| `results/grid.json`, `results/tuned.json` | 탐색 후보와 최종 모델 지표 |
| `results/permutation.csv` | 조정된 Random Forest의 순열 중요도 |
| `과제3_Titanic_특성공학.pdf` | 한국어 보고서 |
