# 과제 4 · 스트리밍 알고리즘 비교

MovieLens 1M의 영화 평점 1,000,209개를 ZIP에서 한 줄씩 읽습니다. Bloom Filter의 멤버십 판정과 Count-Min Sketch의 영화별 평점 건수 추정을 각각 정확한 집합과 빈도 사전에 대조합니다.

## 재현

Python 3.12에서 이 폴더로 이동한 뒤 다음 명령을 실행합니다. 벤치마크에는 표준 라이브러리만 필요하며, 그래프와 PDF 생성 시 의존성을 설치합니다.

```sh
python benchmark.py
python -m pip install -r requirements.txt
python report.py
```

기본 다운로드 출처는 [GroupLens MovieLens 1M](https://grouplens.org/datasets/movielens/1m/)입니다. 직접 받은 공식 ZIP을 사용하려면 `python benchmark.py --archive /path/to/ml-1m.zip`을 실행합니다. 다른 운영체제에서 PDF를 만들 때는 `--font`로 한글 TrueType 글꼴을 지정합니다. Windows 기본 글꼴은 맑은 고딕입니다.

## 측정 방법

- **입력**. ZIP 안의 `ratings.dat`에서 `MovieID`만 파싱합니다. 파일 순서 그대로 1회 스캔하며 원본 레코드를 저장하지 않습니다.
- **Bloom Filter**. 비트 배열을 직접 갱신합니다. 관측된 모든 ID와 고정 시드로 뽑은 미관측 ID 10,000개에 대해 FPR·FNR을 계산합니다.
- **Count-Min Sketch**. 평탄화한 카운터 배열의 각 행에서 한 칸씩 증가시킵니다. 추정값은 행 카운터들의 최솟값입니다. 관측 ID 전체에서 MAE·평균 상대오차·정확 일치율 등을 계산합니다.
- **메모리**. CPython의 `sys.getsizeof`로 구성 객체의 크기를 합산하고, 동일 객체는 한 번만 셉니다. 비트·카운터 배열의 순수 바이트 수도 기록합니다.
- **시간**. 새 구조 생성, ZIP 스트리밍, 파싱, 삽입을 합친 경과 시간입니다. 설정별 기본 3회 반복하며 순서를 라운드마다 섞습니다. 다운로드, 별도 정답 생성, 질의 평가는 제외합니다.

`results/benchmark.json`과 `results/benchmark.csv`에 정확도·메모리·시간 결과가 생성됩니다. `report.py`는 그래프와 한국어 제출용 PDF를 생성합니다.
