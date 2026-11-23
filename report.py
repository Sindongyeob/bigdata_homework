"""Build the Korean PDF report from experiment.py outputs."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

from PIL import Image as PILImage
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Table, TableStyle,
)

from experiment import DEFAULT_OUTPUT, ROOT, URL

NAVY = colors.HexColor("#142737")
TEAL = colors.HexColor("#006b70")
SLATE = colors.HexColor("#4c6372")
PALE = colors.HexColor("#edf5f5")
GRID = colors.HexColor("#dce6e9")
WHITE = colors.white
CONTENT = 496

COLUMNS = [
    ("PassengerId", "정수", "승객 식별 번호", "제외"),
    ("Survived", "이진", "생존 여부 (0 사망, 1 생존)", "타깃"),
    ("Pclass", "순서형", "객실 등급 (1·2·3등)", "전체"),
    ("Name", "문자열", "승객 이름", "제외"),
    ("Sex", "범주형", "기록된 성별", "Exp-1~3"),
    ("Age", "수치형", "나이 (세)", "Exp-1~3"),
    ("SibSp", "정수", "동승 형제·자매 및 배우자 수", "전체"),
    ("Parch", "정수", "동승 부모 및 자녀 수", "전체"),
    ("Ticket", "문자열", "티켓 번호", "제외"),
    ("Fare", "수치형", "요금", "전체"),
    ("Cabin", "문자열", "객실 식별 정보", "제외"),
    ("Embarked", "범주형", "승선항 (C·Q·S)", "Exp-1~3"),
]
TITLES = [
    "데이터와 예측 목표", "탐색적 분석 | 분포와 이상치",
    "탐색적 분석 | 타깃과 상관관계", "특성 공학과 모델 학습",
    "필수 조합의 성능", "전처리 요소별 비교", "변수 선택과 모델 해석", "최종 결론",
]


class Report:
    def __init__(self, results, font):
        pdfmetrics.registerFont(TTFont("Korean", str(font)))
        bold_file = font.with_name("malgunbd.ttf") if font.name.lower() == "malgun.ttf" else font
        pdfmetrics.registerFont(TTFont("KoreanBold", str(bold_file if bold_file.exists() else font)))
        pdfmetrics.registerFontFamily("Korean", normal="Korean", bold="KoreanBold")
        self.results = results
        self.story = []
        self.styles = {
            "title": self.style("title", 29, 40, NAVY, bold=True, after=14),
            "section": self.style("section", 21, 31, NAVY, bold=True, after=12),
            "kicker": self.style("kicker", 10, 16, TEAL, bold=True, after=6),
            "sub": self.style("sub", 12, 19, TEAL, bold=True, before=10, after=4),
            "body": self.style("body", 10.3, 17.5, NAVY, after=9),
            "note": self.style("note", 9, 14.5, SLATE, after=8),
            "cell": self.style("cell", 8.8, 13, NAVY),
            "head": self.style("head", 8.8, 13, WHITE, bold=True),
            "number": self.style("number", 16, 22, TEAL, bold=True, center=True),
            "label": self.style("label", 8.8, 13, SLATE, center=True),
        }

    @staticmethod
    def style(name, size, leading, color, bold=False, center=False, before=0, after=0):
        return ParagraphStyle(
            name, fontName="KoreanBold" if bold else "Korean", fontSize=size,
            leading=leading, textColor=color, wordWrap="CJK",
            alignment=TA_CENTER if center else 0, spaceBefore=before, spaceAfter=after,
        )

    def p(self, text, style="body"):
        self.story.append(Paragraph(text, self.styles[style]))

    def heading(self, index):
        if index:
            self.story.append(PageBreak())
        self.p(f"과제 3  /  {index + 1:02d}", "kicker")
        self.p(TITLES[index], "section")

    def table(self, head, rows, widths, compact=False):
        cells = [[Paragraph(escape(str(value)), self.styles["head"]) for value in head]]
        cells += [[Paragraph(escape(str(value)), self.styles["cell"]) for value in row] for row in rows]
        padding = 4 if compact else 6
        table = Table(cells, colWidths=widths, repeatRows=1, hAlign="LEFT", spaceAfter=10)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [WHITE, PALE]),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, GRID),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
            ("TOPPADDING", (0, 0), (-1, -1), padding),
            ("BOTTOMPADDING", (0, 0), (-1, -1), padding),
        ]))
        self.story.append(table)

    def stats(self, metrics):
        cells = [[Paragraph(value, self.styles["number"]) for value, _ in metrics],
                 [Paragraph(label, self.styles["label"]) for _, label in metrics]]
        table = Table(cells, colWidths=[CONTENT / len(metrics)] * len(metrics),
                      hAlign="LEFT", spaceAfter=12)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, -1), PALE),
            ("TOPPADDING", (0, 0), (-1, 0), 9),
            ("BOTTOMPADDING", (0, 1), (-1, 1), 8),
        ]))
        self.story.append(table)

    def img(self, name, caption, max_height, max_width=CONTENT):
        path = self.results / f"{name}.png"
        with PILImage.open(path) as item:
            width, height = item.size
        scale = min(max_width / width, max_height / height)
        self.story.append(Image(str(path), width=width * scale, height=height * scale))
        self.p(caption, "note")

    def load_json(self, name):
        return json.loads((self.results / f"{name}.json").read_text(encoding="utf-8"))

    def load_csv(self, name):
        return pd.read_csv(self.results / f"{name}.csv")

    def build(self, output, repo_url):
        data = self.load_json("data_summary")
        holdout = self.load_csv("holdout")
        folds = self.load_csv("cv_folds")
        paired = self.load_csv("paired")
        selected = self.load_csv("selection")
        tuned = self.load_json("tuned")
        missing = self.load_csv("missing").set_index("column")
        outliers = self.load_csv("outliers").set_index("column")
        corr = self.load_csv("correlation").set_index("column")
        importance = self.load_csv("permutation").set_index("feature")
        cv = folds.groupby(["experiment", "model"])["val_auc"].agg(["mean", "std"])
        best = max(tuned, key=lambda row: row["auc"])
        best_params = " · ".join(f"{key.split('__')[-1]}={value}" for key, value in best["params"].items())
        shape = tuple(data["shape"])
        train_size, test_size = data["train_rows"], data["test_rows"]
        baseline = holdout.set_index(["experiment", "model"])
        p1, p2, p3 = (data["class_survival"][str(i)] for i in (1, 2, 3))

        def pair(before, after, model):
            return paired[(paired["before"] == before) & (paired["after"] == after)
                          & (paired["model"] == model)].iloc[0]

        self.p("과제 3  /  Titanic 생존 분류", "kicker")
        self.p("특성 공학 파이프라인<br/>구현 및 성능 비교", "title")
        self.stats([(str(shape[0]), "전체 승객"), (str(shape[1]), "원본 열"),
                    (str(test_size), "홀드아웃 승객")])
        self.p("Titanic 공개 탑승객 기록에서 승객의 생존 여부를 예측한다. "
               f"원본 데이터의 shape은 <b>{shape}</b>이며, 타깃 Survived는 사망 0과 생존 1로 구성된다. "
               "수치형·순서형 특성과 범주형 특성을 함께 사용해 전처리 조합을 비교했다.")
        self.p("원본 컬럼 설명", "sub")
        self.table(("컬럼", "유형", "설명", "모델 사용"), COLUMNS, [93, 67, 245, 91], compact=True)
        self.p("‘전체’는 Base와 Exp-1~3에서 사용한 열이다. PassengerId·Name·Ticket은 식별성이 높아 "
               "제외했고, Cabin은 결측 비율이 높아 제외했다. Pclass는 순서가 있는 숫자로 입력했다.", "note")
        self.p("출처  "
               '<link href="https://www.kaggle.com/competitions/titanic/data" color="#006b70">Kaggle Titanic</link>  /  '
               f'<link href={quoteattr(URL)} color="#006b70">분석에 사용한 고정 CSV 사본</link>', "note")

        self.heading(1)
        self.p(f"전체 행의 완전 일치 중복은 {data['duplicates']}건이다. "
               f"아래 EDA는 홀드아웃을 분리한 뒤 학습용 {train_size}행에 대해 수행했다. "
               "결측치 표에는 원본 전체의 결측 건수를 함께 제시했다.")
        miss_rows = [
            (name, int(row["all_missing"]), int(row["train_missing"]), f"{row['train_pct']:.2f}%")
            for name, row in missing.sort_values("train_missing", ascending=False).iterrows()
            if row["all_missing"] > 0
        ]
        miss_rows.append(("나머지 9개 열", 0, 0, "0.00%"))
        self.table(("열", "전체 결측", "학습 결측", "학습 결측률"), miss_rows, [141, 110, 110, 135])
        self.img("distributions", "그림 1. 학습 데이터의 나이·요금 히스토그램과 박스플롯. 결측 관측치는 그림에서 제외했다.", 265)
        self.p("IQR 기준 이상치 후보", "sub")
        out_rows = [(name, int(row["observed"]), f"{row['lower']:.2f} ~ {row['upper']:.2f}",
                     int(row["outliers"])) for name, row in outliers.iterrows()]
        self.table(("변수", "유효 관측", "1.5 × IQR 기준 범위", "범위 밖"), out_rows,
                   [88, 101, 204, 103], compact=True)
        self.p(f"요금 중앙값은 {data['train_fare_median']:.2f}, 최댓값은 {data['train_fare_max']:.2f}로 "
               "오른쪽 꼬리가 길다. Parch는 Q1과 Q3가 모두 0이므로 양수 관측치가 모두 표시된다. "
               "큰 요금이나 동승 인원은 실제 관측값일 수 있어 실험에서 유지했다.", "note")

        self.heading(2)
        died, survived = (data["all_classes"][key] for key in ("0", "1"))
        self.p(f"전체에서 사망 {died}명({died / shape[0]:.1%}), 생존 {survived}명({survived / shape[0]:.1%})이다. "
               f"학습 데이터의 생존율은 여성 {data['sex_survival']['female']:.1%}, "
               f"남성 {data['sex_survival']['male']:.1%}였고, 등급별로는 1등 {p1:.1%}, "
               f"2등 {p2:.1%}, 3등 {p3:.1%}였다.")
        self.img("target", "그림 2. 학습 데이터의 타깃 빈도와 성별 생존율. 전체 분포는 사망 쪽으로 기울어 있다.", 160)
        self.img("correlation", "그림 3. 수치·순서형 특성과 생존 여부의 Pearson 상관계수. 결측은 변수 쌍별로 제외했다.", 302)
        self.p(f"Survived와의 상관은 Pclass r={corr.at['Pclass', 'Survived']:.2f}, "
               f"Fare r={corr.at['Fare', 'Survived']:+.2f}, Age r={corr.at['Age', 'Survived']:.2f}였다. "
               f"Pclass와 Fare도 서로 연관되어(r={corr.at['Pclass', 'Fare']:.2f}) 변수별 효과를 인과적으로 해석하기 어렵다. "
               "Age의 낮은 선형 상관만으로 나이의 비선형 예측력을 판단하기도 어렵다.", "note")

        self.heading(3)
        config_rows = [
            ("Base", "없음", "없음", "없음", "없음", "4 → 4"),
            ("Exp-1", "Mean", "One-Hot", "Standard", "없음", "12 → 12"),
            ("Exp-2", "Median", "Label", "MinMax", "상위 6", "9 → 6"),
            ("Exp-3", "Most Frequent", "One-Hot", "Robust", "상위 6", "12 → 6"),
        ]
        self.table(("실험", "수치 결측", "인코딩", "스케일링", "변수 선택", "입력 열 수"),
                   config_rows, [61, 98, 72, 80, 91, 94])
        self.p("입력 열 수는 전체 학습 세트 적합 시 인코딩 후 → 선택 후 기준이다. Base는 모든 승객에서 "
               "완전한 Pclass·SibSp·Parch·Fare만 그대로 전달한다. 이 정의로 네 조합에서 동일한 승객을 평가한다.")
        self.p("파생 변수와 전처리", "sub")
        self.p("<b>FamilySize</b> = SibSp + Parch + 1. 본인을 포함한 가족 규모다.<br/>"
               "<b>FarePerPerson</b> = Fare / FamilySize. 가족 규모로 정규화한 요금이며 "
               "실제 티켓 인원이나 개인별 지불액을 의미하지 않는다.")
        self.p("Exp-1~3은 두 파생 변수를 생성한 뒤 수치형은 대치 → 스케일링, Sex·Embarked는 "
               "최빈값 대치 → 인코딩을 적용한다. Label은 예측 변수용 OrdinalEncoder로 구현했다. "
               "새 범주는 One-Hot에서 해당 블록을 0으로, Label에서 -1로 처리한다. "
               "Standard는 평균·표준편차, MinMax는 최솟값·최댓값, Robust는 중앙값·IQR을 사용한다.")
        self.p("분할, 학습 및 평가", "sub")
        self.p(f"생존 비율을 유지해 학습 {train_size}명과 홀드아웃 {test_size}명으로 80:20 분할했다. "
               "학습 세트 안에서 층화 5-fold를 구성하고 난수 시드 42를 공통으로 사용했다. "
               "Pipeline과 ColumnTransformer를 통해 각 폴드의 학습 부분에서 대치·인코딩·스케일링·"
               "SelectKBest(ANOVA F)·모델을 적합했다. 기본 분류기는 Logistic Regression(LR, C=1)과 "
               "Random Forest(RF, 트리 200개, 깊이 제한 없음, 최소 잎 표본 1)다.")
        self.p("CV에서는 ROC-AUC와 F1을, 홀드아웃에서는 Accuracy·Precision·Recall·F1·ROC-AUC를 계산했다. "
               "생존을 양성으로 정의했다. 임계값이 필요한 지표에는 predict의 기본 결정을 사용하고 "
               "ROC-AUC에는 생존 확률을 사용했다.", "note")

        self.heading(4)
        self.p("기본 설정으로 학습한 필수 네 조합의 결과다. CV ROC-AUC는 폴드 평균 ± 표본 표준편차이며, "
               f"오른쪽 다섯 지표는 동일한 홀드아웃 {test_size}명에서 계산했다.")
        result_rows = []
        for row in holdout.itertuples(index=False):
            fold_summary = cv.loc[(row.experiment, row.model)]
            result_rows.append((row.experiment, row.model,
                                f"{fold_summary['mean']:.3f} ± {fold_summary['std']:.3f}",
                                f"{row.accuracy:.3f}", f"{row.precision:.3f}", f"{row.recall:.3f}",
                                f"{row.f1:.3f}", f"{row.auc:.3f}"))
        self.table(("실험", "모델", "CV AUC", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
                   result_rows, [53, 41, 95, 64, 62, 60, 48, 73])
        lr = baseline.loc[("Exp-1", "LR")]
        rf = baseline.loc[("Exp-1", "RF")]
        self.p(f"Exp-1은 기본 설정에서 두 모델의 홀드아웃 ROC-AUC가 가장 높았다. "
               f"Base 대비 증가폭은 LR {lr.auc - baseline.loc[('Base', 'LR')].auc:+.3f}, "
               f"RF {rf.auc - baseline.loc[('Base', 'RF')].auc:+.3f}였다. "
               f"ROC-AUC는 LR({lr.auc:.3f}), Accuracy({rf.accuracy:.3f})와 F1({rf.f1:.3f})은 "
               "RF의 Exp-1 결과가 가장 높았다.")
        majority = int(data["test_classes"]["0"]) / test_size
        self.p(f"홀드아웃에서 전원을 사망으로 예측하는 분류기의 Accuracy는 {majority:.3f}다. "
               "클래스 비율로 인한 정확도 해석의 한계를 보완하기 위해 F1과 ROC-AUC도 함께 읽는다.", "note")
        self.p("GridSearchCV", "sub")
        self.p("모델별로 4개 필수 조합 중 평균 CV ROC-AUC가 가장 높은 조합을 선택한 다음 같은 5개 폴드에서 "
               "하이퍼파라미터를 탐색했다. LR은 C ∈ {0.1, 1, 10}, RF는 max_depth ∈ {4, 8, None}과 "
               "min_samples_leaf ∈ {1, 3, 6}의 조합을 사용했다. 최고 CV 설정을 전체 학습 세트에서 재학습했다.")
        tuned_rows = [
            (r["model"], r["preset"], f"{r['cv_auc']:.3f}", f"{r['accuracy']:.3f}",
             f"{r['precision']:.3f}", f"{r['recall']:.3f}", f"{r['f1']:.3f}", f"{r['auc']:.3f}")
            for r in tuned
        ]
        self.table(("모델", "조합", "CV AUC", "Accuracy", "Precision", "Recall", "F1", "ROC-AUC"),
                   tuned_rows, [53, 60, 76, 64, 62, 60, 48, 73])
        for row in tuned:
            parameters = ", ".join(f"{key.split('__')[-1]}={value}" for key, value in row["params"].items())
            self.p(f"{row['model']} 선택 설정 · {escape(parameters)}", "note")

        self.heading(5)
        self.p("동일한 다섯 검증 폴드에서 한 요소씩 바꿔 비교했다. 표의 ΔAUC와 ΔF1은 변경 후 점수에서 "
               "변경 전 점수를 뺀 폴드별 차이의 평균이다. 양수는 변경 후 개선을 뜻한다.")
        pair_names = [
            ("Exp-1", "Median", "Mean → Median"),
            ("Exp-1", "Mode", "Mean → Most Frequent"),
            ("Exp-1", "Label", "One-Hot → Label"),
            ("Exp-1", "MinMax", "Standard → MinMax"),
            ("Exp-1", "Robust", "Standard → Robust"),
            ("Exp-1", "No-Scaling", "Standard → 미적용"),
            ("Exp-2-All", "Exp-2", "Exp-2의 선택 미적용 → 상위 6"),
            ("Exp-3-All", "Exp-3", "Exp-3의 선택 미적용 → 상위 6"),
            ("No-Derived", "Exp-1", "Exp-1의 파생 변수 미포함 → 포함"),
        ]
        pair_rows = []
        for a, b, title in pair_names:
            l, r = pair(a, b, "LR"), pair(a, b, "RF")
            pair_rows.append((title, f"{l.delta_auc:+.4f}", f"{l.delta_f1:+.4f}",
                              f"{r.delta_auc:+.4f}", f"{r.delta_f1:+.4f}"))
        self.table(("한 요소 변경", "LR ΔAUC", "LR ΔF1", "RF ΔAUC", "RF ΔF1"),
                   pair_rows, [204, 73, 73, 73, 73])
        median = pair("Exp-1", "Median", "RF")
        label_lr, label_rf = pair("Exp-1", "Label", "LR"), pair("Exp-1", "Label", "RF")
        lr_max, rf_max = (max(abs(pair("Exp-1", flag, model).delta_auc)
                              for flag in ("MinMax", "Robust", "No-Scaling"))
                          for model in ("LR", "RF"))
        self.p("대치와 인코딩", "sub")
        self.p(f"RF에서 Median은 Mean보다 {int(median.fold_wins)}/5개 폴드에서 AUC가 높았지만 "
               f"평균 F1은 {median.delta_f1:+.4f}였다. Label 전환은 LR AUC {label_lr.delta_auc:+.4f}, "
               f"RF AUC {label_rf.delta_auc:+.4f}로 방향이 달랐다. Label은 Embarked에 숫자 순서를 "
               "부여하므로 LR에서 각 범주의 효과를 표현하는 방식이 One-Hot과 다르다.")
        self.p("스케일링과 파생 변수", "sub")
        d_lr, d_rf = pair("No-Derived", "Exp-1", "LR"), pair("No-Derived", "Exp-1", "RF")
        self.p(f"Standard 대비 다른 스케일링의 AUC 변화는 LR 최대 {lr_max:.4f}, RF 최대 {rf_max:.4f}였다. "
               "스케일은 LR의 규제 강도에 영향을 줄 수 있다. RF의 분할은 주로 값의 순서에 의존해 "
               "이 실험에서 점수 변화가 작았다. "
               f"파생 변수 2개를 함께 추가했을 때 AUC는 LR {d_lr.delta_auc:+.4f}, RF {d_rf.delta_auc:+.4f}, "
               f"F1은 LR {d_lr.delta_f1:+.4f}, RF {d_rf.delta_f1:+.4f} 변했다.")

        self.heading(6)
        selection_rows = []
        for name, ranks in selected.groupby("experiment"):
            chosen = ranks.loc[ranks["selected"], "feature"].tolist()
            selection_rows.append((name, ", ".join(chosen)))
        self.table(("실험", "최종 학습 데이터에서 선택된 열 6개"), selection_rows, [68, 428])
        self.img("selection", "그림 4. 변환 후 열별 ANOVA F 통계량. 청록색 막대는 선택된 6개 열이다.", 210)
        gap_rows = []
        for name in ("Exp-2", "Exp-3"):
            for model in ("LR", "RF"):
                result = pair(f"{name}-All", name, model)
                gap_rows.append((name, model, f"{result.before_gap:.4f} → {result.after_gap:.4f}",
                                 f"{result.delta_auc:+.4f}"))
        self.table(("조합", "모델", "평균 학습–검증 AUC 격차", "검증 ΔAUC"), gap_rows,
                   [70, 66, 231, 129], compact=True)
        self.p("변수 선택 후 LR의 학습–검증 격차는 줄었지만 검증 ROC-AUC도 낮아졌다. "
               "RF에서는 격차가 두 조합에서 모두 커졌다. 따라서 이 상위 6개 규칙으로 "
               "검증 성능과 과적합이 함께 개선됐다는 근거는 부족하다.")
        self.p("Exp-3에서 Sex_female과 Sex_male이 동시에 선택되어 한 범주형 변수에 두 자리가 쓰였다. "
               "단변량 F 점수는 각 열을 따로 평가하므로 열 간 중복 정보와 변수 조합의 효과를 반영하기 어렵다.", "note")

        self.heading(7)
        self.p(f"<b>ROC-AUC 기준으로는 {best['model']} + {best['preset']}({escape(best_params)})</b>가 "
               f"보고한 홀드아웃 결과 중 가장 높았다({best['auc']:.3f}). "
               f"LR 기본 Exp-1({lr.auc:.3f}) 대비 F1은 {lr.f1:.3f}에서 {best['f1']:.3f}으로 낮아졌다. "
               f"기본 설정 RF + Exp-1은 F1 {rf.f1:.3f}로 임계값 기반 성능이 가장 높았다.")
        self.img("roc", "그림 5. GridSearchCV로 선택한 두 모델의 홀드아웃 ROC 곡선.", 173)
        self.img("permutation", "그림 6. 최적화 RF의 원본 변수 순열 중요도. 막대는 25회 평균, 오차막대는 표준편차다.", 152)
        self.p(f"RF에서 Sex를 섞을 때 홀드아웃 AUC 하락이 가장 컸다({importance.at['Sex', 'mean']:.3f}). "
               "원본 열을 섞은 뒤 종속된 파생 변수도 파이프라인에서 다시 계산했다. 중요도는 모델 의존적이다.")
        self.p("Base 대비 향상에는 범주형·나이 입력의 추가와 여러 전처리가 함께 작용했다. "
               "개별 비교에서 파생 변수와 스케일링 효과는 작았고 One-Hot의 성능 우위는 모델에 따라 달랐다. "
               "결과는 한 번의 홀드아웃에 의존하며, 동일 가족·티켓 승객이 분할 양쪽에 있을 수 있다. "
               "작은 점수 차이의 일반화에는 반복 또는 그룹 분할 검증이 필요하다.", "note")
        if repo_url:
            self.p(f'소스코드  <link href={quoteattr(repo_url)} color="#006b70">{escape(repo_url)}</link>', "note")
        else:
            self.p("소스코드 GitHub 주소 미등록", "note")
        self.p(f"재현 환경  Python {data['python']} · scikit-learn {data['sklearn']} · 시드 {data['seed']}", "note")

        output.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output), pagesize=A4, leftMargin=43, rightMargin=43, topMargin=54,
            bottomMargin=53, title="과제 3 | Titanic 특성 공학 파이프라인",
            subject="Titanic 생존 예측의 전처리 조합 및 모델 성능 비교",
        )
        doc.build(self.story, onFirstPage=self.footer, onLaterPages=self.footer)

    @staticmethod
    def footer(canvas, doc):
        width, _ = A4
        canvas.setStrokeColor(GRID)
        canvas.line(49, 44, width - 49, 44)
        canvas.setFillColor(SLATE)
        canvas.setFont("Korean", 8)
        canvas.drawString(49, 29, "Titanic 생존 예측  /  특성 공학 실험")
        canvas.drawRightString(width - 49, 29, f"{doc.page:02d}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--output", type=Path, default=ROOT / "과제3_Titanic_특성공학.pdf")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--repo-url", default="")
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("Provide a Korean TrueType font via --font.")
    Report(args.results, args.font).build(args.output, args.repo_url)
    print(args.output)

