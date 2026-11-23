"""Generate the Korean submission PDF from MovieLens benchmark outputs."""
import argparse
import json
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image as Bitmap
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
)

from benchmark import RESULTS, ROOT

REPO = "https://github.com/Sindongyeob/bigdata_homework/tree/main/assignment-04"
DATASET = "https://grouplens.org/datasets/movielens/1m/"
BLOOM_PAPER = "https://doi.org/10.1145/362686.362692"
CMS_PAPER = "https://doi.org/10.1016/j.jalgor.2003.12.001"

NAVY = colors.HexColor("#172938")
TEAL = colors.HexColor("#056e71")
MUTED = colors.HexColor("#475f6d")
PALE = colors.HexColor("#eef5f6")
LINE = colors.HexColor("#dae5e8")
WIDTH = 496


def kib(size):
    return size / 1024


def pct(value):
    return f"{value:.2%}"


def draw_charts(results, blooms, sketches):
    plt.rcParams.update({
        "font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "axes.axisbelow": True,
    })
    color, accent = "#056e71", "#ca794a"

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), layout="constrained")
    fixed_k = [r for r in blooms if r["k"] == 4]
    fixed_m = [r for r in blooms if r["m"] == 40_000]
    for ax, rows, x, title in [
        (axes[0], fixed_k, "m", "Bit budget sweep · k = 4"),
        (axes[1], fixed_m, "k", "Hash count sweep · m = 40,000"),
    ]:
        x_values = [r[x] for r in rows]
        ax.plot(x_values, [100 * r["fpr"] for r in rows], "o-", color=color, label="Observed")
        ax.plot(x_values, [100 * r["theoretical_fpr"] for r in rows], "s--",
                color=accent, label="Uniform-hash estimate")
        ax.set_title(title, loc="left", weight="bold")
        ax.set(xlabel="Bits allocated" if x == "m" else "Hash words per item", ylabel="FPR (%)")
        ax.set_xticks(x_values, [f"{i:,}" for i in x_values])
        ax.set_ylim(bottom=0)
    axes[1].legend(fontsize=8)
    fig.savefig(results / "bloom_tradeoff.png", dpi=170, facecolor="white")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.1), layout="constrained")
    for ax, rows, x, title in [
        (axes[0], [r for r in sketches if r["depth"] == 4], "width", "Width sweep · depth = 4"),
        (axes[1], [r for r in sketches if r["width"] == 512], "depth", "Depth sweep · width = 512"),
    ]:
        x_values = [r[x] for r in rows]
        ax.plot(x_values, [r["mae"] for r in rows], "o-", color=color)
        ax.set_title(title, loc="left", weight="bold")
        ax.set_xscale("log", base=2)
        ax.set_xticks(x_values, [f"{i:,}" for i in x_values])
        ax.set(xlabel=x.capitalize(), ylabel="MAE (ratings per movie)")
        ax.set_ylim(bottom=0)
    fig.savefig(results / "cms_tradeoff.png", dpi=170, facecolor="white")
    plt.close(fig)


class Report:
    def __init__(self, results, font):
        self.results = results
        self.flow = []
        bold = font.with_name("malgunbd.ttf") if font.name.lower() == "malgun.ttf" else font
        pdfmetrics.registerFont(TTFont("Kr", str(font)))
        pdfmetrics.registerFont(TTFont("KrBold", str(bold if bold.exists() else font)))
        pdfmetrics.registerFontFamily("Kr", normal="Kr", bold="KrBold")
        self.styles = {
            "kicker": self.style("kicker", 10, 16, TEAL, True, after=5),
            "title": self.style("title", 29, 40, NAVY, True, after=14),
            "heading": self.style("heading", 22, 32, NAVY, True, after=11),
            "sub": self.style("sub", 12, 19, TEAL, True, before=8, after=4),
            "body": self.style("body", 10.4, 17.5, NAVY, after=9),
            "note": self.style("note", 9, 14.5, MUTED, after=7),
            "th": self.style("th", 8.7, 13, colors.white, True),
            "td": self.style("td", 8.7, 13, NAVY),
        }

    @staticmethod
    def style(name, size, leading, color, bold=False, before=0, after=0):
        return ParagraphStyle(name, fontName="KrBold" if bold else "Kr", fontSize=size,
                              leading=leading, textColor=color, wordWrap="CJK",
                              spaceBefore=before, spaceAfter=after)

    def p(self, text, style="body"):
        self.flow.append(Paragraph(text, self.styles[style]))

    def page(self, index, title):
        if index > 1:
            self.flow.append(PageBreak())
        self.p(f"과제 4  /  {index:02d}", "kicker")
        self.p(title, "heading" if index > 1 else "title")

    def table(self, header, rows, widths, compact=False):
        data = [[Paragraph(escape(str(x)), self.styles["th"]) for x in header]]
        data += [[Paragraph(escape(str(x)), self.styles["td"]) for x in row] for row in rows]
        pad = 4 if compact else 6
        table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT", spaceAfter=10)
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), TEAL),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, PALE]),
            ("TOPPADDING", (0, 0), (-1, -1), pad),
            ("BOTTOMPADDING", (0, 0), (-1, -1), pad),
            ("LINEBELOW", (0, -1), (-1, -1), 0.6, LINE),
            ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ]))
        self.flow.append(table)

    def image(self, name, caption):
        path = self.results / f"{name}.png"
        with Bitmap.open(path) as source:
            w, h = source.size
        self.flow.append(Image(str(path), width=WIDTH, height=WIDTH * h / w))
        self.p(caption, "note")

    def link(self, url, caption):
        self.p(f'<link href={quoteattr(url)} color="#056e71">{escape(caption)}</link>', "note")

    @staticmethod
    def footer(canvas, doc):
        w, _ = A4
        canvas.setStrokeColor(LINE)
        canvas.line(49, 44, w - 49, 44)
        canvas.setFillColor(MUTED)
        canvas.setFont("Kr", 8)
        canvas.drawString(49, 29, "MovieLens 1M  /  스트리밍 알고리즘 실험")
        canvas.drawRightString(w - 49, 29, f"{doc.page:02d}")

    def build(self, output, repo_url):
        content = json.loads((self.results / "benchmark.json").read_text(encoding="utf-8"))
        env, rows = content["environment"], content["results"]
        n, u = env["records"], env["distinct"]
        blooms = [r for r in rows if r["kind"] == "bloom"]
        sketches = [r for r in rows if r["kind"] == "cms"]
        exact_set, exact_dict = rows[:2]
        bloom = {(r["m"], r["k"]): r for r in blooms}
        cms = {(r["width"], r["depth"]): r for r in sketches}
        draw_charts(self.results, blooms, sketches)

        self.page(1, "스트리밍 알고리즘<br/>정확도·메모리·시간 비교")
        self.p("<b>Bloom Filter와 Count-Min Sketch</b>를 직접 구현해 MovieLens 평점 이벤트를 처리했다. "
               "두 알고리즘에 같은 영화 ID 스트림을 입력하고, 포함 여부와 누적 빈도를 각각의 정확한 기준과 비교했다.")
        self.table(("데이터셋", "평점 이벤트", "관측 영화 ID", "미관측 질의"),
                   [("MovieLens 1M", f"{n:,}", f"{u:,}", f"{env['negative_queries']:,}")],
                   [145, 117, 117, 117])
        self.p("데이터셋과 입력 단위", "sub")
        self.p("MovieLens 1M은 GroupLens가 공개한 익명 영화 평점 데이터다. "
               "ZIP을 디스크에 보관하고 내부의 ratings.dat를 압축 해제 스트림으로 열어 한 줄씩 읽었다. "
               "각 줄의 두 번째 필드 MovieID를 추출해 평점 이벤트 1건으로 집계했다.")
        self.table(("필드", "의미", "실험에서 사용"),
                   [("UserID", "사용자 식별자", "입력에서 제외"),
                    ("MovieID", "영화 식별자", "멤버십과 누적 빈도의 키"),
                    ("Rating", "1~5 정수 평점", "평점값에 관계없이 이벤트당 +1"),
                    ("Timestamp", "Unix 타임스탬프(초)", "파일 순서를 그대로 사용")],
                   [122, 163, 211])
        top_id, top_freq = env["most_common"][0]
        self.p(f"관측 ID 범위는 {env['min_id']:,}~{env['max_id']:,}다. "
               f"최다 평점 영화 ID {top_id:,}에는 {top_freq:,}건이 몰렸고, "
               f"평점이 10건 이하인 영화는 {env['rare_count']:,}개였다. "
               "이 빈도 차이가 Count-Min Sketch의 상대오차에 미치는 영향도 측정했다.")
        self.link(DATASET, "데이터 출처 · GroupLens MovieLens 1M")

        self.page(2, "구현과 측정 방법")
        self.p("Bloom Filter  |  본 적 있는 영화인가", "sub")
        self.p("m비트 bytearray에서 영화 ID마다 k개의 해시 워드를 m으로 나눈 나머지를 구한다. "
               "삽입은 해당 비트를 1로 만들고, 질의는 해당 비트가 모두 1인지 확인한다. "
               "한 비트라도 0이면 미관측으로 판정한다. 모두 1이면 실제 포함 또는 비트 공유로 생긴 거짓 양성이다. "
               "삽입만 지원하므로 이미 추가한 키가 거짓 음성이 되지 않는다.")
        self.p("Count-Min Sketch  |  영화별 평점 이벤트가 몇 건인가", "sub")
        self.p(f"너비 w, 깊이 d의 테이블을 {env['counter_bytes']}바이트 부호 없는 정수 배열로 구현했다. "
               "각 행에서 해시 워드를 w로 나눈 나머지 열을 고르고, 삽입마다 해당 카운터를 1씩 늘린다. "
               "질의는 해당 카운터 d개의 최솟값을 반환한다. 단위 증가와 카운터 범위가 유지되는 한 "
               "각 카운터는 실제 빈도 이상의 값을 담으므로 추정치는 과소 추정되지 않는다.")
        self.p("두 구현은 키별 BLAKE2b 다이제스트를 32비트 워드들로 나눠 사용했다. "
               f"알고리즘별로 다른 도메인 키와 공통 시드 {env['seed']}를 사용했다. "
               "설정을 키워도 기존 해시 워드를 유지해 깊이·해시 수를 통제해서 비교했다.", "note")
        self.p("정확한 기준과 실험 환경", "sub")
        self.p("별도 1회 스캔으로 Counter에 영화별 실제 빈도를 계산했다. 포함 여부의 정확한 기준은 "
               "관측 키 집합으로, 빈도의 기준은 영화별 실제 빈도 사전으로 평가했다. "
               "각 측정 구성은 파일을 처음부터 독립 스캔하며 정답이나 레코드 목록을 참조하지 않는다.")
        self.table(("항목", "설정"),
                   [("실행 환경", f"{env['implementation']} {env['python']} · {env['system']} {env['release']} · {env['arch']}"),
                    ("CPU 식별자", env["cpu"]),
                    ("시간", f"단일 프로세스, 설정별 {env['repeats']}회. 라운드별 실행 순서를 섞고 중앙값과 범위를 기록"),
                    ("측정 구간", "빈 구조 생성, ZIP 읽기·압축 해제, 줄 파싱, 전체 삽입 포함"),
                    ("메모리", "CPython getsizeof로 스캔 후 보유 구조의 크기 합산. 공통 객체는 한 번만 계산")],
                   [106, 390], compact=True)
        self.p("파일 다운로드와 체크섬 확인, 정답 구축, 질의 평가 시간은 제외했다. 메모리는 자료구조 기준이며 "
               "원본 ZIP, 정답, 질의 배열, 해시의 임시 객체 또는 프로세스 최대 RSS를 합산하지 않았다.", "note")

        self.page(3, "정확도  |  Bloom Filter")
        self.p(f"관측 ID {u:,}개 전체로 거짓 음성을 확인했다. "
               f"1~{env['negative_domain_max']:,}에서 관측 ID를 제외한 뒤 고정 시드로 "
               f"{env['negative_queries']:,}개를 비복원 추출해 거짓 양성을 측정했다. "
               "FPR = 거짓 양성 / 미관측 질의 수, FNR = 거짓 음성 / 관측 질의 수로 계산했다.")
        bloom_rows = [
            (f"{r['m']:,}", r["k"], f"{r['fp']:,}", pct(r["fpr"]), pct(r["theoretical_fpr"]),
             pct(r["occupancy"]), r["fn"])
            for r in blooms
        ]
        bloom_rows.append(("정확 set", "–", exact_set["fp"], pct(exact_set["fpr"]), "–", "–", exact_set["fn"]))
        self.table(("m (비트)", "k", "거짓 양성", "실측 FPR", "근사 FPR", "비트 점유", "거짓 음성"),
                   bloom_rows, [83, 42, 77, 78, 79, 68, 69], compact=True)
        self.image("bloom_tradeoff", "그림 1. k=4에서 비트 예산을 변경한 경우와 m=40,000에서 해시 수를 변경한 경우.")
        small, large = bloom[10000, 4], bloom[80000, 4]
        middle, over = bloom[40000, 8], bloom[40000, 16]
        self.p(f"k=4에서 m을 10,000에서 80,000으로 늘리면 FPR은 "
               f"<b>{pct(small['fpr'])} → {pct(large['fpr'])}</b>로 낮아졌다. "
               f"m=40,000에서는 k=8의 FPR이 {pct(middle['fpr'])}였고 "
               f"k=16으로 늘리면 {pct(over['fpr'])}로 다시 높아졌다. "
               f"k=16에서 비트 점유율이 {pct(over['occupancy'])}까지 높아져 충돌로 모두 1이 될 가능성이 커졌다.")
        self.p("근사 FPR은 균등한 해시를 가정해 (1 − exp(−k × 관측 고유 키 수 / m))의 k제곱으로 계산했다. "
               "평점 행 수가 아닌 고유 영화 수를 사용했다. FNR은 모든 설정에서 0이었다.", "note")

        self.page(4, "정확도  |  Count-Min Sketch")
        self.p("관측 ID 전체에서 MAE(영화별 절대오차 평균), 평균 상대오차(|추정−실제|/실제) 및 "
               "정확 일치율을 계산했다. Top 20은 관측 ID라는 동일한 후보 목록에서 실제 빈도와 "
               "추정 빈도 순위 상위 20개의 교집합 크기다. 동률은 영화 ID 오름차순으로 처리했다.")
        cms_rows = [(f"{r['width']:,}", r["depth"], f"{r['mae']:,.2f}", pct(r["mre"]),
                     pct(r["exact_rate"]), f"{r['max_overcount']:,}", f"{r['top20_hits']}/20")
                    for r in sketches]
        cms_rows.append(("정확 사전", "–", "0.00", "0.00%", "100.00%", "0", "20/20"))
        self.table(("너비 w", "깊이 d", "MAE (건)", "평균 상대오차", "일치율", "최대 과대", "Top 20"),
                   cms_rows, [76, 45, 69, 91, 73, 76, 66], compact=True)
        self.image("cms_tradeoff", "그림 2. 한 파라미터를 고정한 검증. 충돌을 줄이면 영화별 평균 절대오차가 낮아진다.")
        ref, widest = cms[512, 4], cms[4096, 4]
        self.p(f"깊이 4를 고정하고 너비를 512에서 4,096으로 늘리자 MAE가 "
               f"{ref['mae']:,.2f}건에서 {widest['mae']:,.2f}건으로 감소했다. "
               f"w=512, d=4에서 10건 이하 영화 {env['rare_count']:,}개의 평균 상대오차는 "
               f"{pct(ref['rare_mre'])}였고, 1,000건 이상 영화 {env['popular_count']:,}개에서는 "
               f"{pct(ref['popular_mre'])}였다. 상위 빈도 순위가 유지돼도 희소 영화의 추정 오차는 클 수 있다.")
        self.p(f"미관측 ID에는 실제 빈도 0으로 상대오차를 정의할 수 없다. w=512, d=4에서 "
               f"미관측 질의 중 {pct(ref['absent_nonzero'])}가 양수로 추정됐다. "
               "모든 설정에서 관측 ID의 과소 추정은 0건이었다. 순위 평가는 후보 ID가 주어진 경우에 해당한다.", "note")

        self.page(5, "메모리와 전체 처리 시간")
        self.p("배열 KiB는 비트·카운터 저장 공간, 보유 KiB는 파이썬 객체를 포함한 구조 크기다. "
               f"처리량은 {n:,}행을 중앙 처리 시간으로 나눈 값이다. 괄호는 반복 측정 최소~최대 시간이다.")
        time_rows = []
        for row in rows:
            kind = row["kind"]
            if kind == "bloom":
                label = f"Bloom {row['m']:,} / {row['k']}"
            elif kind == "cms":
                label = f"CMS {row['width']:,} / {row['depth']}"
            else:
                label = "정확 set" if kind == "set" else "정확 Counter"
            payload = row["payload_bytes"]
            time_rows.append((label, f"{kib(payload):.2f}" if payload is not None else "–",
                              f"{kib(row['retained_bytes']):.2f}", f"{row['seconds']:.3f}",
                              f"{row['min_seconds']:.3f}~{row['max_seconds']:.3f}",
                              f"{row['records_per_second']:,.0f}"))
        self.table(("구조 / 설정", "배열 KiB", "보유 KiB", "중앙값 (초)", "최소~최대 (초)", "행/초"),
                   time_rows, [125, 69, 68, 75, 82, 77], compact=True)
        self.p(f"Bloom(40,000, 8)은 set의 {kib(exact_set['retained_bytes']):.2f} KiB 중 "
               f"{middle['retained_bytes'] / exact_set['retained_bytes']:.1%} 크기였다. "
               f"CMS(4,096, 4)는 정확 Counter의 {kib(exact_dict['retained_bytes']):.2f} KiB 대비 "
               f"{widest['retained_bytes'] / exact_dict['retained_bytes']:.1%}를 사용했다.")
        equal_narrow, equal_wide = cms[128, 4], cms[512, 1]
        self.p(f"카운터가 같은 512개인 CMS(128, 4)와 CMS(512, 1)의 MAE는 "
               f"각각 {equal_narrow['mae']:,.2f}건과 {equal_wide['mae']:,.2f}건이었다. "
               "이 예산에서는 깊이를 늘리는 것보다 한 행의 너비를 확보하는 편이 유리했다.")
        self.p("두 정확한 자료구조가 이 실험에서 각 근사 구조보다 빨랐다. CPython의 내장 set·Counter는 "
               "정수 키용 최적화 경로를 사용하고, 직접 구현한 두 스케치는 매 이벤트에 해시 계산과 "
               "파이썬 반복 갱신이 필요했다. 이 시간 비교는 해당 언어·구현과 입력 포맷에 한정된다.", "note")

        self.page(6, "결론  |  실험 질문에 대한 답")
        self.p("정확도와 메모리 사이에는 어떤 트레이드오프가 있었는가?", "sub")
        self.p(f"<b>메모리를 줄이면 충돌 오차를 감수해야 했다.</b> Bloom(80,000, 4)은 "
               f"{kib(large['retained_bytes']):.2f} KiB에서 FPR {pct(large['fpr'])}를 기록했다. "
               f"CMS(4,096, 4)는 {kib(widest['retained_bytes']):.2f} KiB로 MAE "
               f"{widest['mae']:.2f}건과 정확 일치율 {pct(widest['exact_rate'])}를 기록했다.")
        self.p("파라미터 증가가 항상 성능 향상으로 이어졌는가?", "sub")
        shallow, deep = cms[512, 1], cms[512, 8]
        self.p(f"<b>Bloom에서는 해시 수의 적정점이 있었고 CMS에서는 정확도와 시간이 맞바뀌었다.</b> "
               f"40,000비트에서 k를 8에서 16으로 늘리자 FPR과 처리 시간이 모두 늘었다. "
               f"CMS의 너비를 512로 고정하고 깊이를 1에서 8로 늘리자 MAE는 "
               f"{shallow['mae']:,.2f} → {deep['mae']:,.2f}건, 전체 시간은 "
               f"{shallow['seconds']:.3f} → {deep['seconds']:.3f}초였다.")
        self.p("어떤 알고리즘이 가장 실용적인가?", "sub")
        self.p("<b>이 데이터에서 정확한 영화별 집계가 목적이라면 Counter가 실용적이다.</b> "
               "고유 영화 수가 작아 정확한 사전의 크기가 제한적이고 측정 시간도 짧았다. 두 근사 구조 중 "
               "포함 검사에는 적은 메모리와 빠른 음성 확인이 가능한 Bloom, 키별 누적 빈도에는 CMS가 적합했다.")
        self.p("실제 서비스 로그에 적용한다면?", "sub")
        self.p("<b>대규모 키의 이벤트 빈도 모니터링에는 CMS를 선택한다.</b> 상세 집계가 필요한 인기 키는 "
               "별도 후보 구조와 정확 집계로 확인한다. 재처리 여부의 빠른 사전 확인에는 Bloom을 쓰고, "
               "양성은 정확 저장소로 확인한다. 시간 창 단위 운영이라면 구조를 구간별로 교체한다.")
        self.p("측정 범위는 정해진 영화 ID, 해시 시드 하나와 합성 미관측 질의다. "
               "키가 계속 늘어나는 서비스에서는 처리량과 오차를 대상 로그에서 다시 측정해야 한다.", "note")
        self.flow.append(Spacer(1, 4))
        self.link(repo_url, f"소스코드 · {repo_url}")
        self.link(BLOOM_PAPER, "참고 · Bloom (1970), Space/Time Trade-offs in Hash Coding with Allowable Errors")
        self.link(CMS_PAPER, "참고 · Cormode & Muthukrishnan (2005), An Improved Data Stream Summary")
        self.p(f"데이터 ZIP SHA-256  {env['sha256']}", "note")

        output.parent.mkdir(parents=True, exist_ok=True)
        doc = SimpleDocTemplate(
            str(output), pagesize=A4, leftMargin=43, rightMargin=43, topMargin=54, bottomMargin=53,
            title="과제 4 | 스트리밍 알고리즘 비교", subject="MovieLens 1M · Bloom Filter · Count-Min Sketch",
        )
        doc.build(self.flow, onFirstPage=self.footer, onLaterPages=self.footer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=RESULTS)
    parser.add_argument("--output", type=Path, default=ROOT / "과제4_스트리밍_알고리즘_비교.pdf")
    parser.add_argument("--font", type=Path, default=Path("C:/Windows/Fonts/malgun.ttf"))
    parser.add_argument("--repo-url", default=REPO)
    args = parser.parse_args()
    if not args.font.is_file():
        parser.error("Provide a Korean TrueType font via --font.")
    Report(args.results, args.font).build(args.output, args.repo_url)
    print(args.output)
