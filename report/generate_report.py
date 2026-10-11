"""LEGACY v1.0 report generator.

This script is retained only to reproduce the historical English PDF. It is
not part of the v1.1 delivery flow and must not be presented as a v1.1 paper.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER, TA_JUSTIFY
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import (
    BaseDocTemplate,
    Frame,
    Image,
    KeepTogether,
    PageBreak,
    PageTemplate,
    Paragraph,
    Preformatted,
    Spacer,
    Table,
    TableStyle,
)


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "output/pdf/A1_NumPy_MLP_Report.pdf"


def load_json(relative: str):
    return json.loads((ROOT / relative).read_text())


def load_csv(relative: str) -> list[dict[str, str]]:
    with (ROOT / relative).open(newline="") as handle:
        return list(csv.DictReader(handle))


RUNS = load_csv("results/aggregate/all_runs.csv")
RUN = {row["run_id"]: row for row in RUNS}
TESTS = {row["run_id"]: row for row in load_json("results/aggregate/final_test_metrics.json")}
GRADIENT = load_json("results/verification/gradient_check.json")
TINY = load_json("results/verification/tiny_overfit.json")
MNIST_BENCH = load_json("results/benchmarks/mnist_100_steps.json")
CIFAR_BENCH = load_json("results/benchmarks/cifar10_100_steps.json")
DATA_CHECK = load_json("results/verification/data_validation.json")

assert len(RUNS) == 16 and all(row["status"] == "completed" for row in RUNS)
assert len(TESTS) == 10 and GRADIENT["status"] == "passed"
assert DATA_CHECK["mnist"]["status"] == DATA_CHECK["cifar10"]["status"] == "passed"


PAGE_W, PAGE_H = A4
MARGIN_X = 1.75 * cm
MARGIN_TOP = 1.8 * cm
MARGIN_BOTTOM = 1.65 * cm


class ReportDocTemplate(BaseDocTemplate):
    def __init__(self, filename: str) -> None:
        super().__init__(
            filename,
            pagesize=A4,
            leftMargin=MARGIN_X,
            rightMargin=MARGIN_X,
            topMargin=MARGIN_TOP,
            bottomMargin=MARGIN_BOTTOM,
            title="A Reproducible NumPy-Only Multilayer Perceptron",
            author="A1 Course Project",
            subject="MNIST and CIFAR-10 image classification experiments",
        )
        frame = Frame(
            self.leftMargin,
            self.bottomMargin,
            self.width,
            self.height,
            id="normal",
        )
        self.addPageTemplates(PageTemplate(id="paper", frames=frame, onPage=self.header_footer))

    @staticmethod
    def header_footer(canvas, doc) -> None:
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#D9DEE7"))
        canvas.setLineWidth(0.4)
        if doc.page > 1:
            canvas.line(MARGIN_X, PAGE_H - 1.18 * cm, PAGE_W - MARGIN_X, PAGE_H - 1.18 * cm)
            canvas.setFont("Helvetica", 7.5)
            canvas.setFillColor(colors.HexColor("#5B6472"))
            canvas.drawString(MARGIN_X, PAGE_H - 0.94 * cm, "A1 NumPy MLP Image Classification")
        canvas.line(MARGIN_X, 1.08 * cm, PAGE_W - MARGIN_X, 1.08 * cm)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor("#5B6472"))
        canvas.drawString(MARGIN_X, 0.72 * cm, "Reproducible experimental report")
        canvas.drawRightString(PAGE_W - MARGIN_X, 0.72 * cm, f"Page {doc.page}")
        canvas.restoreState()


styles = getSampleStyleSheet()
styles.add(
    ParagraphStyle(
        "PaperTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=23,
        leading=28,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#17233C"),
        spaceAfter=15,
    )
)
styles.add(
    ParagraphStyle(
        "Subtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=11,
        leading=15,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5565"),
    )
)
styles.add(
    ParagraphStyle(
        "Abstract",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.2,
        leading=13.2,
        alignment=TA_JUSTIFY,
        leftIndent=0.55 * cm,
        rightIndent=0.55 * cm,
        borderColor=colors.HexColor("#CBD5E1"),
        borderWidth=0.6,
        borderPadding=10,
        backColor=colors.HexColor("#F7F9FC"),
    )
)
styles.add(
    ParagraphStyle(
        "Section",
        parent=styles["Heading1"],
        fontName="Helvetica-Bold",
        fontSize=15,
        leading=18,
        textColor=colors.HexColor("#173B67"),
        spaceBefore=7,
        spaceAfter=7,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "Subsection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=11.5,
        leading=14,
        textColor=colors.HexColor("#274C77"),
        spaceBefore=6,
        spaceAfter=4,
        keepWithNext=True,
    )
)
styles.add(
    ParagraphStyle(
        "BodyPaper",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=9.15,
        leading=13.1,
        alignment=TA_JUSTIFY,
        spaceAfter=5,
    )
)
styles.add(
    ParagraphStyle(
        "Caption",
        parent=styles["BodyText"],
        fontName="Helvetica-Oblique",
        fontSize=7.8,
        leading=10,
        alignment=TA_CENTER,
        textColor=colors.HexColor("#4B5565"),
        spaceBefore=3,
        spaceAfter=7,
    )
)
styles.add(
    ParagraphStyle(
        "CodeBox",
        parent=styles["Code"],
        fontName="Courier",
        fontSize=7.6,
        leading=10,
        leftIndent=8,
        rightIndent=8,
        borderColor=colors.HexColor("#D5DCE8"),
        borderWidth=0.5,
        borderPadding=7,
        backColor=colors.HexColor("#F7F9FC"),
        spaceAfter=7,
    )
)


def p(text: str, style: str = "BodyPaper") -> Paragraph:
    return Paragraph(text, styles[style])


def section(number: str, title: str) -> Paragraph:
    return p(f"{number} {title}", "Section")


def subsection(number: str, title: str) -> Paragraph:
    return p(f"{number} {title}", "Subsection")


def table(data: list[list], widths: list[float], alignments: list[str] | None = None, font_size: float = 7.6) -> Table:
    result = Table(data, colWidths=widths, repeatRows=1, hAlign="CENTER")
    commands = [
        ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#284B73")),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("FONTNAME", (0, 1), (-1, -1), "Helvetica"),
        ("FONTSIZE", (0, 0), (-1, -1), font_size),
        ("LEADING", (0, 0), (-1, -1), font_size + 2.2),
        ("GRID", (0, 0), (-1, -1), 0.35, colors.HexColor("#C7CED9")),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, colors.HexColor("#F3F6FA")]),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]
    if alignments:
        for index, alignment in enumerate(alignments):
            commands.append(("ALIGN", (index, 1), (index, -1), alignment))
    result.setStyle(TableStyle(commands))
    return result


def figure(relative: str, caption: str, max_height: float = 10.4 * cm) -> list:
    path = ROOT / relative
    image = Image(str(path))
    scale = min((PAGE_W - 2 * MARGIN_X) / image.imageWidth, max_height / image.imageHeight)
    image.drawWidth = image.imageWidth * scale
    image.drawHeight = image.imageHeight * scale
    return [image, p(caption, "Caption")]


def pct(value) -> str:
    return f"{100 * float(value):.2f}%"


def f4(value) -> str:
    return f"{float(value):.4f}"


story: list = []
story += [Spacer(1, 1.4 * cm)]
story += [p("A Reproducible NumPy-Only Multilayer Perceptron", "PaperTitle")]
story += [p("From first principles to MNIST and CIFAR-10 controlled experiments", "Subtitle")]
story += [Spacer(1, 0.55 * cm)]
story += [p("A1 Course Project Experimental Report", "Subtitle")]
story += [p("11 October 2026", "Subtitle")]
story += [Spacer(1, 1.1 * cm)]
abstract = (
    "<b>Abstract.</b> This work implements a reusable multilayer perceptron entirely with NumPy and evaluates it on "
    "MNIST and CIFAR-10 under a strict train-validation-test protocol. Forward propagation, backpropagation, stable "
    "losses, regularization, optimizers, checkpointing and metrics were implemented without a deep-learning framework. "
    f"The baseline reached {pct(TESTS['B-M']['test_accuracy'])} MNIST test accuracy and "
    f"{pct(TESTS['B-C']['test_accuracy'])} CIFAR-10 test accuracy. A validation-selected Momentum configuration reached "
    f"{pct(TESTS['O-MOM-0.05']['test_accuracy'])} on MNIST. Sixteen formal runs cover cross-entropy versus Softmax-MSE, "
    "L1/L2/Dropout, four optimizer families, and three batch regimes. All reported numbers are read directly from saved "
    "run artifacts; no test result was used for model selection."
)
story += [p(abstract, "Abstract")]
story += [Spacer(1, 0.45 * cm)]
story += [p("<b>Keywords:</b> multilayer perceptron; NumPy; backpropagation; MNIST; CIFAR-10; reproducibility", "BodyPaper")]
story += [PageBreak()]

story += [section("1", "Introduction")]
story += [p(
    "The objective is a correct and auditable educational implementation rather than a state-of-the-art vision model. "
    "The same dense-network core is used for both datasets, allowing the experiments to isolate losses, regularizers, "
    "optimizers and batch regimes. The design enforces row-major batches X in R^(B x D), explicitly averaged data loss, "
    "and a single division by B in the loss gradient. This convention prevents a common backpropagation error in which "
    "the batch factor is applied twice."
)]
story += [p(
    "The study answers three questions. First, can a compact NumPy implementation pass numerical correctness gates and "
    "train both datasets end to end? Second, how do cross-entropy, probability-space MSE and three regularization choices "
    "behave under a controlled MNIST baseline? Third, how do SGD, Momentum, AdaGrad and Adam compare when each algorithm "
    "receives the same number of predeclared learning-rate candidates?"
)]
story += [section("2", "Methods")]
story += [subsection("2.1", "Network and backpropagation")]
story += [p(
    "The MNIST network is 784-128-64-10 and the CIFAR-10 network is 3072-256-128-10. Hidden layers use ReLU and "
    "Kaiming-normal initialization; the linear output layer uses Xavier-normal initialization. Biases start at zero. "
    "All training tensors and parameters are float32, while finite-difference checks use float64."
)]
story += [Preformatted(
    "Z = A @ W + b\nH = max(Z, 0)\ndW = A.T @ dZ\ndb = sum(dZ, axis=0)\ndA = dZ @ W.T",
    styles["CodeBox"],
)]
story += [subsection("2.2", "Stable losses")]
story += [p(
    "Cross-entropy is evaluated through row-wise log-sum-exp. Let m_i = max_j z_ij. The per-sample term is "
    "m_i + log(sum_j exp(z_ij - m_i)) - z_i,y_i, and the logits gradient is (P - one_hot(y))/B. "
    "This remains finite for logits near +/-1000."
)]
story += [p(
    "The alternative objective is squared error on Softmax probabilities: L = (1/(2B)) sum ||P - Y||^2. Its full "
    "Softmax Jacobian-vector product is used; the cross-entropy gradient is not reused. Therefore CE and MSE objective "
    "values are not compared directly. Common validation accuracy and validation CE provide the comparison metrics."
)]
story += [subsection("2.3", "Regularization and optimization")]
story += [p(
    "L1 and L2 penalties apply only to weight matrices. L1 uses lambda sum |W| and subgradient lambda sign(W); L2 uses "
    "(lambda/2)||W||^2 and gradient lambda W. Inverted Dropout with p=0.2 follows each hidden activation during training "
    "and becomes an identity during evaluation without advancing the training random generator."
)]
story += [p(
    "The optimizers are mini-batch SGD, classical Momentum, AdaGrad and Adam with bias correction. Each parameter has "
    "state keyed by a stable name. The candidate lists were frozen before formal optimizer runs: SGD {0.05, 0.1}, "
    "Momentum {0.01, 0.05}, AdaGrad {0.01, 0.05}, and Adam {0.001, 0.003}."
)]

story += [section("3", "Implementation and Correctness")]
story += [subsection("3.1", "Software architecture")]
story += [p(
    "The package separates layers, model assembly, losses, regularization, optimizers, metrics, data, trainer, "
    "checkpoints and reporting. Training follows zero_grad -> forward -> loss/gradient -> backward -> regularization "
    "gradient -> optimizer step. Checkpoints contain only NumPy arrays plus UTF-8 JSON metadata and are loaded with "
    "allow_pickle=False. Each run stores its immutable config, environment, data manifest, CSV history, best checkpoint, "
    "summary, console log and figures."
)]
max_mlp_abs = max(item["max_abs_error"] for item in GRADIENT["full_mlp"].values())
correctness_rows = [
    ["Check", "Observed result", "Gate"],
    ["pytest", "21 passed", "all pass"],
    ["CE finite difference", f"max abs {GRADIENT['cross_entropy']['max_abs_error']:.2e}", "abs <= 1e-7 or rel <= 1e-4"],
    ["Softmax-MSE finite difference", f"max abs {GRADIENT['softmax_mse']['max_abs_error']:.2e}", "same"],
    ["full MLP finite difference", f"max abs {max_mlp_abs:.2e}", "same"],
    ["tiny-set overfit", f"loss {TINY['initial_loss']:.4f} -> {TINY['final_loss']:.6f}; acc {pct(TINY['accuracy'])}", "100% train acc"],
    ["checkpoint round trip", "identical validation metrics", "match"],
]
story += [subsection("3.2", "Verification evidence")]
story += [table(correctness_rows, [4.0 * cm, 7.0 * cm, 5.4 * cm], ["LEFT", "LEFT", "LEFT"])]
story += [p(
    "The largest sampled full-network absolute gradient discrepancy was "
    f"{max_mlp_abs:.3e}. The real-data verification also checks official checksums, exact counts, disjoint fixed splits, "
    "label ranges and finite float32 preprocessing outputs."
)]

story += [PageBreak(), section("4", "Data and Experimental Protocol")]
data_rows = [
    ["Dataset", "Official train", "Train", "Validation", "Test", "Input", "Preprocess"],
    ["MNIST", "60,000", "55,000", "5,000", "10,000", "784", "divide by 255"],
    ["CIFAR-10", "50,000", "45,000", "5,000", "10,000", "3072", "train-only RGB standardization"],
]
story += [table(data_rows, [2.1 * cm, 2.3 * cm, 2.0 * cm, 2.1 * cm, 1.9 * cm, 1.6 * cm, 4.4 * cm])]
story += [p(
    "Both official training sets are stratified with split_seed=20261011. The split is persisted and shared by every "
    "run. MNIST is converted per batch from uint8 to float32 and divided by 255. CIFAR-10 channel statistics are fitted "
    "only on its 45,000-sample training subset: mean [125.305, 122.946, 113.880] and standard deviation "
    "[62.977, 62.066, 66.678]. Validation selects checkpoints by accuracy, then CE, then earlier epoch."
)]
story += [p(
    "The test split was not opened during training or hyperparameter selection. A machine-readable selection file was "
    "written before the first test evaluation. Ten frozen representative checkpoints were then evaluated exactly once; "
    "batch-strategy subset models were intentionally kept validation-only."
)]
benchmark_rows = [
    ["Dataset", "Network params", "100-step throughput", "Train-only epoch estimate", "Formal epochs"],
    ["MNIST", f"{MNIST_BENCH['parameter_count']:,}", f"{MNIST_BENCH['samples_per_second']:,.0f} samples/s", f"{MNIST_BENCH['estimated_full_epoch_seconds_train_only']:.2f} s", "12"],
    ["CIFAR-10", f"{CIFAR_BENCH['parameter_count']:,}", f"{CIFAR_BENCH['samples_per_second']:,.0f} samples/s", f"{CIFAR_BENCH['estimated_full_epoch_seconds_train_only']:.2f} s", "15"],
]
story += [subsection("4.1", "Compute budget")]
story += [table(benchmark_rows, [2.2 * cm, 3.0 * cm, 4.1 * cm, 4.1 * cm, 2.1 * cm])]
story += [p(
    "The host was macOS 15.7.3 x86_64 with 8 logical CPUs, uv-managed CPython 3.12.13 and NumPy 2.5.4. Benchmarks "
    "showed that all P0/P1 runs were far below the 45-minute governance limit, so no full-dataset core experiment was "
    "reduced. Only the controlled batch study used 5,000 training samples."
)]

story += [section("5", "Baseline Results")]
baseline_rows = [
    ["Run", "Dataset", "Best epoch", "Val accuracy", "Val CE", "Test accuracy", "Test CE", "Wall time"],
    ["B-M", "MNIST", RUN["B-M"]["best_epoch"], pct(RUN["B-M"]["best_val_accuracy"]), f4(RUN["B-M"]["best_val_ce"]), pct(TESTS["B-M"]["test_accuracy"]), f4(TESTS["B-M"]["test_ce"]), f"{float(RUN['B-M']['wall_seconds']):.1f} s"],
    ["B-C", "CIFAR-10", RUN["B-C"]["best_epoch"], pct(RUN["B-C"]["best_val_accuracy"]), f4(RUN["B-C"]["best_val_ce"]), pct(TESTS["B-C"]["test_accuracy"]), f4(TESTS["B-C"]["test_ce"]), f"{float(RUN['B-C']['wall_seconds']):.1f} s"],
]
story += [table(baseline_rows, [1.4 * cm, 2.0 * cm, 1.8 * cm, 2.3 * cm, 1.8 * cm, 2.3 * cm, 1.8 * cm, 2.0 * cm])]
story += [p(
    f"The MNIST baseline generalized closely from {pct(RUN['B-M']['best_val_accuracy'])} validation to "
    f"{pct(TESTS['B-M']['test_accuracy'])} test accuracy. The CIFAR-10 baseline reached "
    f"{pct(TESTS['B-C']['test_accuracy'])}; because the input is a flattened image, it has no convolutional inductive "
    "bias and is not presented as competitive with CNNs."
)]
story += figure("results/formal/B-M/figures/learning_curves.png", "Figure 1. MNIST baseline learning curves. The best validation accuracy occurs at epoch 10.")
story += [PageBreak()]
story += figure("results/formal/B-C/figures/learning_curves.png", "Figure 2. CIFAR-10 baseline learning curves. The widening train-validation gap reveals overfitting.")

story += [section("6", "Loss and Regularization Study")]
loss_reg_ids = ["B-M", "L-MSE", "R-L1", "R-L2", "R-DROP"]
loss_rows = [["Run", "Change from baseline", "Best epoch", "Val acc", "Test acc", "Test CE", "Wall time"]]
descriptions = {
    "B-M": "CE, no regularizer",
    "L-MSE": "Softmax probability MSE",
    "R-L1": "L1 lambda=1e-5",
    "R-L2": "L2 lambda=1e-4",
    "R-DROP": "Dropout p=0.2",
}
for run_id in loss_reg_ids:
    loss_rows.append([
        run_id,
        descriptions[run_id],
        RUN[run_id]["best_epoch"],
        pct(RUN[run_id]["best_val_accuracy"]),
        pct(TESTS[run_id]["test_accuracy"]),
        f4(TESTS[run_id]["test_ce"]),
        f"{float(RUN[run_id]['wall_seconds']):.1f} s",
    ])
story += [table(loss_rows, [1.5 * cm, 4.2 * cm, 1.7 * cm, 2.0 * cm, 2.0 * cm, 1.8 * cm, 2.0 * cm])]
story += [p(
    "Softmax-MSE was the clearest negative result: with the same architecture, seed, batch order, learning rate and epoch "
    f"budget, it reached {pct(RUN['L-MSE']['best_val_accuracy'])} versus {pct(RUN['B-M']['best_val_accuracy'])} for CE. "
    "The MSE gradient shrinks through the Softmax Jacobian, slowing optimization. The three tested regularizers were close "
    "to the baseline. L1 produced the strongest frozen test result among them, but the study uses one seed and does not "
    "claim statistical significance."
)]
story += figure("results/aggregate/loss_regularization_accuracy.png", "Figure 3. Best validation accuracy for the controlled loss and regularization runs.", 8.5 * cm)

story += [PageBreak(), section("7", "Optimizer Study")]
optimizer_ids = ["B-M", "O-SGD-0.1", "O-MOM-0.01", "O-MOM-0.05", "O-ADA-0.01", "O-ADA-0.05", "O-ADAM-0.001", "O-ADAM-0.003"]
optimizer_rows = [["Run", "Algorithm", "LR", "Best epoch", "Val acc", "Val CE", "Wall time", "Selected"]]
selected = {"O-SGD-0.1", "O-MOM-0.05", "O-ADA-0.01", "O-ADAM-0.003"}
for run_id in optimizer_ids:
    optimizer_rows.append([
        run_id,
        RUN[run_id]["optimizer"],
        RUN[run_id]["learning_rate"],
        RUN[run_id]["best_epoch"],
        pct(RUN[run_id]["best_val_accuracy"]),
        f4(RUN[run_id]["best_val_ce"]),
        f"{float(RUN[run_id]['wall_seconds']):.1f} s",
        "yes" if run_id in selected else "no",
    ])
story += [table(optimizer_rows, [2.5 * cm, 2.1 * cm, 1.4 * cm, 1.5 * cm, 1.8 * cm, 1.7 * cm, 1.8 * cm, 1.6 * cm], font_size=7.1)]
story += [p(
    f"Momentum at learning rate 0.05 achieved the best validation accuracy ({pct(RUN['O-MOM-0.05']['best_val_accuracy'])}) "
    f"and {pct(TESTS['O-MOM-0.05']['test_accuracy'])} test accuracy. Adam at 0.003 reached its best validation result "
    "at epoch 7, earlier than the selected SGD and Momentum models, but validation CE later deteriorated. AdaGrad's two "
    "candidates tied on accuracy; 0.01 was selected because its validation CE was lower."
)]
story += figure("results/aggregate/optimizer_candidates_accuracy.png", "Figure 4. Best validation accuracy for all predeclared optimizer candidates.", 8.3 * cm)
story += figure("results/aggregate/optimizer_by_steps.png", "Figure 5. Optimizer validation accuracy indexed by cumulative update count.", 8.4 * cm)

story += [PageBreak(), section("8", "Batch-Strategy Study")]
batch_ids = ["D-ONLINE", "D-MINI", "D-BATCH"]
batch_rows = [["Run", "Batch size", "Samples seen", "Updates", "Best val acc", "Wall time", "Interpretation"]]
interpretation = {
    "D-ONLINE": "many noisy updates",
    "D-MINI": "balanced updates",
    "D-BATCH": "only 10 updates",
}
for run_id in batch_ids:
    batch_rows.append([
        run_id,
        RUN[run_id]["batch_size"],
        RUN[run_id]["samples_seen"],
        RUN[run_id]["optimizer_steps"],
        pct(RUN[run_id]["best_val_accuracy"]),
        f"{float(RUN[run_id]['wall_seconds']):.1f} s",
        interpretation[run_id],
    ])
story += [table(batch_rows, [2.1 * cm, 1.7 * cm, 2.2 * cm, 1.8 * cm, 2.3 * cm, 1.8 * cm, 4.0 * cm])]
story += [p(
    "All three runs traverse the same fixed 5,000-sample subset for 10 epochs and therefore see 50,000 examples, but "
    "they make 50,000, 400 and 10 updates respectively. The comparison is intentionally not described as equal compute. "
    "At the shared learning rate of 0.05, online updates were highly variable but ultimately strongest, while full-batch "
    "training remained under-optimized after only ten parameter updates. This illustrates why sample exposure and update "
    "count must both accompany batch-size claims."
)]
story += figure("results/aggregate/batch_strategies_accuracy.png", "Figure 6. Controlled batch-strategy validation results on the fixed 5,000-sample subset.", 8.6 * cm)

story += [section("9", "Frozen Test Evaluation")]
test_order = ["B-M", "L-MSE", "R-L1", "R-L2", "R-DROP", "O-SGD-0.1", "O-MOM-0.05", "O-ADA-0.01", "O-ADAM-0.003", "B-C"]
test_rows = [["Run", "Dataset", "Val accuracy", "Test accuracy", "Test CE"]]
for run_id in test_order:
    test_rows.append([
        run_id,
        TESTS[run_id]["dataset"].upper(),
        pct(TESTS[run_id]["validation_accuracy"]),
        pct(TESTS[run_id]["test_accuracy"]),
        f4(TESTS[run_id]["test_ce"]),
    ])
story += [table(test_rows, [3.2 * cm, 2.8 * cm, 3.3 * cm, 3.3 * cm, 2.5 * cm])]
story += [p(
    "Validation and test values are close for the MNIST models, supporting the split and checkpoint protocol. The table "
    "is descriptive rather than a new selection round. In particular, the regularizer with the highest test accuracy was "
    "not used to launch another training run."
)]
story += figure("results/formal/B-M/test_evaluation_confusion.png", "Figure 7. MNIST baseline test confusion matrix.", 8.6 * cm)
story += [PageBreak()]
story += figure("results/formal/B-C/test_evaluation_confusion.png", "Figure 8. CIFAR-10 baseline test confusion matrix.", 10.0 * cm)

story += [section("10", "Discussion and Limitations")]
story += [p(
    "The experiments demonstrate that a compact, framework-free implementation can be both correct and useful for "
    "controlled studies. Stable cross-entropy optimized more effectively than probability-space MSE under matched "
    "conditions. Momentum delivered the strongest MNIST checkpoint in this finite candidate set. The regularization "
    "results were modest because the baseline already generalized well and the study did not perform an expansive "
    "regularization search."
)]
story += [p(
    "There are four important limitations. First, most experiments use one initialization and shuffle seed, so differences "
    "are not estimates of statistical significance. Second, CIFAR-10 is deliberately treated as flattened pixels; spatial "
    "locality is absent and the result should not be compared as if it were a CNN. Third, the controlled batch experiment "
    "uses a 5,000-sample subset and a shared learning rate, making it a study of update regimes rather than a claim about "
    "each regime's optimally tuned accuracy. Fourth, optimizer candidate counts are equal but small."
)]
story += [p(
    "Future work could add three-seed repeats, a carefully budgeted RMSProp comparison, early stopping and a convolutional "
    "model outside the NumPy-MLP assignment scope. None of these optional extensions is needed to establish the two "
    "required additional studies."
)]

story += [section("11", "Reproducibility")]
story += [p(
    "The source ZIP excludes raw data and caches. Reproduction begins by creating the uv environment, preparing checksum-"
    "verified datasets, running tests, and training from the versioned TOML configs. Each run uses explicit split, "
    "initialization, shuffle and Dropout seeds."
)]
story += [Preformatted(
    "UV_CACHE_DIR=/tmp/a1-uv-cache uv sync --group dev\n"
    "MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache uv run pytest -q\n"
    "MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli prepare-data --dataset mnist --root data/raw/mnist --download\n"
    "MPLCONFIGDIR=/tmp/a1-mpl UV_CACHE_DIR=/tmp/a1-uv-cache uv run python -m a1_mlp.cli train --config configs/mnist_baseline.toml",
    styles["CodeBox"],
)]
story += [p(
    "The experiment table can be regenerated without retraining by running experiments/aggregate.py. The report itself "
    "reads the aggregate CSV, test JSON files, benchmark JSON files and verification evidence and fails if core artifacts "
    "are missing or incomplete."
)]

story += [section("12", "AI Assistance and Human Verification")]
story += [p(
    "Codex/ChatGPT assisted with architecture design translation, implementation, test construction, debugging, execution "
    "orchestration, data-source troubleshooting, aggregation, writing and PDF generation. AI involvement was comprehensive "
    "and is disclosed as required by the course materials. No metric or command result was invented: all numerical claims "
    "come from artifacts produced by local execution. Human responsibility remains for reviewing the final report, "
    "confirming the course portal's current naming/size rules, and performing the actual submission."
)]

story += [section("13", "Conclusion")]
story += [p(
    f"The completed NumPy-only framework passed numerical and end-to-end correctness gates, reached "
    f"{pct(TESTS['B-M']['test_accuracy'])} on MNIST and {pct(TESTS['B-C']['test_accuracy'])} on CIFAR-10, and produced "
    "a traceable 16-run experiment set covering both required additional topics. The results support cross-entropy for "
    "this classification setting, show the practical value of Momentum within the tested budget, and expose the limits "
    "of a flattened-pixel MLP on CIFAR-10."
)]

story += [section("References", "")]
references = [
    "[1] D. E. Rumelhart, G. E. Hinton, and R. J. Williams, Learning representations by back-propagating errors, Nature, 1986.",
    "[2] Y. LeCun, C. Cortes, and C. J. C. Burges, The MNIST database of handwritten digits.",
    "[3] A. Krizhevsky, Learning Multiple Layers of Features from Tiny Images, Technical Report, University of Toronto, 2009.",
    "[4] D. P. Kingma and J. Ba, Adam: A Method for Stochastic Optimization, ICLR, 2015.",
    "[5] N. Srivastava et al., Dropout: A Simple Way to Prevent Neural Networks from Overfitting, JMLR, 2014.",
]
for reference in references:
    story += [p(reference)]

story += [PageBreak(), section("Appendix A", "Formal Run Inventory")]
inventory_rows = [["Run", "Dataset", "Objective", "Optimizer", "LR", "Batch", "Best val", "Steps"]]
for row in RUNS:
    inventory_rows.append([
        row["run_id"],
        row["dataset"],
        row["loss"],
        row["optimizer"],
        row["learning_rate"],
        row["batch_size"],
        pct(row["best_val_accuracy"]),
        row["optimizer_steps"],
    ])
story += [table(inventory_rows, [2.6 * cm, 1.8 * cm, 2.7 * cm, 2.1 * cm, 1.5 * cm, 1.4 * cm, 2.1 * cm, 2.0 * cm], font_size=6.7)]
story += [p(
    "All 16 formal runs completed successfully. No failed or unfavorable run was removed from the inventory. Raw epoch "
    "histories, configuration hashes, environment manifests and console logs remain under results/formal/."
)]

OUTPUT.parent.mkdir(parents=True, exist_ok=True)
ReportDocTemplate(str(OUTPUT)).build(story)
print(OUTPUT)

