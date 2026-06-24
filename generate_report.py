"""Generate report.pdf: a usage guide + efficiency comparison of the three
neural-network algorithms (Hebb / Perceptron / Adaline) on the 8x8 digit set.

Run:  python generate_report.py
"""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")                     # headless, file output only
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

import digits
from algorithms import HebbNetwork, Perceptron, AdalineClassifier

ALGOS = ["Hebb", "Perceptron", "Adaline"]
COLORS = {"Hebb": "#C62828", "Perceptron": "#1565C0", "Adaline": "#2E7D32"}
NOISE_LEVELS = [0, 5, 10, 15, 20, 25]      # percent of pixels flipped
VARIATIONS = 30                            # samples per digit at each noise level


# ── train models on the clean digit glyphs, timing the training ─────
def train_models():
    X, labels = digits.base_glyphs()
    n, n_cls = X.shape[1], len(labels)
    models, train_time = {}, {}

    t = time.perf_counter()
    h = HebbNetwork(n); h.train((2 * X - 1).astype(np.float32))
    train_time["Hebb"] = time.perf_counter() - t
    models["Hebb"] = {"net": h, "labels": labels, "bip": (2 * X - 1).astype(np.float32)}

    t = time.perf_counter()
    p = Perceptron(n, n_cls); p.train(X, list(range(n_cls)))
    train_time["Perceptron"] = time.perf_counter() - t
    models["Perceptron"] = {"net": p, "labels": labels}

    t = time.perf_counter()
    a = AdalineClassifier(n, n_cls); a.train(X, list(range(n_cls)))
    train_time["Adaline"] = time.perf_counter() - t
    models["Adaline"] = {"net": a, "labels": labels}

    return models, train_time


def predict(algo, model, vec):
    x = vec.astype(np.float32)
    if algo == "Hebb":
        r = model["net"].recall(2 * x - 1)
        best = int(np.argmax([np.dot(r, p) for p in model["bip"]]))
        return model["labels"][best]
    elif algo == "Perceptron":
        return model["labels"][model["net"].predict(x)]
    else:
        pred, _ = model["net"].predict(x)
        return model["labels"][pred]


def benchmark(models):
    """Return accuracy[algo] -> list over NOISE_LEVELS, and per-sample infer time."""
    accuracy = {a: [] for a in ALGOS}
    infer_us = {a: [] for a in ALGOS}
    for noise in NOISE_LEVELS:
        X, y = digits.generate_dataset(VARIATIONS, noise / 100.0, seed=1)
        for a in ALGOS:
            t = time.perf_counter()
            correct = sum(1 for v, t_lbl in zip(X, y)
                          if predict(a, models[a], v) == t_lbl)
            dt = time.perf_counter() - t
            accuracy[a].append(correct / len(y) * 100)
            infer_us[a].append(dt / len(y) * 1e6)     # microseconds/sample
    return accuracy, infer_us


# ── PDF pages ───────────────────────────────────────────────────────
def text_page(pdf, title, body, mono=False):
    fig = plt.figure(figsize=(8.27, 11.69))    # A4 portrait
    fig.text(0.08, 0.93, title, fontsize=20, fontweight="bold", color="#1e2a3a")
    fig.text(0.08, 0.905, "ANN Pattern Classifier", fontsize=10, color="#888")
    fig.text(0.08, 0.88, "_" * 78, fontsize=10, color="#ccc")
    fig.text(0.08, 0.86, body, fontsize=10.5, va="top", ha="left",
             family="monospace" if mono else "sans-serif",
             linespacing=1.5, wrap=True)
    fig.patch.set_facecolor("white")
    plt.axis("off")
    pdf.savefig(fig); plt.close(fig)


def charts_page(pdf, accuracy, infer_us, train_time):
    fig, axes = plt.subplots(2, 2, figsize=(8.27, 11.69))
    fig.suptitle("Efficiency Comparison", fontsize=18, fontweight="bold",
                 color="#1e2a3a", y=0.97)

    # 1) accuracy vs noise
    ax = axes[0, 0]
    for a in ALGOS:
        ax.plot(NOISE_LEVELS, accuracy[a], marker="o", color=COLORS[a], label=a)
    ax.set_title("Accuracy vs. input noise")
    ax.set_xlabel("Noise (% pixels flipped)"); ax.set_ylabel("Accuracy (%)")
    ax.set_ylim(0, 105); ax.grid(alpha=0.3); ax.legend(fontsize=8)

    # 2) training time
    ax = axes[0, 1]
    vals = [train_time[a] * 1000 for a in ALGOS]
    ax.bar(ALGOS, vals, color=[COLORS[a] for a in ALGOS])
    ax.set_title("Training time (10 glyphs)")
    ax.set_ylabel("milliseconds")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # 3) inference time per sample (avg over noise levels)
    ax = axes[1, 0]
    vals = [float(np.mean(infer_us[a])) for a in ALGOS]
    ax.bar(ALGOS, vals, color=[COLORS[a] for a in ALGOS])
    ax.set_title("Inference time per sample")
    ax.set_ylabel("microseconds")
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}", ha="center", va="bottom", fontsize=8)

    # 4) mean accuracy
    ax = axes[1, 1]
    vals = [float(np.mean(accuracy[a])) for a in ALGOS]
    ax.bar(ALGOS, vals, color=[COLORS[a] for a in ALGOS])
    ax.set_title("Mean accuracy (all noise levels)")
    ax.set_ylabel("Accuracy (%)"); ax.set_ylim(0, 105)
    for i, v in enumerate(vals):
        ax.text(i, v, f"{v:.1f}%", ha="center", va="bottom", fontsize=8)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    pdf.savefig(fig); plt.close(fig)


def table_page(pdf, accuracy, infer_us, train_time):
    fig = plt.figure(figsize=(8.27, 11.69))
    # full-page axes with explicit coordinates so the table and the notes
    # occupy non-overlapping bands of the page.
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")

    ax.text(0.08, 0.93, "Results Table", fontsize=20, fontweight="bold",
            color="#1e2a3a", transform=ax.transAxes)

    headers = ["Algorithm", "Train (ms)", "Infer (µs/ea)",
               "Acc @0%", "Acc @10%", "Acc @25%", "Mean acc"]
    rows = []
    for a in ALGOS:
        rows.append([
            a,
            f"{train_time[a]*1000:.1f}",
            f"{np.mean(infer_us[a]):.1f}",
            f"{accuracy[a][0]:.0f}%",
            f"{accuracy[a][2]:.0f}%",
            f"{accuracy[a][5]:.0f}%",
            f"{np.mean(accuracy[a]):.1f}%",
        ])
    # bbox = [left, bottom, width, height] in axes fraction → upper band only
    tbl = ax.table(cellText=rows, colLabels=headers, cellLoc="center",
                   bbox=[0.08, 0.72, 0.84, 0.14])
    tbl.auto_set_font_size(False); tbl.set_fontsize(9)
    for c in range(len(headers)):
        tbl[0, c].set_facecolor("#1e2a3a"); tbl[0, c].set_text_props(color="white")

    note = (
        "Notes\n"
        "• Dataset: 8x8 digits 0-9, 30 noisy variations per digit at each noise level.\n"
        "• Hebb is a Hopfield associative memory; with 10 correlated digits it is\n"
        "  over its ~0.138·N capacity (≈9 patterns for 64 cells), so accuracy is low\n"
        "  and degrades fast with noise.\n"
        "• Perceptron and Adaline are linear classifiers and separate the digit\n"
        "  glyphs well; Adaline's LMS training usually gives the cleanest margins.\n"
        "• All three are extremely fast to train and run at this scale (sub-millisecond)."
    )
    # notes band sits well below the table
    ax.text(0.08, 0.60, note, fontsize=10, va="top", ha="left",
            family="monospace", linespacing=1.6, transform=ax.transAxes)
    pdf.savefig(fig); plt.close(fig)


HOW_TO_USE = """\
This desktop app lets you draw 8x8 black/white patterns and classify them with
three classic neural-network algorithms: Hebb, Perceptron and Adaline.

The window has three tabs and a shared Algorithm picker in the dark header.

1) TRAIN tab
   • Draw a pattern on the 8x8 grid (click or drag cells; click again to erase).
   • Type a label and press "Add to Training Set".
   • Each algorithm keeps its OWN training set (switch with the header picker).
   • Click a saved pattern in the list to load it back onto the canvas.
   • Press "Train Network -> Save". This trains the selected algorithm and saves
     the patterns + weights to a folder named  ALGORITHM_8/  in the app's data
     directory (the project folder when run from source, ~/ANN_data when bundled).

2) TEST tab
   • Draw an input, then press "Run" (or Enter).
   • The app LOADS the saved model for the selected algorithm from disk and
     classifies your drawing.
   • Hebb additionally redraws the canvas with the recalled stored pattern.

3) STATS tab
   • Benchmarks all three models at once on the built-in digit dataset (0-9).
   • Choose the number of noisy variations per digit and the noise level.
   • "Train all 3 on digit glyphs" trains+saves the 8x8 models from the built-in
     digits (labels 0-9), so you can evaluate immediately.
   • "Run All Models" reports per-model accuracy and a per-digit breakdown.

Tip: for the cleanest results, train on distinct patterns and keep their number
well under the grid's capacity.
"""

ALGO_TEXT = """\
Hebb  (Hopfield associative memory)
   Learns a symmetric weight matrix W = sum(p·pT) over bipolar patterns.
   Recall iterates  x <- sign(W·x)  until it settles into a stored attractor.
   + Biologically inspired, content-addressable memory, one-shot training.
   - Tiny capacity (~0.138·N patterns), crosstalk between similar patterns,
     and spurious states -> imperfect recall when overloaded.

Perceptron
   A linear classifier with one weight vector per class. On each misclassified
   sample it nudges the correct class up and the wrong class down.
   + Simple, fast, converges if classes are linearly separable.
   - No probabilistic output; can be unstable on non-separable data.

Adaline  (Adaptive Linear Neuron, Widrow-Hoff / LMS)
   Like the Perceptron but trains on the CONTINUOUS output with gradient descent
   on squared error, giving smoother, better-conditioned decision boundaries.
   + Robust margins, smooth convergence, confidence scores via softmax.
   - Needs a learning-rate/epoch budget; still only a linear model.

Why compare them?
   On the 8x8 digits, the two linear classifiers (Perceptron, Adaline) strongly
   outperform Hebb, which is an associative memory rather than a classifier and
   is past its capacity with 10 digits. See the charts and table that follow.
"""


def main():
    print("Training models…")
    models, train_time = train_models()
    print("Benchmarking across noise levels…")
    accuracy, infer_us = benchmark(models)

    out = "report.pdf"
    with PdfPages(out) as pdf:
        text_page(pdf, "How to Use", HOW_TO_USE, mono=True)
        text_page(pdf, "The Algorithms", ALGO_TEXT, mono=True)
        charts_page(pdf, accuracy, infer_us, train_time)
        table_page(pdf, accuracy, infer_us, train_time)

    print(f"Wrote {out}")
    for a in ALGOS:
        print(f"  {a:<11} mean acc {np.mean(accuracy[a]):5.1f}%  "
              f"train {train_time[a]*1000:6.2f} ms  "
              f"infer {np.mean(infer_us[a]):6.2f} µs/sample")


if __name__ == "__main__":
    main()
