# ANN Pattern Classifier

A desktop GUI for drawing patterns and classifying them with three classic neural-network algorithms — Hebb, Perceptron, and Adaline — all implemented from scratch in NumPy.

## Features

- **Draw** black/white patterns on an 8×8 grid (click or drag cells to paint/erase)
- **Train** any of the three algorithms on your custom patterns, with labels you define
- **Test** classification by drawing an input and hitting Run (or Enter)
  - Hebb additionally redraws the canvas with the recalled stored pattern
  - Adaline shows a softmax confidence score
- **Stats** tab benchmarks all three algorithms at once on the built-in digit dataset (0–9), with configurable noise level and variation count
- **Persistent models** — weights and patterns are saved to disk as `.npz` files and reloaded automatically
- **PDF report** — `generate_report.py` produces a four-page PDF with algorithm explanations, accuracy-vs-noise charts, training/inference timings, and a results table

## Algorithms

| Algorithm | Training rule | Key trait |
|-----------|---------------|-----------|
| **Hebb** | Hopfield associative memory — `W += outer(p, p)` over bipolar patterns | Content-addressable recall; one-shot training; limited capacity (~0.138·N patterns) |
| **Perceptron** | Mistake-driven correction — reward correct class, penalise wrong class | Converges on linearly separable data; no probabilistic output |
| **Adaline** | LMS / delta rule — gradient descent on continuous squared error | Smoother margins than Perceptron; softmax confidence scores |

## Requirements

- Python 3.9+
- `numpy`
- `tkinter` (included with most Python installations)
- `matplotlib` (only for `generate_report.py`)

Install dependencies:

```bash
pip install numpy matplotlib
```

## Usage

### Run from source

```bash
python main.py
```

### Generate the PDF report

```bash
python generate_report.py
# writes report.pdf in the current directory
```

### Build a standalone macOS .app

```bash
bash build_app.sh
# output: dist/ANN Classifier.app
```

The build script creates an isolated virtual environment, installs `numpy` and `PyInstaller`, and produces a windowed `.app` bundle. Trained models are written to `~/ANN_data/` when running from the bundle (or to the project directory when running from source).

## Project structure

```
main.py             — Tkinter GUI (DrawCanvas, AnnApp)
algorithms.py       — HebbNetwork, Perceptron, AdalineClassifier
digits.py           — Built-in 8×8 digit glyphs (0–9) and noisy dataset generator
generate_report.py  — PDF report with benchmarks and algorithm notes
build_app.sh        — macOS .app build via PyInstaller
```

## How the app works

1. **TRAIN tab** — draw a pattern, enter a label, press *Add to Training Set*. Each algorithm keeps its own set. Press *Train Network → Save* to train and write weights to `ALGORITHM_8/` in the data directory.
2. **TEST tab** — draw an input and press *Run*. The app loads the saved model from disk and classifies the drawing.
3. **STATS tab** — press *Train all 3 on digit glyphs* to train all algorithms on the built-in 0–9 dataset, then *Run All Models* to see accuracy and per-digit breakdowns.

## License

See [LICENSE](LICENSE).
