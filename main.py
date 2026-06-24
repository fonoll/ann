import os
import sys
import tkinter as tk
from tkinter import messagebox
import numpy as np
from algorithms import HebbNetwork, Perceptron, AdalineClassifier
import digits

# ── constants ──────────────────────────────────────────────────────
# When bundled by PyInstaller the app folder is read-only, so save the
# trained models in a writable per-user data folder instead.
if getattr(sys, "frozen", False):
    BASE_DIR = os.path.join(os.path.expanduser("~"), "ANN_data")
    os.makedirs(BASE_DIR, exist_ok=True)
else:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
GRID_SIZE  = 8           # app is locked to an 8×8 grid
ALGORITHMS = ["Hebb", "Perceptron", "Adaline"]

HDR_BG     = "#1e2a3a"
HDR_FG     = "#dce6f0"
BG         = "#f2f3f5"
TAB_ON     = "#1565C0"
TAB_OFF    = "#c9ced6"
TAB_ON_FG  = "#ffffff"
TAB_OFF_FG = "#3a4252"

# drawing-canvas colors
GRID_PX  = 480           # total drawing area in pixels
CELL_ON  = "#000000"
CELL_OFF = "#ffffff"
GRID_CLR = "#bbbbbb"      # grid line color
BORDER   = "#222222"      # outer frame color


def model_dir(algo: str, size: int) -> str:
    """Folder that stores one algorithm's data for one grid size."""
    return os.path.join(BASE_DIR, f"{algo}_{size}")


class DrawCanvas:
    """A black/white drawing grid backed by a single tk.Canvas.

    Uses canvas rectangle items (not one widget per cell) so it stays fast
    and responsive even on large grids like 32x32 (1024 cells).
    Click or drag to paint cells black; click a black cell to erase it.
    Place `self.frame` in a parent with pack()/grid().
    """

    def __init__(self, parent: tk.Widget, size: int = 8):
        self._size = 0
        self._cs = 1
        self._data: np.ndarray = np.array([])
        self._rects: list = []
        self._drag_val = 1

        # outer border frame supplies the visible edge; highlightthickness=0
        # keeps event coords aligned with item coords.
        self.frame = tk.Frame(parent, bg=BORDER, bd=0)
        self.widget = tk.Canvas(self.frame, bg=CELL_OFF, cursor="crosshair",
                                highlightthickness=0, bd=0)
        self.widget.pack(padx=2, pady=2)

        self.widget.bind("<Button-1>",  self._press)
        self.widget.bind("<B1-Motion>", self._motion)

        self.set_size(size)

    # public ─────────────────────────────────────────────────────────

    def set_size(self, size: int):
        self._size = size
        self._cs = max(4, GRID_PX // size)
        px = self._cs * size
        self.widget.config(width=px, height=px)
        self.widget.delete("all")
        self._rects = []
        self._data = np.zeros(size * size, dtype=np.int32)
        ol = GRID_CLR if self._cs >= 5 else ""
        cs = self._cs
        for i in range(size * size):
            r, c = divmod(i, size)
            x1, y1 = c * cs, r * cs
            rid = self.widget.create_rectangle(
                x1, y1, x1 + cs, y1 + cs, fill=CELL_OFF, outline=ol, width=1)
            self._rects.append(rid)

    def reset(self):
        self._data[:] = 0
        for rid in self._rects:
            self.widget.itemconfig(rid, fill=CELL_OFF)

    def get_data(self) -> np.ndarray:
        return self._data.copy()

    def set_data(self, data: np.ndarray):
        self._data[:] = data
        for i, rid in enumerate(self._rects):
            self.widget.itemconfig(rid, fill=CELL_ON if data[i] else CELL_OFF)

    # private ────────────────────────────────────────────────────────

    def _idx(self, x: int, y: int):
        c, r = int(x) // self._cs, int(y) // self._cs
        if 0 <= r < self._size and 0 <= c < self._size:
            return r * self._size + c
        return None

    def _paint(self, idx: int, val: int):
        if self._data[idx] != val:
            self._data[idx] = val
            self.widget.itemconfig(self._rects[idx],
                                   fill=CELL_ON if val else CELL_OFF)

    def _press(self, ev):
        idx = self._idx(ev.x, ev.y)
        if idx is None:
            return
        # the toggle of the first cell sets the paint value for the whole drag
        self._drag_val = 1 - int(self._data[idx])
        self._paint(idx, self._drag_val)

    def _motion(self, ev):
        idx = self._idx(ev.x, ev.y)
        if idx is not None:
            self._paint(idx, self._drag_val)


class AnnApp:

    def __init__(self, root: tk.Tk):
        self.root = root
        root.title("ANN Classifier")
        root.configure(bg=BG)
        root.resizable(False, False)

        # shared state
        self._grid_var    = tk.IntVar(value=8)
        self._algo_var    = tk.StringVar(value="Hebb")
        self._result_var  = tk.StringVar(value="—")
        self._train_stat  = tk.StringVar(value="")
        self._test_stat   = tk.StringVar(value="")
        self._count_var   = tk.StringVar(value="")
        self._train_hdr   = tk.StringVar(value="")

        # in-memory working training set, per algorithm (for current grid size)
        self._patterns = {a: [] for a in ALGORITHMS}   # a -> [(label, ndarray)]

        self._build_header()
        self._build_tabbar()
        self._build_pages()
        self._show_page("train")
        self._reload_all_patterns()
        self._on_algo_change()

        root.bind("<Return>", lambda _e: self._run())

    # ── header ───────────────────────────────────────────────────────

    def _build_header(self):
        hdr = tk.Frame(self.root, bg=HDR_BG, padx=20, pady=14)
        hdr.pack(fill=tk.X)
        hdr.configure(height=48)
        hdr.pack_propagate(False)

        # algorithm picker — hidden on the STATS tab (which runs all 3 at once)
        self._algo_frame = tk.Frame(hdr, bg=HDR_BG)
        self._algo_frame.pack(side=tk.LEFT)

        tk.Label(self._algo_frame, text="Algorithm:", bg=HDR_BG, fg=HDR_FG,
                 font=("Arial", 11, "bold")).pack(side=tk.LEFT, padx=(0, 10))
        for algo in ALGORITHMS:
            tk.Radiobutton(self._algo_frame, text=algo,
                           variable=self._algo_var, value=algo,
                           command=self._on_algo_change,
                           bg=HDR_BG, fg=HDR_FG, selectcolor="#2e3f52",
                           activebackground=HDR_BG, activeforeground=HDR_FG,
                           font=("Arial", 11)).pack(side=tk.LEFT, padx=5)

        # shown instead when the picker is hidden
        self._algo_hint = tk.Label(
            hdr, text="STATS — evaluating all three algorithms",
            bg=HDR_BG, fg="#7e8aa0", font=("Arial", 11, "italic"))

    # ── tab bar ──────────────────────────────────────────────────────

    def _build_tabbar(self):
        bar = tk.Frame(self.root, bg=BG, pady=4)
        bar.pack(fill=tk.X, padx=8, pady=(6, 0))

        self._tab_btns = {}
        for key, text in (("train", "TRAIN"), ("test", "TEST"), ("stats", "STATS")):
            b = tk.Button(bar, text=text, font=("Arial", 13, "bold"),
                          relief=tk.FLAT, bd=0, cursor="hand2",
                          padx=40, pady=10,
                          command=lambda k=key: self._show_page(k))
            b.pack(side=tk.LEFT, padx=(0, 4))
            self._tab_btns[key] = b

    def _build_pages(self):
        container = tk.Frame(self.root, bg=BG)
        container.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self._pages = {}
        for key in ("train", "test", "stats"):
            page = tk.Frame(container, bg=BG)
            page.grid(row=0, column=0, sticky="nsew")
            self._pages[key] = page

        self._build_train_page(self._pages["train"])
        self._build_test_page(self._pages["test"])
        self._build_stats_page(self._pages["stats"])

    def _show_page(self, key: str):
        self._pages[key].lift()
        if key == "stats":
            self._algo_frame.pack_forget()
            self._algo_hint.pack(side=tk.LEFT)
        else:
            self._algo_hint.pack_forget()
            self._algo_frame.pack(side=tk.LEFT)
        if key == "test":
            self._refresh_test_status()
        elif key == "stats":
            self._refresh_stats_status()
        for k, b in self._tab_btns.items():
            on = (k == key)
            b.config(bg=TAB_ON if on else TAB_OFF,
                     fg=TAB_ON_FG if on else TAB_OFF_FG,
                     activebackground=TAB_ON if on else TAB_OFF,
                     activeforeground=TAB_ON_FG if on else TAB_OFF_FG)

    # ── TRAIN page ───────────────────────────────────────────────────

    def _build_train_page(self, parent):
        left = tk.Frame(parent, bg=BG, padx=18, pady=14)
        left.pack(side=tk.LEFT, anchor="n")

        tk.Label(left, textvariable=self._train_hdr, bg=BG,
                 font=("Arial", 11, "bold"), fg="#1565C0").pack(anchor="w", pady=(0, 6))

        self._dc_train = DrawCanvas(left, self._grid_var.get())
        self._dc_train.frame.pack()

        self._flat_btn(left, "Reset", self._dc_train.reset, "#546E7A").pack(
            anchor="center", pady=(10, 0))

        right = tk.Frame(parent, bg=BG, padx=14, pady=14, width=250)
        right.pack(side=tk.LEFT, fill=tk.Y, anchor="n")
        right.pack_propagate(False)

        self._heading(right, "Pattern Label")
        self._label_entry = tk.Entry(right, font=("Arial", 11), width=22,
                                     relief=tk.SOLID, bd=1)
        self._label_entry.pack(padx=6, pady=(0, 6))
        self._label_entry.insert(0, "Pattern A")

        tk.Button(right, text="Add to Training Set",
                  command=self._add_pattern,
                  font=("Arial", 11, "bold"), bg="#2E7D32", fg="white",
                  activebackground="#1B5E20", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2", pady=6).pack(
            fill=tk.X, padx=6, pady=(0, 4))

        self._spacer(right, 10)
        self._heading(right, "Training Set")

        lf = tk.Frame(right, bg=BG)
        lf.pack(fill=tk.BOTH, expand=True, padx=6)
        sb = tk.Scrollbar(lf)
        sb.pack(side=tk.RIGHT, fill=tk.Y)
        self._listbox = tk.Listbox(lf, yscrollcommand=sb.set,
                                   font=("Courier", 10), height=8,
                                   selectbackground="#1565C0",
                                   activestyle="none", bd=1, relief=tk.SOLID)
        self._listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        self._listbox.bind("<<ListboxSelect>>", self._on_pattern_select)
        sb.config(command=self._listbox.yview)

        row = tk.Frame(right, bg=BG)
        row.pack(fill=tk.X, padx=6, pady=(5, 2))
        tk.Button(row, text="Remove", command=self._remove_pattern,
                  font=("Arial", 10), bg="#C62828", fg="white",
                  relief=tk.FLAT, cursor="hand2",
                  padx=8, pady=3).pack(side=tk.LEFT)
        tk.Button(row, text="Clear All", command=self._clear_patterns,
                  font=("Arial", 10), bg="#757575", fg="white",
                  relief=tk.FLAT, cursor="hand2",
                  padx=8, pady=3).pack(side=tk.LEFT, padx=(6, 0))

        tk.Label(right, textvariable=self._count_var, bg=BG,
                 font=("Arial", 9), fg="#999").pack(anchor="w", padx=6, pady=(2, 4))

        tk.Frame(right, height=1, bg="#bbbbbb").pack(fill=tk.X, padx=6, pady=4)

        # Train & save lives here now
        tk.Button(right, text="Train Network  →  Save",
                  command=self._train_network,
                  font=("Arial", 11, "bold"), bg="#E65100", fg="white",
                  activebackground="#BF360C", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2", pady=7).pack(
            fill=tk.X, padx=6, pady=(2, 4))

        tk.Label(right, textvariable=self._train_stat, bg=BG,
                 font=("Arial", 9), fg="#777",
                 wraplength=222, justify="left").pack(anchor="w", padx=6)

    # ── TEST page ────────────────────────────────────────────────────

    def _build_test_page(self, parent):
        left = tk.Frame(parent, bg=BG, padx=18, pady=14)
        left.pack(side=tk.LEFT, anchor="n")

        tk.Label(left, text="Draw test input:", bg=BG,
                 font=("Arial", 10, "bold"), fg="#555").pack(anchor="w", pady=(0, 6))

        self._dc_test = DrawCanvas(left, self._grid_var.get())
        self._dc_test.frame.pack()

        self._flat_btn(left, "Reset", self._reset_test, "#546E7A").pack(
            anchor="center", pady=(10, 0))

        right = tk.Frame(parent, bg=BG, padx=14, pady=14, width=250)
        right.pack(side=tk.LEFT, fill=tk.Y, anchor="n")
        right.pack_propagate(False)

        self._spacer(right, 4)
        tk.Button(right, text="Run  ↵",
                  command=self._run,
                  font=("Arial", 11, "bold"), bg="#1565C0", fg="white",
                  activebackground="#0D47A1", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2", pady=8).pack(
            fill=tk.X, padx=6, pady=(0, 6))

        self._spacer(right, 10)
        self._heading(right, "Result")

        result_frame = tk.Frame(right, bg="white",
                                highlightthickness=1, highlightbackground="#cccccc")
        result_frame.pack(fill=tk.X, padx=6, pady=(0, 6))
        tk.Label(result_frame, textvariable=self._result_var,
                 bg="white", font=("Arial", 13),
                 wraplength=210, justify="left",
                 padx=10, pady=14, fg="#111111").pack(fill=tk.X)

        tk.Label(right, textvariable=self._test_stat,
                 bg=BG, font=("Arial", 9), fg="#777",
                 wraplength=222, justify="left").pack(anchor="w", padx=6)

    # ── helpers ──────────────────────────────────────────────────────

    def _flat_btn(self, parent, text, cmd, color):
        return tk.Button(parent, text=text, command=cmd,
                         font=("Arial", 10), bg=color, fg="white",
                         activebackground=color, activeforeground="white",
                         relief=tk.FLAT, cursor="hand2", padx=18, pady=4)

    def _heading(self, parent, text):
        tk.Label(parent, text=text, bg=BG,
                 font=("Arial", 10, "bold"), fg="#444").pack(anchor="w", padx=6)
        tk.Frame(parent, height=1, bg="#bbbbbb").pack(
            fill=tk.X, padx=6, pady=(2, 6))

    def _spacer(self, parent, h=10):
        tk.Frame(parent, height=h, bg=BG).pack()

    def _refresh_listbox(self):
        algo = self._algo_var.get()
        self._listbox.delete(0, tk.END)
        for i, (lbl, _) in enumerate(self._patterns[algo]):
            self._listbox.insert(tk.END, f"  {i + 1:>2}.  {lbl}")
        n = len(self._patterns[algo])
        self._count_var.set(f"{n} pattern(s) for {algo}")

    def _on_pattern_select(self, _event):
        sel = self._listbox.curselection()
        if not sel:
            return
        algo = self._algo_var.get()
        idx = sel[0]
        if idx < len(self._patterns[algo]):
            label, data = self._patterns[algo][idx]
            self._dc_train.set_data(data)
            self._train_stat.set(f'Showing "{label}" on the canvas.')

    # ── persistence ──────────────────────────────────────────────────

    def _load_patterns(self, algo: str, size: int):
        """Return [(label, ndarray)] saved for (algo, size), or []."""
        pf = os.path.join(model_dir(algo, size), "patterns.npz")
        if not os.path.exists(pf):
            return []
        d = np.load(pf, allow_pickle=False)
        patterns = d["patterns"]
        labels = [str(l) for l in d["labels"]]
        return [(labels[i], patterns[i].astype(np.int32))
                for i in range(len(labels))]

    def _reload_all_patterns(self):
        size = self._grid_var.get()
        for a in ALGORITHMS:
            self._patterns[a] = self._load_patterns(a, size)

    def _model_exists(self, algo: str, size: int) -> bool:
        folder = model_dir(algo, size)
        return (os.path.exists(os.path.join(folder, "weights.npz"))
                and os.path.exists(os.path.join(folder, "patterns.npz")))

    def _load_model(self, algo: str, size: int):
        """Reconstruct a trained model from disk; returns dict or None."""
        if not self._model_exists(algo, size):
            return None
        folder = model_dir(algo, size)
        pd = np.load(os.path.join(folder, "patterns.npz"), allow_pickle=False)
        wd = np.load(os.path.join(folder, "weights.npz"), allow_pickle=False)
        patterns = pd["patterns"].astype(np.float32)
        labels = [str(l) for l in pd["labels"]]
        n, n_cls = patterns.shape[1], len(labels)

        if algo == "Hebb":
            net = HebbNetwork(n)
            net.W = wd["W"].astype(np.float32)
            net.b = wd["b"].astype(np.float32) if "b" in wd else np.zeros(n, dtype=np.float32)
            bip = (2 * patterns - 1).astype(np.float32)
            return {"net": net, "labels": labels, "bip": bip}
        elif algo == "Perceptron":
            net = Perceptron(n, n_cls)
            net.W = wd["W"].astype(np.float32)
            net.b = wd["b"].astype(np.float32)
            return {"net": net, "labels": labels}
        else:  # Adaline
            net = AdalineClassifier(n, n_cls)
            net.W = wd["W"].astype(np.float32)
            net.b = wd["b"].astype(np.float32)
            return {"net": net, "labels": labels}

    # ── header callbacks ─────────────────────────────────────────────

    def _on_algo_change(self):
        algo = self._algo_var.get()
        size = self._grid_var.get()
        self._train_hdr.set(f"Training data for: {algo}  ({size}×{size})")
        self._refresh_listbox()
        self._result_var.set("—")
        saved = "saved model on disk" if self._model_exists(algo, size) \
            else "no saved model yet"
        self._train_stat.set(f"{algo}: {saved}.")
        self._refresh_test_status()

    def _refresh_test_status(self):
        algo = self._algo_var.get()
        size = self._grid_var.get()
        if self._model_exists(algo, size):
            self._test_stat.set(
                f"Loaded from: {algo}_{size}/  —  Run to classify.")
        else:
            self._test_stat.set(
                f"No saved {algo} model for {size}×{size}. "
                f"Train it in the TRAIN tab first.")

    # ── TRAIN actions ────────────────────────────────────────────────

    def _add_pattern(self):
        algo = self._algo_var.get()
        label = self._label_entry.get().strip()
        if not label:
            messagebox.showwarning("Label required", "Enter a label first.")
            return
        data = self._dc_train.get_data()
        if not data.any():
            messagebox.showwarning("Empty canvas", "Draw something first.")
            return
        self._patterns[algo].append((label, data.copy()))
        self._refresh_listbox()
        self._train_stat.set(f'Added "{label}" to {algo}. Click Train Network to save.')
        cur = self._label_entry.get().strip()
        if cur and cur[-1].isalpha() and cur[-1] != "Z":
            self._label_entry.delete(0, tk.END)
            self._label_entry.insert(0, cur[:-1] + chr(ord(cur[-1]) + 1))

    def _remove_pattern(self):
        sel = self._listbox.curselection()
        if not sel:
            return
        algo = self._algo_var.get()
        self._patterns[algo].pop(sel[0])
        self._refresh_listbox()

    def _clear_patterns(self):
        algo = self._algo_var.get()
        if not self._patterns[algo]:
            return
        if not messagebox.askyesno("Clear all",
                                   f"Remove all {algo} training patterns?"):
            return
        self._patterns[algo].clear()
        self._refresh_listbox()
        self._train_stat.set(f"{algo} working set cleared "
                             f"(saved model on disk is untouched).")

    def _train_network(self):
        algo  = self._algo_var.get()
        size  = self._grid_var.get()
        pairs = self._patterns[algo]
        if not pairs:
            messagebox.showinfo("No data",
                                f"Add {algo} patterns first.")
            return

        self._train_stat.set(f"Training {algo}…")
        self.root.update_idletasks()

        labels   = [lbl for lbl, _ in pairs]
        patterns = np.array([d for _, d in pairs], dtype=np.float32)
        n_cls    = self._train_and_save(algo, size, patterns, labels)

        self._train_stat.set(
            f"{algo} trained on {n_cls} pattern(s).\n"
            f"Saved to:  {algo}_{size}/")
        self._refresh_test_status()

    def _train_and_save(self, algo: str, size: int,
                        patterns: np.ndarray, labels: list) -> int:
        """Train one algorithm and persist patterns + weights to its folder."""
        n = patterns.shape[1]
        n_cls = len(labels)
        folder = model_dir(algo, size)
        os.makedirs(folder, exist_ok=True)

        np.savez(os.path.join(folder, "patterns.npz"),
                 patterns=patterns, labels=np.array(labels))

        if algo == "Hebb":
            net = HebbNetwork(n)
            net.train((2 * patterns - 1).astype(np.float32))
            np.savez(os.path.join(folder, "weights.npz"), W=net.W, b=net.b)
        elif algo == "Perceptron":
            net = Perceptron(n, n_cls)
            net.train(patterns, list(range(n_cls)))
            np.savez(os.path.join(folder, "weights.npz"), W=net.W, b=net.b)
        else:  # Adaline
            net = AdalineClassifier(n, n_cls)
            net.train(patterns, list(range(n_cls)))
            np.savez(os.path.join(folder, "weights.npz"), W=net.W, b=net.b)
        return n_cls

    def _predict_label(self, algo: str, model: dict, vec: np.ndarray) -> str:
        """Predict the class label for one input vector using a loaded model."""
        x = vec.astype(np.float32)
        if algo == "Hebb":
            recalled = model["net"].recall(2 * x - 1)
            best = int(np.argmax([np.dot(recalled, p) for p in model["bip"]]))
            return model["labels"][best]
        elif algo == "Perceptron":
            return model["labels"][model["net"].predict(x)]
        else:  # Adaline
            pred, _ = model["net"].predict(x)
            return model["labels"][pred]

    # ── TEST actions ─────────────────────────────────────────────────

    def _reset_test(self):
        self._dc_test.reset()
        self._result_var.set("—")

    def _run(self):
        algo = self._algo_var.get()
        size = self._grid_var.get()
        model = self._load_model(algo, size)   # load from the folder
        if model is None:
            messagebox.showinfo(
                "No saved model",
                f"No saved {algo} model for {size}×{size}.\n"
                f"Train it in the TRAIN tab first.")
            return

        x = self._dc_test.get_data().astype(np.float32)
        labels = model["labels"]

        if algo == "Hebb":
            net = model["net"]
            recalled = net.recall(2 * x - 1)
            best = int(np.argmax([np.dot(recalled, p) for p in model["bip"]]))
            recalled_bin = np.clip((recalled + 1) / 2, 0, 1).astype(np.int32)
            self._dc_test.set_data(recalled_bin)
            self._result_var.set(f"Recalled:\n{labels[best]}")

        elif algo == "Perceptron":
            pred = model["net"].predict(x)
            self._result_var.set(f"Class:\n{labels[pred]}")

        else:  # Adaline
            pred, conf = model["net"].predict(x)
            self._result_var.set(f"Class:\n{labels[pred]}\n{conf:.1%} confidence")

        self._test_stat.set(f"Loaded {algo}_{size}/ and classified.")

    # ── STATS page ───────────────────────────────────────────────────

    def _build_stats_page(self, parent):
        self._stats_count  = tk.IntVar(value=5)
        self._stats_noise  = tk.IntVar(value=6)
        self._ds_size_var  = tk.StringVar(value="")
        self._stats_status = tk.StringVar(value="")

        wrap = tk.Frame(parent, bg=BG, padx=20, pady=14)
        wrap.pack(fill=tk.BOTH, expand=True)

        tk.Label(wrap, text="Benchmark all models on the built-in digit dataset",
                 bg=BG, font=("Arial", 13, "bold"), fg="#1e2a3a").pack(anchor="w")
        tk.Label(wrap,
                 text="Uses the 8×8 digits 0–9 (with noisy variations) and the "
                      "8×8 saved models. Models must be trained with labels "
                      "\"0\"–\"9\" for accuracy to be meaningful.",
                 bg=BG, font=("Arial", 9), fg="#777",
                 justify="left", wraplength=560).pack(anchor="w", pady=(2, 10))

        # dataset controls
        ctl = tk.Frame(wrap, bg=BG)
        ctl.pack(fill=tk.X, pady=(0, 6))
        tk.Label(ctl, text="Variations per digit:", bg=BG,
                 font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Spinbox(ctl, from_=1, to=20, width=4, textvariable=self._stats_count,
                   command=self._update_ds_label, font=("Arial", 10)).pack(
            side=tk.LEFT, padx=(4, 16))
        tk.Label(ctl, text="Noise %:", bg=BG, font=("Arial", 10)).pack(side=tk.LEFT)
        tk.Scale(ctl, from_=0, to=25, orient=tk.HORIZONTAL, length=160,
                 variable=self._stats_noise, bg=BG, highlightthickness=0,
                 command=lambda _v: self._update_ds_label()).pack(
            side=tk.LEFT, padx=(4, 16))
        tk.Label(ctl, textvariable=self._ds_size_var, bg=BG,
                 font=("Arial", 10, "bold"), fg="#1565C0").pack(side=tk.LEFT)

        # buttons
        btns = tk.Frame(wrap, bg=BG)
        btns.pack(fill=tk.X, pady=(4, 8))
        tk.Button(btns, text="Train all 3 on digit glyphs",
                  command=self._train_all_on_digits,
                  font=("Arial", 10, "bold"), bg="#2E7D32", fg="white",
                  activebackground="#1B5E20", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2", padx=12, pady=6).pack(side=tk.LEFT)
        tk.Button(btns, text="Run All Models",
                  command=self._run_stats,
                  font=("Arial", 10, "bold"), bg="#1565C0", fg="white",
                  activebackground="#0D47A1", activeforeground="white",
                  relief=tk.FLAT, cursor="hand2", padx=12, pady=6).pack(
            side=tk.LEFT, padx=(8, 0))

        tk.Label(wrap, textvariable=self._stats_status, bg=BG,
                 font=("Arial", 9), fg="#777").pack(anchor="w", pady=(0, 6))

        # results report (monospace, read-only)
        self._stats_text = tk.Text(wrap, height=18, width=60,
                                   font=("Courier", 11), bg="white",
                                   relief=tk.SOLID, bd=1, padx=10, pady=8,
                                   wrap="none")
        self._stats_text.pack(fill=tk.BOTH, expand=True)
        self._stats_text.insert("1.0",
                                "Click 'Run All Models' to evaluate.")
        self._stats_text.config(state=tk.DISABLED)

        self._update_ds_label()

    def _update_ds_label(self):
        self._ds_size_var.set(f"= {10 * self._stats_count.get()} samples")

    def _refresh_stats_status(self):
        present = [a for a in ALGORITHMS if self._model_exists(a, 8)]
        if present:
            self._stats_status.set("8×8 models found on disk: " + ", ".join(present))
        else:
            self._stats_status.set("No 8×8 models yet — use 'Train all 3 on digit "
                                   "glyphs', or train them in the TRAIN tab.")

    def _set_stats_text(self, text: str):
        self._stats_text.config(state=tk.NORMAL)
        self._stats_text.delete("1.0", tk.END)
        self._stats_text.insert("1.0", text)
        self._stats_text.config(state=tk.DISABLED)

    def _train_all_on_digits(self):
        if not messagebox.askyesno(
            "Train on digit glyphs",
            "Train and OVERWRITE the 8×8 models for all three algorithms "
            "using the built-in digit glyphs (labels 0–9)?\n\n"
            "This replaces any existing 8×8 models."):
            return
        X, labels = digits.base_glyphs()
        for algo in ALGORITHMS:
            self._train_and_save(algo, 8, X, labels)
        self._reload_all_patterns()
        self._refresh_listbox()
        self._refresh_stats_status()
        messagebox.showinfo(
            "Done",
            "Trained all three algorithms on the 8×8 digit glyphs.\n"
            "Now click 'Run All Models' to evaluate.")

    def _run_stats(self):
        variations = self._stats_count.get()
        noise_pct  = self._stats_noise.get()
        X, y = digits.generate_dataset(variations, noise_pct / 100.0, seed=0)
        total = len(y)

        lines = [
            f"Dataset: {total} samples  "
            f"(10 digits × {variations} variations), noise {noise_pct}%",
            "",
            f"{'Model':<12}{'Trained':<9}{'Accuracy':<11}{'Correct/Total'}",
            "-" * 50,
        ]
        per_digit = {}
        for algo in ALGORITHMS:
            model = self._load_model(algo, 8)
            if model is None:
                lines.append(f"{algo:<12}{'no':<9}{'—':<11}{'—'}")
                per_digit[algo] = None
                continue
            self._stats_status.set(f"Evaluating {algo}…")
            self.root.update_idletasks()
            correct = 0
            pc = {d: 0 for d in digits.DIGITS}
            pt = {d: 0 for d in digits.DIGITS}
            for vec, true in zip(X, y):
                pred = self._predict_label(algo, model, vec)
                pt[true] += 1
                if pred == true:
                    correct += 1
                    pc[true] += 1
            acc = correct / total * 100
            lines.append(f"{algo:<12}{'yes':<9}{acc:>6.1f}%    {correct:>3}/{total}")
            per_digit[algo] = (pc, pt)

        # per-digit accuracy breakdown
        lines += ["", "Per-digit accuracy (%)"]
        head = f"{'Model':<12}" + "".join(f"{d:>5}" for d in digits.DIGITS)
        lines.append(head)
        lines.append("-" * len(head))
        for algo in ALGORITHMS:
            pd = per_digit[algo]
            if pd is None:
                lines.append(f"{algo:<12}" + "".join(f"{'-':>5}" for _ in digits.DIGITS))
                continue
            pc, pt = pd
            cells = "".join(
                f"{(pc[d] / pt[d] * 100):>5.0f}" if pt[d] else f"{'-':>5}"
                for d in digits.DIGITS)
            lines.append(f"{algo:<12}{cells}")

        self._set_stats_text("\n".join(lines))
        self._refresh_stats_status()


# ── entry point ────────────────────────────────────────────────────
if __name__ == "__main__":
    root = tk.Tk()
    AnnApp(root)
    root.mainloop()
