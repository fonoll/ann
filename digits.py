"""Built-in 8x8 digit dataset (0-9) for evaluating the trained models.

Each digit has one hand-drawn 8x8 glyph; `generate_dataset` produces several
noisy variations per digit so models can be tested on data they were not
trained on.
"""
import numpy as np

DIGITS = [str(d) for d in range(10)]
GRID = 8

# 8 rows x 8 cols, '#' = on (black), '.' = off (white)
DIGIT_GLYPHS = {
    "0": [
        ".######.",
        "##....##",
        "##....##",
        "##....##",
        "##....##",
        "##....##",
        "##....##",
        ".######.",
    ],
    "1": [
        "...##...",
        "..###...",
        ".####...",
        "...##...",
        "...##...",
        "...##...",
        "...##...",
        ".######.",
    ],
    "2": [
        ".######.",
        "##....##",
        ".....##.",
        "....##..",
        "...##...",
        "..##....",
        ".##.....",
        "########",
    ],
    "3": [
        ".######.",
        "##....##",
        "......##",
        "...####.",
        "......##",
        "......##",
        "##....##",
        ".######.",
    ],
    "4": [
        "....###.",
        "...####.",
        "..##.##.",
        ".##..##.",
        "########",
        ".....##.",
        ".....##.",
        ".....##.",
    ],
    "5": [
        "########",
        "##......",
        "##......",
        "#######.",
        "......##",
        "......##",
        "##....##",
        ".######.",
    ],
    "6": [
        "..####..",
        ".##.....",
        "##......",
        "#######.",
        "##....##",
        "##....##",
        "##....##",
        ".######.",
    ],
    "7": [
        "########",
        "......##",
        ".....##.",
        "....##..",
        "...##...",
        "..##....",
        "..##....",
        "..##....",
    ],
    "8": [
        ".######.",
        "##....##",
        "##....##",
        ".######.",
        "##....##",
        "##....##",
        "##....##",
        ".######.",
    ],
    "9": [
        ".######.",
        "##....##",
        "##....##",
        "##....##",
        ".#######",
        "......##",
        ".....##.",
        "..####..",
    ],
}


def _parse(rows) -> np.ndarray:
    assert len(rows) == GRID, "glyph must have 8 rows"
    flat = np.zeros(GRID * GRID, dtype=np.float32)
    for r, row in enumerate(rows):
        assert len(row) == GRID, f"row '{row}' must be 8 chars"
        for c, ch in enumerate(row):
            flat[r * GRID + c] = 1.0 if ch == "#" else 0.0
    return flat


def base_glyphs():
    """Return (X, labels): one clean 8x8 glyph per digit 0-9."""
    X = np.array([_parse(DIGIT_GLYPHS[d]) for d in DIGITS], dtype=np.float32)
    return X, list(DIGITS)


def generate_dataset(variations: int = 5, noise: float = 0.06, seed: int = 0):
    """Return (X, y): `variations` noisy copies per digit at 8x8.

    noise = fraction of the 64 pixels randomly flipped in each variation.
    The first variation of every digit is the clean glyph (noise-free).
    """
    rng = np.random.default_rng(seed)
    baseX, baseY = base_glyphs()
    n_pixels = GRID * GRID
    k = int(round(noise * n_pixels))

    X, y = [], []
    for vec, label in zip(baseX, baseY):
        for v_i in range(variations):
            sample = vec.copy()
            if v_i > 0 and k > 0:                  # keep one clean copy
                idx = rng.choice(n_pixels, size=k, replace=False)
                sample[idx] = 1.0 - sample[idx]
            X.append(sample)
            y.append(label)
    return np.array(X, dtype=np.float32), y
