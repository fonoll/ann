import numpy as np


class HebbNetwork:
    """
    Associative memory trained with the Hebb rule (bipolar ±1 patterns).

    The idea: neurons that fire together, wire together.
    If pixel i and pixel j are both ON (+1) or both OFF (-1) in a pattern,
    the connection w_ij gets stronger. If they differ, it gets weaker.

    Algorithm from the paper — applied once per pattern:
        Step 0: w_i = 0,  b = 0          (start with no knowledge)
        Step 1: for each pattern p:
        Step 2:   x_i = s_i = p_i        (input  = the pattern itself)
        Step 3:   y    = t   = p          (target = the same pattern)
        Step 4:   w_i(new) = w_i(old) + x_i * y
                  b(new)   = b(old)   + y

    After training, recall(x) snaps a noisy / partial input back to the
    nearest stored pattern by repeatedly computing: x ← sign(W·x + b)
    """

    def __init__(self, n: int):
        # n = number of neurons = number of pixels in the grid
        self.n = n

        # Step 0: weight matrix and bias start at zero
        self.W = np.zeros((n, n), dtype=np.float32)  # shape (n, n)
        self.b = np.zeros(n,      dtype=np.float32)  # shape (n,)

    def train(self, patterns: np.ndarray):
        """
        Learn every pattern in one pass (no gradient, no epochs).

        patterns: array of shape (N, n) with bipolar values −1 and +1.
                  Each row is one stored pattern.
        """
        # Step 0: reset so re-training always starts clean
        self.W = np.zeros((self.n, self.n), dtype=np.float32)
        self.b = np.zeros(self.n,           dtype=np.float32)

        for p in patterns:
            p = p.astype(np.float32)  # ensure float for the math below

            # Step 2 & 3: x = s = p  and  y = t = p
            # (for associative memory the input and target are the same pattern)

            # Step 4 — weights:
            #   w_i(new) = w_i(old) + x_i * y   for every neuron i
            #   Doing this for ALL output neurons j at once gives:
            #   W[i,j] += p[i] * p[j]  →  W += outer(p, p)
            self.W += np.outer(p, p)

            # Step 4 — bias:
            #   b(new) = b(old) + y = b(old) + p
            self.b += p

        # A neuron must not be connected to itself (would always self-reinforce),
        # so we zero the main diagonal of W.
        np.fill_diagonal(self.W, 0)

    def recall(self, x: np.ndarray, max_iter: int = 20) -> np.ndarray:
        """
        Recover the stored pattern closest to x.

        Each step updates every neuron:
            net_j = sum_i( W[j,i] * x[i] ) + b[j]   (weighted vote + bias)
            x_j   = sign(net_j)                       (+1 if net > 0, −1 otherwise)

        This is repeated until nothing changes (settled) or max_iter is reached.
        """
        x = np.array(x, dtype=np.float32).copy()

        for _ in range(max_iter):
            # Compute the net input for every neuron in one matrix multiply
            net = self.W @ x + self.b

            # Fire +1 if net > 0, −1 if net < 0; treat exact 0 as +1
            x_new = np.sign(net)
            x_new[x_new == 0] = 1.0

            # Stop early if the pattern has settled (no neuron changed its state)
            if np.array_equal(x_new, x):
                break

            x = x_new

        return x


class Perceptron:
    """
    Multi-class Perceptron classifier.

    One weight vector per class. The winning class is whichever scores highest:
        score_k = W[k] · x + b[k]   (dot product of class-k weights with input)
        prediction = argmax(scores)

    Training rule — only fires when the prediction is WRONG:
        Step 0: W = 0,  b = 0
        Step 1: repeat for many epochs:
        Step 2:   for each sample (x, true_class):
        Step 3:     pred = argmax( W · x + b )      (current best guess)
        Step 4:     if pred != true_class:           (mistake → update)
                      W[true_class] += lr * x        (reward the correct class)
                      b[true_class] += lr
                      W[pred]       -= lr * x        (penalise the wrong class)
                      b[pred]       -= lr

    No update happens on correct predictions — the network only learns from mistakes.
    """

    def __init__(self, n_inputs: int, n_classes: int):
        # Step 0: all weights and biases start at zero
        # W shape: (n_classes, n_inputs) — one weight row per class
        self.W = np.zeros((n_classes, n_inputs), dtype=np.float32)
        self.b = np.zeros(n_classes,             dtype=np.float32)

    def train(self, X: np.ndarray, y: list,
              epochs: int = 200, lr: float = 0.1):
        """
        X:      (N, n_inputs) array of training patterns (values 0 or 1).
        y:      list of N integer class indices (0 … n_classes-1).
        epochs: how many full passes over the data.
        lr:     learning rate — how big each correction step is.
        """
        X = np.array(X, dtype=np.float32)
        y = np.array(y, dtype=np.int32)

        for _ in range(epochs):
            for xi, yi in zip(X, y):
                # Step 3: score every class and pick the highest
                scores = self.W @ xi + self.b   # shape (n_classes,)
                pred   = int(np.argmax(scores)) # index of the winning class

                # Step 4: only update if the prediction was wrong
                if pred != yi:
                    # The correct class lost — push its weights toward xi
                    self.W[yi] += lr * xi
                    self.b[yi] += lr

                    # The wrong class won — push its weights away from xi
                    self.W[pred] -= lr * xi
                    self.b[pred] -= lr

    def predict(self, x: np.ndarray) -> int:
        # Score every class and return the index of the highest scorer
        scores = self.W @ np.array(x, dtype=np.float32) + self.b
        return int(np.argmax(scores))


class AdalineClassifier:
    """
    Multi-class Adaline classifier (Widrow-Hoff / LMS / delta rule).

    Like the Perceptron, but crucially different in one way:
    the Perceptron only updates on mistakes; Adaline updates on EVERY sample,
    minimising the continuous error between the raw output and the target.
    This gives smoother, better-conditioned decision boundaries.

    Each class k has its own weight vector W[k] and bias b[k].
    The raw output for class k is:   out_k = W[k] · x + b[k]
    The target for class k is:       t_k   = 1  if k == true_class, else 0

    Training rule (delta / LMS rule) — applied on every sample, every epoch:
        Step 0: W ≈ 0,  b = 0
        Step 1: repeat for many epochs:
        Step 2:   for each sample (x, true_class):
        Step 3:     out  = W · x + b                    (raw scores, all classes)
        Step 4:     t    = one-hot vector for true_class (1 at index, 0 elsewhere)
                    err  = t - out                       (how far off each class is)
                    W   += lr * outer(err, x)            (nudge weights toward less error)
                    b   += lr * err

    Unlike the Perceptron, updates happen even when correct — the error is
    always driven toward zero, not just flipped from wrong to right.
    """

    def __init__(self, n_inputs: int, n_classes: int):
        # Step 0: tiny random weights (not zero — avoids all classes updating identically)
        rng = np.random.default_rng(42)
        self.W = rng.normal(0, 0.01, (n_classes, n_inputs)).astype(np.float32)
        self.b = np.zeros(n_classes, dtype=np.float32)

    def train(self, X: np.ndarray, y: list,
              epochs: int = 300, lr: float = 0.001):
        """
        X:      (N, n_inputs) array of training patterns (values 0 or 1).
        y:      list of N integer class indices (0 … n_classes-1).
        epochs: how many full passes over the data.
        lr:     learning rate — step size for each weight correction.
        """
        X     = np.array(X, dtype=np.float32)
        y     = np.array(y, dtype=np.int32)
        n_cls = self.W.shape[0]

        # Build one-hot target matrix: T[i, k] = 1 if sample i belongs to class k
        # Example for 3 classes: class 1 → [0, 1, 0]
        T = np.zeros((len(y), n_cls), dtype=np.float32)
        for i, yi in enumerate(y):
            T[i, yi] = 1.0

        for _ in range(epochs):
            for xi, ti in zip(X, T):
                # Step 3: compute raw output scores for every class
                out = self.W @ xi + self.b   # shape (n_cls,)

                # Step 4: error = target − output  (positive → output too low)
                err = ti - out               # shape (n_cls,)

                # Adjust every weight by how much it contributed to the error:
                #   W[k, i] += lr * err[k] * xi   for all k, i at once
                self.W += lr * np.outer(err, xi)
                self.b += lr * err

    def predict(self, x: np.ndarray):
        """
        Return (predicted_class_index, confidence).
        Confidence is the softmax of the raw scores — a probability between 0 and 1.
        """
        scores = self.W @ np.array(x, dtype=np.float32) + self.b

        # Winning class = highest raw score
        pred = int(np.argmax(scores))

        # Softmax turns raw scores into a probability distribution
        # Subtract max first for numerical stability (avoids huge exp values)
        exp_s = np.exp(scores - scores.max())
        conf  = float(exp_s[pred] / exp_s.sum())

        return pred, conf
