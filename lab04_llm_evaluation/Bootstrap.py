import numpy as np
from sklearn.metrics import f1_score

def bootstrap_f1(y_true, y_pred, n=1000, seed=42):
    rng = np.random.default_rng(seed)
    scores = []
    n_samples = len(y_true)
    for _ in range(n):
        idx = rng.integers(0, n_samples, n_samples)
        scores.append(f1_score(
            [y_true[i] for i in idx],
            [y_pred[i] for i in idx],
            pos_label="suspicious", zero_division=0,
        ))
    return np.percentile(scores, [2.5, 97.5])

# Пример: [0.83, 0.97] для Model B