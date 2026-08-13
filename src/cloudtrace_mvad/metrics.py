from __future__ import annotations

from itertools import product

import numpy as np
from sklearn.metrics import (
    average_precision_score, roc_auc_score, precision_score, recall_score, f1_score,
    confusion_matrix, matthews_corrcoef,
)


def binary_metrics(y_true: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, float]:
    y = np.asarray(y_true, dtype=int)
    s = np.asarray(scores, dtype=float)
    pred = (s >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred, labels=[0, 1]).ravel()
    return {
        "ROC_AUC": float(roc_auc_score(y, s)),
        "PR_AUC": float(average_precision_score(y, s)),
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "F1": float(f1_score(y, pred, zero_division=0)),
        "Specificity": float(tn / max(tn + fp, 1)),
        "FPR": float(fp / max(tn + fp, 1)),
        "FNR": float(fn / max(fn + tp, 1)),
        "MCC": float(matthews_corrcoef(y, pred)),
    }


def summarize_runs(frame, model_col: str = "Model"):
    numeric = [c for c in frame.columns if c not in {model_col, "Seed"}]
    rows = []
    for model, group in frame.groupby(model_col, sort=False):
        row = {model_col: model, "Runs": len(group)}
        for metric in numeric:
            vals = group[metric].astype(float)
            row[f"{metric}_mean"] = vals.mean()
            row[f"{metric}_sd"] = vals.std(ddof=1)
            row[metric] = f"{vals.mean():.4f} ± {vals.std(ddof=1):.4f}"
        rows.append(row)
    import pandas as pd
    return pd.DataFrame(rows)


def exact_sign_flip_test(differences: np.ndarray) -> tuple[float, float]:
    d = np.asarray(differences, dtype=float)
    observed = abs(float(d.mean()))
    permuted = []
    for signs in product([-1.0, 1.0], repeat=len(d)):
        permuted.append(abs(float(np.mean(d * np.array(signs)))))
    count = sum(x >= observed - 1e-15 for x in permuted)
    p = (count + 1) / (len(permuted) + 1)
    sd = d.std(ddof=1)
    effect = float(d.mean() / sd) if sd > 0 else float("inf")
    return effect, float(p)
