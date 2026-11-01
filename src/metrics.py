"""Per-state evaluation metrics, in the shape the report table requires.

For each activity, one-vs-rest from the multiclass confusion matrix:
    sensitivity (recall) = TP / (TP + FN)
    specificity          = TN / (TN + FP)
    per-class accuracy   = (TP + TN) / total
plus a single multiclass accuracy and macro-F1 reported alongside.

The report's "Overall Accuracy" column is the per-class one-vs-rest accuracy.
A single multiclass accuracy is also reported above the table so the two
cannot be confused - they are different numbers and both are legitimate.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def confusion(y_true: np.ndarray, y_pred: np.ndarray, n_states: int) -> np.ndarray:
    """Counts, shape (n_states, n_states), rows = true, cols = predicted."""
    y_true = np.asarray(y_true, int).ravel()
    y_pred = np.asarray(y_pred, int).ravel()
    if y_true.shape != y_pred.shape:
        raise ValueError("y_true and y_pred must be the same length")

    cm = np.zeros((n_states, n_states), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm


def per_state_report(y_true: np.ndarray, y_pred: np.ndarray,
                     labels: list[str]) -> pd.DataFrame:
    """The evaluation table: one row per activity.

    Columns match the report template - number of samples, sensitivity,
    specificity, accuracy - with precision and F1 added since they cost
    nothing and make the discussion of the confusions easier to write.
    """
    n = len(labels)
    cm = confusion(y_true, y_pred, n)
    total = cm.sum()

    rows = []
    for k, name in enumerate(labels):
        tp = int(cm[k, k])
        fn = int(cm[k, :].sum() - tp)
        fp = int(cm[:, k].sum() - tp)
        tn = int(total - tp - fn - fp)

        sens = tp / (tp + fn) if (tp + fn) else float("nan")
        spec = tn / (tn + fp) if (tn + fp) else float("nan")
        prec = tp / (tp + fp) if (tp + fp) else float("nan")
        f1 = (2 * prec * sens / (prec + sens)
              if prec and sens and not np.isnan(prec * sens) else 0.0)

        rows.append({
            "State (Activity)": name,
            "Number of Samples": tp + fn,
            "Sensitivity": sens,
            "Specificity": spec,
            "Precision": prec,
            "F1": f1,
            "Overall Accuracy": (tp + tn) / total,
            "TP": tp, "FN": fn, "FP": fp, "TN": tn,
        })
    return pd.DataFrame(rows)


def overall(y_true: np.ndarray, y_pred: np.ndarray,
            n_states: int | None = None) -> dict:
    """Multiclass accuracy and macro-averaged F1 across all windows.

    `n_states` should be passed explicitly whenever it is known. Inferring it
    from the observed labels is only safe if every state actually appears: if
    a state is absent from both y_true and y_pred the inferred count shrinks,
    that state silently drops out of the macro average, and the macro figures
    are then computed over a different number of classes than the per-state
    table reports.
    """
    y_true = np.asarray(y_true, int).ravel()
    y_pred = np.asarray(y_pred, int).ravel()
    if n_states is None:
        n_states = int(max(y_true.max(), y_pred.max()) + 1)

    report = per_state_report(y_true, y_pred, [str(k) for k in range(n_states)])
    return {
        "n_windows": int(len(y_true)),
        "accuracy": float((y_true == y_pred).mean()),
        "macro_f1": float(report["F1"].mean()),
        "macro_sensitivity": float(report["Sensitivity"].mean()),
        "macro_specificity": float(report["Specificity"].mean()),
    }


def format_report(report: pd.DataFrame, decimals: int = 4) -> pd.DataFrame:
    """Round the rate columns for presentation, dropping the raw counts."""
    cols = ["State (Activity)", "Number of Samples", "Sensitivity",
            "Specificity", "Overall Accuracy"]
    out = report[cols].copy()
    for c in ("Sensitivity", "Specificity", "Overall Accuracy"):
        out[c] = out[c].round(decimals)
    return out
