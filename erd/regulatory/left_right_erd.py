"""Minimal left/right metric helper required by algorithm_core.thresholds."""

from __future__ import annotations

import numpy as np


def compute_left_right_metrics(y_true_binary, y_pred_binary, *, right_hand_score=None):
    """Compute binary left/right metrics with 0=left, 1=right."""

    true = np.asarray(y_true_binary, dtype=np.int64)
    pred = np.asarray(y_pred_binary, dtype=np.int64)
    if true.ndim != 1 or pred.ndim != 1 or true.shape != pred.shape:
        raise ValueError("Binary y_true/y_pred must be aligned one-dimensional arrays.")
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    tp = int(np.sum((true == 1) & (pred == 1)))
    total = int(true.size)

    def div(num, den):
        return float(num) / float(den) if float(den) else float("nan")

    left_recall = div(tn, tn + fp)
    right_recall = div(tp, tp + fn)
    left_precision = div(tn, tn + fn)
    right_precision = div(tp, tp + fp)
    f1_left = div(2 * tn, 2 * tn + fp + fn)
    f1_right = div(2 * tp, 2 * tp + fp + fn)
    roc_auc = float("nan")
    if right_hand_score is not None and np.unique(true).size == 2:
        try:
            from sklearn.metrics import roc_auc_score

            roc_auc = float(roc_auc_score(true, np.asarray(right_hand_score, dtype=np.float64)))
        except Exception:
            roc_auc = float("nan")
    return {
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
        "total": total,
        "accuracy": div(tn + tp, total),
        "balanced_accuracy": float(np.nanmean([left_recall, right_recall])),
        "f1_macro": float(np.nanmean([f1_left, f1_right])),
        "f1_weighted": div(f1_left * (tn + fp) + f1_right * (tp + fn), total),
        "precision_macro": float(np.nanmean([left_precision, right_precision])),
        "recall_macro": float(np.nanmean([left_recall, right_recall])),
        "left_recall": left_recall,
        "right_recall": right_recall,
        "worst_class_recall": float(np.nanmin([left_recall, right_recall])),
        "roc_auc": roc_auc,
        "confusion_matrix": f"[[{tn},{fp}],[{fn},{tp}]]",
    }
