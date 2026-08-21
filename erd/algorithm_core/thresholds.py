"""Threshold selection for the fixed-rule hierarchical algorithm."""

# 计算和选择阈值

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

import numpy as np

from algo.erd.regulatory.left_right_erd import compute_left_right_metrics
from algo.erd.regulatory.threshold_calibration import select_threshold_candidate as _select_laterality_candidate
from algo.erd.regulatory.threshold_calibration import threshold_candidates
from .constants import LEFT_ID, REST_ID, RIGHT_ID
from .criterion import (
    MI_SUCCESS_CRITERION_NAME,
    MI_SUCCESS_CRITERION_VERSION,
    get_mi_success_drop_threshold_pct,
)
from .exceptions import NonFiniteDataError


def predict_rest_action_from_scores(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Return 0 for rest and 1 for action using the fixed Stage1 direction."""
    return np.where(np.asarray(scores, dtype=np.float64) >= float(threshold), 1, 0).astype(np.int64)


def predict_left_right_from_scores(scores: Sequence[float], threshold: float) -> np.ndarray:
    """Return 1 for left and 2 for right using the fixed laterality direction."""
    return np.where(np.asarray(scores, dtype=np.float64) >= float(threshold), LEFT_ID, RIGHT_ID).astype(np.int64)


def action_metric_row(y_true_binary: Sequence[int], y_pred_binary: Sequence[int]) -> dict[str, Any]:
    true = np.asarray(y_true_binary, dtype=np.int64)
    pred = np.asarray(y_pred_binary, dtype=np.int64)
    tn = int(np.sum((true == 0) & (pred == 0)))
    fp = int(np.sum((true == 0) & (pred == 1)))
    fn = int(np.sum((true == 1) & (pred == 0)))
    tp = int(np.sum((true == 1) & (pred == 1)))

    def div(num: float, den: float) -> float:
        return float(num) / float(den) if float(den) else float("nan")

    rest_recall = div(tn, tn + fp)
    action_recall = div(tp, tp + fn)
    rest_precision = div(tn, tn + fn)
    action_precision = div(tp, tp + fp)
    f1_rest = div(2 * rest_precision * rest_recall, rest_precision + rest_recall)
    f1_action = div(2 * action_precision * action_recall, action_precision + action_recall)
    total = int(true.size)
    return {
        "tn_rest": tn,
        "fp_rest_to_action": fp,
        "fn_action_to_rest": fn,
        "tp_action": tp,
        "total": total,
        "accuracy": div(tn + tp, total),
        "balanced_accuracy": float(np.nanmean([rest_recall, action_recall])),
        "macro_f1": float(np.nanmean([f1_rest, f1_action])),
        "rest_recall": rest_recall,
        "action_recall": action_recall,
        "rest_false_trigger_rate": div(fp, tn + fp),
        "action_miss_rate": div(fn, tp + fn),
        "confusion_matrix": f"[[{tn},{fp}],[{fn},{tp}]]",
    }


def action_binary_labels(labels: Sequence[int]) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    return np.where(y == REST_ID, 0, 1).astype(np.int64)


def select_cal10_action_threshold(scores: Sequence[float], y_true_binary: Sequence[int]) -> dict[str, Any]:
    """Select Cal-10 Stage1 threshold using the existing fixed ordering."""
    values = np.asarray(scores, dtype=np.float64)
    y = np.asarray(y_true_binary, dtype=np.int64)
    if values.shape != y.shape or values.ndim != 1 or set(np.unique(y).tolist()) != {0, 1}:
        raise NonFiniteDataError("Cal-10 action threshold needs aligned rest/action labels.")
    if not np.isfinite(values).all():
        raise NonFiniteDataError("Cal-10 action scores must be finite.")
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(values):
        pred = predict_rest_action_from_scores(values, float(threshold))
        rows.append({"threshold": float(threshold), **action_metric_row(y, pred)})
    return max(
        rows,
        key=lambda row: (
            float(row["balanced_accuracy"]),
            min(float(row["rest_recall"]), float(row["action_recall"])),
            -float(row["rest_false_trigger_rate"]),
            -abs(float(row["threshold"])),
            float(row["threshold"]),
        ),
    )


def select_cal10_laterality_threshold(scores: Sequence[float], labels: Sequence[int]) -> dict[str, Any]:
    """Select Cal-10 Stage2 threshold using the existing laterality ordering."""
    values = np.asarray(scores, dtype=np.float64)
    data_y = np.asarray(labels, dtype=np.int64)
    if values.shape != data_y.shape or values.ndim != 1 or set(np.unique(data_y).tolist()) != {LEFT_ID, RIGHT_ID}:
        raise NonFiniteDataError("Cal-10 laterality threshold needs aligned left/right labels.")
    if not np.isfinite(values).all():
        raise NonFiniteDataError("Cal-10 laterality scores must be finite.")
    binary_y = np.where(data_y == LEFT_ID, 0, 1).astype(np.int64)
    rows: list[dict[str, Any]] = []
    for threshold in threshold_candidates(values):
        pred_binary = np.where(values >= float(threshold), 0, 1).astype(np.int64)
        metrics = compute_left_right_metrics(binary_y, pred_binary, right_hand_score=-values)
        rows.append({"threshold": float(threshold), **metrics})
    return _select_laterality_candidate(rows)

def _validated_rest_action_scores(loo_rest_action_scores: Sequence[float]) -> np.ndarray:
    scores = np.asarray(loo_rest_action_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size == 0 or not np.isfinite(scores).all():
        raise NonFiniteDataError("Rest action threshold needs finite leave-one-rest-out scores.")
    return scores


def rest_action_threshold_details(
    loo_rest_action_scores: Sequence[float],
    *,
    threshold: float | None = None,
) -> dict[str, Any]:
    """Deprecated Rest-N threshold shell; formal calibration no longer uses Rest scores."""
    scores = _validated_rest_action_scores(loo_rest_action_scores)
    criterion_threshold = get_mi_success_drop_threshold_pct()
    actual_threshold = criterion_threshold if threshold is None else float(threshold)
    if not np.isfinite(actual_threshold):
        raise NonFiniteDataError("Rest action threshold must be finite.")
    return {
        "original_scores": [float(score) for score in scores],
        "sorted_scores": [float(score) for score in np.sort(scores)],
        "method": "criterion_function",
        "criterion_function": MI_SUCCESS_CRITERION_NAME,
        "criterion_function_version": MI_SUCCESS_CRITERION_VERSION,
        "n": int(scores.size),
        "final_threshold": criterion_threshold,
        "threshold": actual_threshold,
        "matches_threshold": bool(np.isclose(criterion_threshold, actual_threshold)),
        "deprecated_input": "loo_rest_action_scores",
        "note": "Formal Rest-N calibration does not derive MI threshold from Rest scores.",
    }


def select_rest_action_threshold(loo_rest_action_scores: Sequence[float]) -> float:
    """Compatibility wrapper returning the current fixed MI Power Drop criterion."""
    details = rest_action_threshold_details(loo_rest_action_scores)
    return float(details["final_threshold"])


def select_rest10_action_threshold(loo_rest_action_scores: Sequence[float]) -> float:
    """Compatibility wrapper retaining Rest-10's ten-score contract."""
    scores = np.asarray(loo_rest_action_scores, dtype=np.float64)
    if scores.ndim != 1 or scores.size != 10 or not np.isfinite(scores).all():
        raise NonFiniteDataError("Rest-10 action threshold needs ten finite leave-one-rest-out scores.")
    return select_rest_action_threshold(scores)


def select_threshold_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    """Expose the existing laterality threshold candidate selector."""
    return _select_laterality_candidate(rows)
