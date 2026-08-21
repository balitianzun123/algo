"""Minimal threshold helpers required by algorithm_core.thresholds."""

from __future__ import annotations

from typing import Any, Iterable, Sequence

import numpy as np


def threshold_candidates(scores: Sequence[float]) -> np.ndarray:
    values = np.asarray(scores, dtype=np.float64)
    if values.ndim != 1 or values.size == 0 or not np.isfinite(values).all():
        raise ValueError("calibration scores must be a nonempty finite one-dimensional array.")
    unique = np.unique(values)
    candidates = [float(np.nextafter(unique[0], -np.inf)), 0.0]
    if unique.size > 1:
        candidates.extend((unique[:-1] + (unique[1:] - unique[:-1]) / 2.0).tolist())
    candidates.append(float(np.nextafter(unique[-1], np.inf)))
    return np.asarray(sorted(set(float(value) for value in candidates)), dtype=np.float64)


def select_threshold_candidate(rows: Iterable[dict[str, Any]]) -> dict[str, Any]:
    candidates = list(rows)
    if not candidates:
        raise ValueError("threshold candidate rows must not be empty.")
    return min(
        candidates,
        key=lambda row: (
            -float(row["balanced_accuracy"]),
            -float(row["worst_class_recall"]),
            abs(float(row["threshold"])),
            float(row["threshold"]),
        ),
    )
