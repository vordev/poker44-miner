"""Reward and scoring utilities for Poker44 poker bot detection."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score


def _recall_at_fpr(
    y_score: np.ndarray,
    y_true: np.ndarray,
    *,
    max_fpr: float = 0.05,
) -> tuple[float, float]:
    """Best bot recall reachable while keeping human false-positive rate bounded."""
    labels = np.asarray(y_true, dtype=int)
    scores = np.asarray(y_score, dtype=float)
    positive_count = int(np.sum(labels == 1))
    negative_count = int(np.sum(labels == 0))
    if positive_count <= 0 or negative_count <= 0 or scores.size == 0:
        return 0.0, 0.0

    order = np.argsort(-scores, kind="mergesort")
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels == 1)
    fp = np.cumsum(sorted_labels == 0)
    recall = tp / max(positive_count, 1)
    fpr = fp / max(negative_count, 1)

    allowed = fpr <= float(max_fpr)
    if not np.any(allowed):
        return 0.0, 0.0

    allowed_indices = np.flatnonzero(allowed)
    best_local = int(allowed_indices[np.argmax(recall[allowed])])
    return float(recall[best_local]), float(fpr[best_local])


def reward(y_pred: np.ndarray, y_true: np.ndarray) -> tuple[float, dict]:
    """
    Compute a rank-first reward that protects humans without rewarding top-k guessing.
    """
    if y_pred.size and np.any(y_true == 1):
        ap_score = average_precision_score(y_true, y_pred)
    else:
        ap_score = 0.0

    bot_recall, fpr = _recall_at_fpr(y_pred, y_true, max_fpr=0.05)
    human_safety_penalty = 1.0

    base_score = 0.75 * ap_score + 0.25 * bot_recall
    rew = base_score * human_safety_penalty

    res = {
        "fpr": fpr,
        "bot_recall": bot_recall,
        "ap_score": ap_score,
        "human_safety_penalty": human_safety_penalty,
        "base_score": base_score,
        "reward": rew,
    }
    return rew, res
