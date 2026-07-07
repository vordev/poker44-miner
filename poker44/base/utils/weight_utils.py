"""Minimal weight processing utilities used by the base neuron classes."""

from __future__ import annotations

from typing import Iterable, Tuple

import numpy as np


def process_weights_for_netuid(
    uids: Iterable[int],
    weights: np.ndarray,
    netuid: int,
    subtensor,
    metagraph,
) -> Tuple[np.ndarray, np.ndarray]:
    """Clamp and normalize weights for submission."""

    weights = np.nan_to_num(np.asarray(weights, dtype=np.float32), nan=0.0)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total > 0:
        weights = weights / total
    else:
        weights = np.zeros_like(weights)

    return np.asarray(list(uids), dtype=np.int64), weights


def convert_weights_and_uids_for_emit(
    uids: np.ndarray, weights: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Convert weights to uint16 representation expected by bittensor."""

    weights = np.nan_to_num(weights, nan=0.0)
    weights = np.maximum(weights, 0.0)
    total = float(weights.sum())
    if total > 0:
        weights = weights / total

    uint_weights = (weights * 65535).astype(np.uint16)
    uint_uids = np.asarray(uids, dtype=np.uint16)
    return uint_uids, uint_weights
