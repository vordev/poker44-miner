"""Utility helpers for Poker44 neurons."""

from __future__ import annotations

import time
import bittensor as bt


def ttl_get_block(neuron) -> int:
    """
    Return the latest known block height, falling back gracefully if subtensor
    is temporarily unavailable.
    """
    try:
        return int(neuron.subtensor.get_current_block())
    except Exception:
        try:
            return int(neuron.metagraph.block.item())
        except Exception:
            # Last resort: monotonic placeholder so epoch logic still advances.
            return int(time.time())
