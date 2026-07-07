"""Load a trained model and score chunks inside the live miner.

Import the SAME feature module the trainer used, so serve-time features match
train-time features exactly. Drop-in usage in neurons/miner.py:

    from miner_training.serve import ModelScorer
    self.scorer = ModelScorer("miner_training/model.pkl")  # in __init__

    async def forward(self, synapse):
        scores = self.scorer.score_chunks(synapse.chunks or [])
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        return synapse

If the model file is missing/unreadable, score_chunks falls back to 0.5 so the
miner still returns a valid, length-matched response instead of crashing.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import List, Optional

from miner_training.features import FEATURE_NAMES, extract_features
# Imported so the saved ensemble unpickles (its class must be importable here).
from miner_training.ensemble import RecencyEnsemble  # noqa: F401


class ModelScorer:
    def __init__(self, model_path: str = "miner_training/model.pkl") -> None:
        self.model = None
        self.feature_names: List[str] = FEATURE_NAMES
        path = Path(model_path)
        if path.exists():
            with path.open("rb") as fh:
                blob = pickle.load(fh)
            self.model = blob["model"]
            self.feature_names = blob.get("feature_names", FEATURE_NAMES)
            if self.feature_names != FEATURE_NAMES:
                raise ValueError(
                    "Feature schema drift: the saved model's feature_names do not match "
                    "the current features.py. Retrain, or pin features.py to the trained version."
                )

    def score_chunk(self, group: List[dict]) -> float:
        if self.model is None:
            return 0.5
        vec = extract_features(group)
        proba = self.model.predict_proba([vec])[0][1]
        return round(float(min(1.0, max(0.0, proba))), 6)

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        """One score per chunk; length always matches len(chunks)."""
        if self.model is None:
            return [0.5 for _ in chunks]
        if not chunks:
            return []
        matrix = [extract_features(g) for g in chunks]
        proba = self.model.predict_proba(matrix)[:, 1]
        return [round(float(min(1.0, max(0.0, p))), 6) for p in proba]


_DEFAULT: Optional[ModelScorer] = None


def score_chunks(chunks: List[List[dict]], model_path: str = "miner_training/model.pkl") -> List[float]:
    """Convenience singleton entry point."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ModelScorer(model_path)
    return _DEFAULT.score_chunks(chunks)
