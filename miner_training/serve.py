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

import os
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
        self._idx: List[int] = list(range(len(FEATURE_NAMES)))
        # Experiment flag: if the model ranks backwards on live (AP<0.5), inverting
        # the output (1 - p) corrects it. Reversible via env, no retrain needed.
        self.invert = os.getenv("POKER44_INVERT_SCORES", "").strip().lower() in {"1", "true", "yes", "on"}
        path = Path(model_path)
        if path.exists():
            with path.open("rb") as fh:
                blob = pickle.load(fh)
            self.model = blob["model"]
            self.feature_names = blob.get("feature_names", FEATURE_NAMES)
            # A drift-selected model uses only a SUBSET of features; map its feature
            # names to positions in the current full feature vector.
            try:
                self._idx = [FEATURE_NAMES.index(n) for n in self.feature_names]
            except ValueError as exc:
                raise ValueError(
                    f"Saved model expects a feature not in the current features.py ({exc}). "
                    "Retrain, or pin features.py to the trained version."
                ) from exc

    def _vec(self, group: List[dict]) -> List[float]:
        full = extract_features(group)
        return [full[i] for i in self._idx]

    def _finalize(self, proba: float) -> float:
        p = (1.0 - proba) if self.invert else proba
        return round(float(min(1.0, max(0.0, p))), 6)

    def score_chunk(self, group: List[dict]) -> float:
        if self.model is None:
            return 0.5
        return self._finalize(self.model.predict_proba([self._vec(group)])[0][1])

    def score_chunks(self, chunks: List[List[dict]]) -> List[float]:
        """One score per chunk; length always matches len(chunks)."""
        if self.model is None:
            return [0.5 for _ in chunks]
        if not chunks:
            return []
        matrix = [self._vec(g) for g in chunks]
        proba = self.model.predict_proba(matrix)[:, 1]
        return [self._finalize(p) for p in proba]


_DEFAULT: Optional[ModelScorer] = None


def score_chunks(chunks: List[List[dict]], model_path: str = "miner_training/model.pkl") -> List[float]:
    """Convenience singleton entry point."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ModelScorer(model_path)
    return _DEFAULT.score_chunks(chunks)
