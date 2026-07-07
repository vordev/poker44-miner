"""Reference Poker44 miner with simple chunk-level behavioral heuristics."""

# from __future__ import annotations

import sys
import time
from collections import Counter
from pathlib import Path
from typing import Tuple

import bittensor as bt

# Make the repo root importable so `miner_training` resolves when the miner is
# launched as a script (run_miner.sh also exports PYTHONPATH to the repo root).
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from poker44.base.miner import BaseMinerNeuron
from poker44.utils.model_manifest import (
    build_local_model_manifest,
    evaluate_manifest_compliance,
    manifest_digest,
)
from poker44.validator.synapse import DetectionSynapse


class Miner(BaseMinerNeuron):
    """
    Reference heuristic miner.

    It aggregates simple behavior signals over each chunk and returns a bot-risk
    score per chunk. The goal is not SOTA accuracy, but a deterministic and
    explainable baseline that is meaningfully better than random.
    """

    def __init__(self, config=None):
        super(Miner, self).__init__(config=config)
        bt.logging.info("🤖 Poker44 Miner started")
        repo_root = Path(__file__).resolve().parents[1]

        # Load the trained model (served live). Falls back to the built-in heuristic
        # if the model or its dependencies are missing, so the miner always answers.
        self.scorer = None
        model_path = repo_root / "miner_training" / "model.pkl"
        try:
            from miner_training.serve import ModelScorer

            self.scorer = ModelScorer(str(model_path))
            if self.scorer.model is None:
                bt.logging.warning(
                    f"No trained model at {model_path}; serving heuristic fallback. "
                    "Run `python -m miner_training.train --all` to create it."
                )
            else:
                bt.logging.info(f"Loaded trained model: {model_path}")
        except Exception as exc:  # missing package / lightgbm / corrupt pickle
            bt.logging.warning(f"Model load failed ({exc}); serving heuristic fallback.")

        using_model = self.scorer is not None and self.scorer.model is not None

        # The manifest's implementation_sha256 must cover the code actually serving
        # predictions, so include the model pipeline when the trained model is live.
        implementation_files = [Path(__file__).resolve()]
        if using_model:
            for rel in (
                "miner_training/features.py",
                "miner_training/ensemble.py",
                "miner_training/serve.py",
                "miner_training/model.pkl",
            ):
                candidate = repo_root / rel
                if candidate.exists():
                    implementation_files.append(candidate.resolve())

        self.model_manifest = build_local_model_manifest(
            repo_root=repo_root,
            implementation_files=implementation_files,
            defaults={
                # Set POKER44_MODEL_REPO_URL and POKER44_MODEL_REPO_COMMIT (env) to YOUR
                # public model repo + real commit hash for `transparent` compliance.
                "model_name": "poker44-recency-ensemble" if using_model else "poker44-reference-heuristic",
                "model_version": "2" if using_model else "1",
                "framework": "lightgbm+scikit-learn" if using_model else "python-heuristic",
                "license": "MIT",
                "repo_url": "https://github.com/Poker44/Poker44-subnet",
                "notes": (
                    "Recency-weighted ensemble over hero-behavior features."
                    if using_model
                    else "Reference heuristic miner shipped with the Poker44 subnet."
                ),
                "open_source": True,
                "inference_mode": "remote",
                "training_data_statement": (
                    "Trained only on the public Poker44 training benchmark (api.poker44.net); "
                    "one label per chunk group."
                    if using_model
                    else "Reference heuristic miner. No training step. Uses only runtime chunk features."
                ),
                "training_data_sources": (
                    ["poker44-public-benchmark"] if using_model else ["none"]
                ),
                "private_data_attestation": (
                    "This miner does not train on validator-only evaluation data."
                ),
            },
        )
        self.manifest_compliance = evaluate_manifest_compliance(self.model_manifest)
        self.manifest_digest = manifest_digest(self.model_manifest)
        self._log_manifest_startup(repo_root)
        
        # # Attach handlers after initialization
        # self.axon.attach(
        #     forward_fn = self.forward,
        #     blacklist_fn = self.blacklist,
        #     priority_fn = self.priority,
        # )
        # bt.logging.info("Attaching forward function to miner axon.")
        
        bt.logging.info(f"Axon created: {self.axon}")

    def _log_manifest_startup(self, repo_root: Path) -> None:
        bt.logging.info("Open-sourced miner manifest standard active for this miner.")
        bt.logging.info(
            f"Miner transparency status: {self.manifest_compliance['status']} "
            f"(missing_fields={self.manifest_compliance['missing_fields']})"
        )
        bt.logging.info(
            f"Manifest summary | model={self.model_manifest.get('model_name', '')} "
            f"version={self.model_manifest.get('model_version', '')} "
            f"repo={self.model_manifest.get('repo_url', '')} "
            f"commit={self.model_manifest.get('repo_commit', '')} "
            f"open_source={self.model_manifest.get('open_source')}"
        )
        bt.logging.info(
            f"Manifest digest={self.manifest_digest} "
            f"inference_mode={self.model_manifest.get('inference_mode', '')}"
        )
        bt.logging.info(
            "Miner prep docs available | "
            f"miner_doc={repo_root / 'docs' / 'miner.md'}"
        )

    async def forward(self, synapse: DetectionSynapse) -> DetectionSynapse:
        """Assign one bot-risk score per chunk (trained model, heuristic fallback)."""
        chunks = synapse.chunks or []
        if self.scorer is not None and self.scorer.model is not None:
            scores = self.scorer.score_chunks(chunks)
            source = "trained model"
        else:
            scores = [self.score_chunk(chunk) for chunk in chunks]
            source = "heuristic"
        synapse.risk_scores = scores
        synapse.predictions = [s >= 0.5 for s in scores]
        synapse.model_manifest = dict(self.model_manifest)
        # Optional, fail-safe: capture the unlabeled live queries for drift analysis
        # (no-op unless POKER44_CAPTURE=1). Never affects the response.
        try:
            from miner_training.live_capture import capture_chunks
            capture_chunks(chunks, scores)
        except Exception:
            pass
        bt.logging.info(f"Scored {len(chunks)} chunks with {source}.")
        return synapse

    @staticmethod
    def _clamp01(value: float) -> float:
        return max(0.0, min(1.0, value))

    @classmethod
    def _score_hand(cls, hand: dict) -> float:
        actions = hand.get("actions") or []
        players = hand.get("players") or []
        streets = hand.get("streets") or []
        outcome = hand.get("outcome") or {}

        action_counts = Counter(action.get("action_type") for action in actions)
        meaningful_actions = max(
            1,
            sum(
                action_counts.get(kind, 0)
                for kind in ("call", "check", "bet", "raise", "fold")
            ),
        )

        call_ratio = action_counts.get("call", 0) / meaningful_actions
        check_ratio = action_counts.get("check", 0) / meaningful_actions
        fold_ratio = action_counts.get("fold", 0) / meaningful_actions
        raise_ratio = action_counts.get("raise", 0) / meaningful_actions
        street_depth = len(streets) / 3.0
        showdown_flag = 1.0 if outcome.get("showdown") else 0.0

        player_count_signal = 0.0
        if players:
            player_count_signal = (6 - min(len(players), 6)) / 4.0

        score = 0.0
        score += 0.32 * street_depth
        score += 0.22 * showdown_flag
        score += 0.18 * cls._clamp01(call_ratio / 0.35)
        score += 0.12 * cls._clamp01(check_ratio / 0.30)
        score += 0.08 * cls._clamp01(player_count_signal)
        score -= 0.18 * cls._clamp01(fold_ratio / 0.55)
        score -= 0.10 * cls._clamp01(raise_ratio / 0.20)

        return cls._clamp01(score)

    @classmethod
    def score_chunk(cls, chunk: list[dict]) -> float:
        if not chunk:
            return 0.5

        hand_scores = [cls._score_hand(hand) for hand in chunk]
        avg_score = sum(hand_scores) / len(hand_scores)

        return round(cls._clamp01(avg_score), 6)

    async def blacklist(self, synapse: DetectionSynapse) -> Tuple[bool, str]:
        """Determine whether to blacklist incoming requests."""
        return self.common_blacklist(synapse)

    async def priority(self, synapse: DetectionSynapse) -> float:
        """Assign priority based on caller's stake."""
        return self.caller_priority(synapse)


if __name__ == "__main__":
    with Miner() as miner:
        bt.logging.info("Random miner running...")
        while True:
            bt.logging.info(f"Miner UID: {miner.uid} | Incentive: {miner.metagraph.I[miner.uid]}")
            time.sleep(5 * 60)
