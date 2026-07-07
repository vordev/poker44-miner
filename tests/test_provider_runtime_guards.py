import os
import unittest
from unittest.mock import patch

from poker44.validator.forward import _compute_windowed_rewards, _finalize_provider_cycle
from poker44.validator.runtime_provider import ProviderRuntimeConfig, ProviderRuntimeDatasetProvider


class _DummyProvider:
    def __init__(self) -> None:
        self.mark_calls = 0

    def mark_last_batch_evaluated(self) -> None:
        self.mark_calls += 1


class _DummyValidator:
    def __init__(self) -> None:
        self.provider = _DummyProvider()


class _DummyClient:
    def __init__(self, batches):
        self.batches = batches

    def get(self, path):
        self.last_path = path
        return {
            "chunkId": "chunk-1",
            "chunkHash": "hash-1",
            "windowStart": "2026-06-27T00:00:00Z",
            "windowEnd": "2026-06-28T12:00:00Z",
            "batches": self.batches,
        }


class _DummyRuntimeManager:
    def __init__(self, batches):
        self.client = _DummyClient(batches)
        self.status = {"available_hands": 999, "ready_for_evaluation": True}

    def ensure_runtime_ready(self):
        return True


class ProviderRuntimeGuardTests(unittest.TestCase):
    def test_defaults_to_public_eval_api_base_url(self):
        with patch.dict(
            os.environ,
            {},
            clear=True,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.api_base_url, "https://api.poker44.net")
        self.assertEqual(cfg.internal_secret, "")
        self.assertEqual(cfg.validator_id, "validator_hotkey")

    def test_rejects_placeholder_internal_secret(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_EVAL_API_BASE_URL": "http://127.0.0.1:3001",
                "POKER44_PROVIDER_INTERNAL_SECRET": "force-start-secret",
            },
            clear=False,
        ):
            with self.assertRaisesRegex(RuntimeError, "placeholder"):
                ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

    def test_allows_missing_internal_secret_for_signed_validator_auth(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_EVAL_API_BASE_URL": "http://127.0.0.1:3001",
            },
            clear=False,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:3001")
        self.assertEqual(cfg.internal_secret, "")
        self.assertEqual(cfg.validator_id, "validator_hotkey")
        self.assertEqual(cfg.request_timeout_seconds, 180)

    def test_accepts_real_internal_secret(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_EVAL_API_BASE_URL": "http://127.0.0.1:3001",
                "POKER44_PROVIDER_INTERNAL_SECRET": "real-secret-value",
            },
            clear=False,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.api_base_url, "http://127.0.0.1:3001")
        self.assertEqual(cfg.internal_secret, "real-secret-value")
        self.assertEqual(cfg.validator_id, "validator_hotkey")
        self.assertEqual(cfg.request_timeout_seconds, 180)

    def test_provider_request_timeout_can_be_overridden(self):
        with patch.dict(
            os.environ,
            {
                "POKER44_PROVIDER_REQUEST_TIMEOUT_SECONDS": "180",
            },
            clear=False,
        ):
            cfg = ProviderRuntimeConfig.from_env(default_validator_id="validator_hotkey")

        self.assertEqual(cfg.request_timeout_seconds, 180)

    def test_provider_cycle_finalization_requires_completed_evaluation(self):
        validator = _DummyValidator()

        _finalize_provider_cycle(validator, evaluation_completed=False)
        self.assertEqual(validator.provider.mark_calls, 0)

        _finalize_provider_cycle(validator, evaluation_completed=True)
        self.assertEqual(validator.provider.mark_calls, 1)

    def test_fetch_hand_batch_uses_full_backend_snapshot_even_when_limit_is_passed(self):
        batches = [
            {"hands": [{"hand_id": f"h-{index}"}], "is_bot": index % 2 == 0}
            for index in range(5)
        ]
        cfg = ProviderRuntimeConfig(
            api_base_url="http://127.0.0.1:3001",
            internal_secret="",
            validator_id="validator_hotkey",
        )
        provider = ProviderRuntimeDatasetProvider(cfg)
        provider.manager = _DummyRuntimeManager(batches)

        fetched = provider.fetch_hand_batch(limit=2)

        self.assertEqual(len(fetched), 5)
        self.assertEqual(provider.stats["batch_count"], 5)
        self.assertEqual(provider.stats["requested_limit"], "all")

    def test_reward_window_uses_current_backend_snapshot_size(self):
        validator = _DummyValidator()
        validator.reward_window = 40
        validator.current_eval_sample_count = 100
        validator.prediction_buffer = {88: [0.1] * 50 + [0.9] * 50}
        validator.label_buffer = {88: [False] * 50 + [True] * 50}
        validator.coverage_buffer = {88: [1.0]}
        validator.latency_buffer = {88: [10.0]}

        _rewards, metrics = _compute_windowed_rewards(validator, [88])

        self.assertEqual(metrics[0]["sample_count"], 100)


if __name__ == "__main__":
    unittest.main()
