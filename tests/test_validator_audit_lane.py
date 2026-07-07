import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from poker44.validator.audit import (
    ValidatorAuditLane,
    VerathosClientConfig,
    VerathosAuditClient,
    build_validator_audit_evidence,
)


class ValidatorAuditLaneTests(unittest.TestCase):
    def test_local_audit_lane_persists_useful_record_without_provider(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "audit_reports.json.enc"
            lane = ValidatorAuditLane.from_env(path=target)

            evidence = build_validator_audit_evidence(
                validator_uid=74,
                validator_hotkey="5Validator",
                forward_count=12,
                dataset_hash="dataset-123",
                provider_stats={
                    "competition_epoch_id": "epoch-1",
                    "active_chunk_id": "chunk-1",
                    "active_chunk_hash": "chunk-hash-1",
                },
                competition_rows=[
                    {
                        "uid": 211,
                        "hotkey": "5MinerA",
                        "model_name": "poker44-benchmark-supervised",
                        "model_version": "v1",
                        "repo_url": "https://github.com/Poker44/miner-a",
                        "repo_commit": "abc123",
                        "manifest_digest": "digest-a",
                        "reward": 0.612,
                        "ap_score": 0.58,
                        "bot_recall": 0.66,
                        "human_safety_penalty": 0.0,
                        "coverage_rate": 1.0,
                        "latency_mean_seconds": 1.4,
                        "sample_count": 40,
                    }
                ],
                chunk_count=8,
                total_hands=320,
                human_chunk_count=4,
                bot_chunk_count=4,
                suspicion_summary={"tracked_miners": 1},
                compliance_summary={"transparent_miners": 1},
                served_chunk_summary={"unique_chunk_count": 8},
                max_rows=8,
            )

            record = lane.record_cycle(evidence=evidence)

            self.assertEqual(record["status"], "local_only")
            self.assertEqual(record["competition_epoch_id"], "epoch-1")
            self.assertEqual(record["top_uid"], 211)
            self.assertTrue(target.exists())
            encrypted_payload = target.read_text(encoding="utf-8")
            self.assertIn('"ciphertext_b64"', encrypted_payload)
            self.assertNotIn("dataset-123", encrypted_payload)
            self.assertNotIn("poker44-benchmark-supervised", encrypted_payload)
            summary_path = Path(tmpdir) / "audit_reports.summary.json"
            self.assertTrue(summary_path.exists())
            summary_payload = summary_path.read_text(encoding="utf-8")
            self.assertNotIn('"evidence"', summary_payload)
            self.assertNotIn("poker44-benchmark-supervised", summary_payload)
            summary = lane.public_summary()
            self.assertEqual(summary["provider"], "none")
            self.assertEqual(summary["last_status"], "local_only")

    def test_verathos_client_parses_verified_response(self):
        config = VerathosClientConfig(
            base_url="https://api.verathos.ai/v1",
            api_key="secret",
            model="verathos-model",
            timeout_seconds=10.0,
        )
        client = VerathosAuditClient(config)

        class _Response:
            def raise_for_status(self):
                return None

            def json(self):
                return {
                    "id": "chatcmpl-123",
                    "model": "verathos-model",
                    "proof_verified": True,
                    "verification": {"verified": True, "proof_id": "proof-1"},
                    "choices": [
                        {
                            "message": {
                                "content": (
                                    '{"summary":"Looks good","findings":["stable"],'
                                    '"next_steps":["monitor"],"confidence":"high",'
                                    '"integrity_flags":[]}'
                                )
                            }
                        }
                    ],
                    "usage": {"total_tokens": 123},
                }

        with patch("poker44.validator.audit.requests.post", return_value=_Response()):
            result = client.run_audit({"hello": "world"})

        self.assertTrue(result["proof_verified"])
        self.assertEqual(result["provider_request_id"], "chatcmpl-123")
        self.assertEqual(result["audit_output"]["summary"], "Looks good")

    def test_from_env_builds_verathos_lane_when_configured(self):
        with tempfile.TemporaryDirectory() as tmpdir, patch.dict(
            os.environ,
            {
                "POKER44_AUDIT_PROVIDER": "verathos",
                "POKER44_VERATHOS_API_KEY": "secret",
                "POKER44_VERATHOS_MODEL": "verathos-model",
            },
            clear=False,
        ):
            target = Path(tmpdir) / "audit_reports.json.enc"
            lane = ValidatorAuditLane.from_env(path=target)

        self.assertEqual(lane.provider, "verathos")
        self.assertIsNotNone(lane.verathos_client)


if __name__ == "__main__":
    unittest.main()
