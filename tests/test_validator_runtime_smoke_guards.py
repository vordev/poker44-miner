import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from neurons.validator import Validator
from poker44.validator.forward import _persist_model_manifest_registry


class ValidatorRuntimeSmokeGuardTests(unittest.TestCase):
    def test_persist_model_manifest_registry_handles_mixed_uid_key_types(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            target = Path(tmpdir) / "model_manifests.json"
            registry = {
                "12": {"uid": 12, "manifest_digest": "a", "model_manifest": {}},
                3: {"uid": 3, "manifest_digest": "b", "model_manifest": {}},
            }

            _persist_model_manifest_registry(target, registry)

            self.assertTrue(target.exists())
            payload = target.read_text(encoding="utf-8")
            self.assertIn('"3"', payload)
            self.assertIn('"12"', payload)

    def test_runtime_snapshot_tolerates_missing_runtime_mode_during_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            dummy = SimpleNamespace()
            dummy.provider = None
            dummy.wallet = SimpleNamespace(
                hotkey=SimpleNamespace(ss58_address="5DummyHotkey"),
            )
            dummy.config = SimpleNamespace(
                netuid=126,
                neuron=SimpleNamespace(full_path=tmpdir),
            )
            dummy.runtime_info = {"python_version": "test"}
            dummy.competition_scores_payload = []
            dummy.audit_summary = {"provider": "none", "last_status": "local_only"}
            dummy.resolve_uid = lambda _hotkey: 74
            dummy.runtime_snapshot_path = Path(tmpdir) / "validator_runtime.json"
            dummy.network_snapshot_path = Path(tmpdir) / "network_snapshot.json"

            with patch("neurons.validator.write_runtime_snapshot") as write_snapshot, patch(
                "neurons.validator.collect_network_snapshot",
                return_value={"network": "ok"},
            ), patch(
                "neurons.validator.build_signed_runtime_request",
                return_value={},
            ), patch("neurons.validator.post_runtime_snapshot", return_value=(False, "skipped")):
                Validator._write_runtime_snapshot(dummy, status="initializing")

            payload = write_snapshot.call_args_list[0].args[1]
            self.assertEqual(payload["runtime_mode"], "initializing")
            self.assertEqual(payload["validator_uid"], 74)
            self.assertEqual(payload["audit"]["provider"], "none")


if __name__ == "__main__":
    unittest.main()
