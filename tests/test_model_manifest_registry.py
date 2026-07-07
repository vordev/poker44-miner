import json
import tempfile
import unittest
from pathlib import Path

from poker44.validator.integrity import (
    load_json_registry,
    normalize_uid_key_registry,
    persist_json_registry,
    remove_uid_from_compliance_registry,
    remove_uid_from_model_manifest_registry,
    remove_uid_from_suspicion_registry,
)


class ModelManifestRegistryTests(unittest.TestCase):
    def test_persist_normalizes_mixed_uid_key_types(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            registry_path = Path(tmp_dir) / "model_manifests.json"
            registry_path.write_text(
                json.dumps(
                    {
                        "14": {
                            "uid": 14,
                            "manifest_digest": "old",
                            "model_manifest": {"model_name": "old"},
                        }
                    }
                ),
                encoding="utf-8",
            )

            registry = load_json_registry(registry_path)
            registry[136] = {
                "uid": 136,
                "manifest_digest": "new",
                "model_manifest": {"model_name": "new"},
            }

            normalized_registry = normalize_uid_key_registry(registry)
            registry.clear()
            registry.update(normalized_registry)
            persist_json_registry(registry_path, normalized_registry)

            self.assertEqual(set(registry.keys()), {"14", "136"})

            persisted = json.loads(registry_path.read_text(encoding="utf-8"))
            self.assertEqual(set(persisted.keys()), {"14", "136"})
            self.assertEqual(persisted["136"]["uid"], 136)

    def test_uid_removal_helpers_prune_registries(self) -> None:
        manifest_registry = {
            "14": {"uid": 14},
            "136": {"uid": 136},
        }
        compliance_registry = {
            "miners": {
                "14": {"status": "transparent"},
                "136": {"status": "opaque"},
            },
            "summary": {
                "tracked_miners": 2,
                "transparent_miners": 1,
                "opaque_miners": 1,
                "last_forward_count": 9,
            },
        }
        suspicion_registry = {
            "miners": {
                "14": {"uid": 14},
                "136": {"uid": 136},
            },
            "summary": {
                "tracked_miners": 2,
                "last_forward_count": 9,
            },
        }

        self.assertTrue(remove_uid_from_model_manifest_registry(manifest_registry, 136))
        self.assertTrue(remove_uid_from_compliance_registry(compliance_registry, 136))
        self.assertTrue(remove_uid_from_suspicion_registry(suspicion_registry, 136))

        self.assertEqual(set(manifest_registry.keys()), {"14"})
        self.assertEqual(set(compliance_registry["miners"].keys()), {"14"})
        self.assertEqual(set(suspicion_registry["miners"].keys()), {"14"})
        self.assertEqual(compliance_registry["summary"]["tracked_miners"], 1)
        self.assertEqual(compliance_registry["summary"]["transparent_miners"], 1)
        self.assertEqual(compliance_registry["summary"]["opaque_miners"], 0)
        self.assertEqual(suspicion_registry["summary"]["tracked_miners"], 1)


if __name__ == "__main__":
    unittest.main()
