"""Helpers for publishing and validating Poker44 miner model manifests."""

from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional

MIN_REQUIRED_MANIFEST_FIELDS = [
    "open_source",
    "repo_url",
    "repo_commit",
    "model_name",
    "model_version",
    "training_data_statement",
    "private_data_attestation",
]
REFERENCE_MINER_MODEL_NAME = "poker44-reference-heuristic"
REFERENCE_REPO_URL = "https://github.com/Poker44/Poker44-subnet"
GIT_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")


def _parse_bool(value: str | None, *, default: bool = False) -> bool:
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "y", "on"}


def _sha256_for_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted((p.resolve() for p in paths), key=lambda p: str(p)):
        digest.update(str(path).encode("utf-8"))
        with path.open("rb") as handle:
            while True:
                chunk = handle.read(1024 * 1024)
                if not chunk:
                    break
                digest.update(chunk)
    return digest.hexdigest()


def build_local_model_manifest(
    *,
    repo_root: Path,
    implementation_files: Iterable[Path],
    defaults: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Build a serializable manifest for the miner's current implementation."""
    implementation_paths = [path.resolve() for path in implementation_files]
    implementation_sha256 = _sha256_for_files(implementation_paths)
    default_values = dict(defaults or {})

    manifest: Dict[str, Any] = {
        "schema_version": "1",
        "open_source": _parse_bool(
            os.getenv("POKER44_MODEL_OPEN_SOURCE"),
            default=bool(default_values.get("open_source", True)),
        ),
        "model_name": os.getenv(
            "POKER44_MODEL_NAME",
            str(default_values.get("model_name", "poker44-reference-heuristic")),
        ),
        "model_version": os.getenv(
            "POKER44_MODEL_VERSION",
            str(default_values.get("model_version", "dev")),
        ),
        "framework": os.getenv(
            "POKER44_MODEL_FRAMEWORK",
            str(default_values.get("framework", "python-heuristic")),
        ),
        "license": os.getenv(
            "POKER44_MODEL_LICENSE",
            str(default_values.get("license", "MIT")),
        ),
        "repo_url": os.getenv(
            "POKER44_MODEL_REPO_URL",
            str(default_values.get("repo_url", "")),
        ).strip(),
        "repo_commit": os.getenv(
            "POKER44_MODEL_REPO_COMMIT",
            str(default_values.get("repo_commit", "")),
        ).strip(),
        "artifact_url": os.getenv(
            "POKER44_MODEL_ARTIFACT_URL",
            str(default_values.get("artifact_url", "")),
        ).strip(),
        "artifact_sha256": os.getenv(
            "POKER44_MODEL_ARTIFACT_SHA256",
            str(default_values.get("artifact_sha256", "")),
        ).strip(),
        "model_card_url": os.getenv(
            "POKER44_MODEL_CARD_URL",
            str(default_values.get("model_card_url", "")),
        ).strip(),
        "training_data_statement": os.getenv(
            "POKER44_MODEL_TRAINING_DATA_STATEMENT",
            str(default_values.get("training_data_statement", "")),
        ).strip(),
        "training_data_sources": [
            item.strip()
            for item in os.getenv(
                "POKER44_MODEL_TRAINING_DATA_SOURCES",
                ",".join(default_values.get("training_data_sources", [])),
            ).split(",")
            if item.strip()
        ],
        "private_data_attestation": os.getenv(
            "POKER44_MODEL_PRIVATE_DATA_ATTESTATION",
            str(default_values.get("private_data_attestation", "")),
        ).strip(),
        "inference_mode": os.getenv(
            "POKER44_MODEL_INFERENCE_MODE",
            str(default_values.get("inference_mode", "remote")),
        ).strip(),
        "implementation_sha256": implementation_sha256,
        "implementation_files": [
            str(path.relative_to(repo_root)) if path.is_relative_to(repo_root) else str(path)
            for path in implementation_paths
        ],
        "notes": os.getenv(
            "POKER44_MODEL_NOTES",
            str(default_values.get("notes", "")),
        ).strip(),
    }
    return normalize_model_manifest(manifest)


def normalize_model_manifest(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Return a compact, JSON-stable manifest dictionary."""
    if not manifest:
        return {}

    normalized: Dict[str, Any] = {}
    for key, value in manifest.items():
        if value is None:
            continue
        if isinstance(value, bool):
            normalized[key] = value
            continue
        if isinstance(value, (int, float)):
            normalized[key] = value
            continue
        if isinstance(value, (list, tuple)):
            cleaned_list: List[Any] = []
            for item in value:
                if item is None:
                    continue
                cleaned_item = str(item).strip()
                if cleaned_item:
                    cleaned_list.append(cleaned_item)
            if cleaned_list:
                normalized[key] = cleaned_list
            continue

        cleaned = str(value).strip()
        if cleaned:
            normalized[key] = cleaned

    if "open_source" in manifest:
        raw = manifest.get("open_source")
        if isinstance(raw, bool):
            normalized["open_source"] = raw
        else:
            normalized["open_source"] = _parse_bool(str(raw), default=False)

    return normalized


def manifest_digest(manifest: Optional[Mapping[str, Any]]) -> str:
    """Return a stable digest for change detection."""
    normalized = normalize_model_manifest(manifest)
    payload = json.dumps(normalized, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _uses_reference_repo(manifest: Mapping[str, Any]) -> bool:
    return str(manifest.get("repo_url", "")).strip().rstrip("/") == REFERENCE_REPO_URL


def _is_reference_miner_manifest(manifest: Mapping[str, Any]) -> bool:
    return str(manifest.get("model_name", "")).strip() == REFERENCE_MINER_MODEL_NAME


def _has_implementation_files(manifest: Mapping[str, Any]) -> bool:
    value = manifest.get("implementation_files")
    if not isinstance(value, (list, tuple)):
        return False
    return any(str(item).strip() for item in value)


def _looks_like_git_commit(value: Any) -> bool:
    return bool(GIT_COMMIT_RE.fullmatch(str(value).strip()))


def evaluate_manifest_compliance(manifest: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    """Classify whether a manifest meets the current transparent-miner standard."""
    if not manifest:
        return {
            "status": "opaque",
            "missing_fields": list(MIN_REQUIRED_MANIFEST_FIELDS),
            "required_fields": list(MIN_REQUIRED_MANIFEST_FIELDS),
            "open_source": False,
            "policy_violations": [],
        }

    missing_fields: List[str] = []
    for field in MIN_REQUIRED_MANIFEST_FIELDS:
        value = manifest.get(field)
        if field == "open_source":
            if not bool(value):
                missing_fields.append(field)
            continue
        if value is None:
            missing_fields.append(field)
            continue
        if isinstance(value, str) and not value.strip():
            missing_fields.append(field)
            continue

    if not _has_implementation_files(manifest):
        missing_fields.append("implementation_files")
    if not str(manifest.get("implementation_sha256", "")).strip():
        missing_fields.append("implementation_sha256")

    policy_violations: List[str] = []
    if not _looks_like_git_commit(manifest.get("repo_commit", "")):
        policy_violations.append("repo_commit_invalid")
    if _uses_reference_repo(manifest) and not _is_reference_miner_manifest(manifest):
        policy_violations.append("repo_url_must_point_to_model_repo")

    status = "transparent" if not missing_fields and not policy_violations else "opaque"
    return {
        "status": status,
        "missing_fields": missing_fields,
        "required_fields": list(MIN_REQUIRED_MANIFEST_FIELDS),
        "open_source": bool(manifest.get("open_source", False)),
        "policy_violations": policy_violations,
    }
