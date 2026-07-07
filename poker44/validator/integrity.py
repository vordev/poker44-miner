"""Integrity helpers for validator-side evaluation-integrity tracking."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Optional

from poker44.utils.model_manifest import (
    MIN_REQUIRED_MANIFEST_FIELDS,
    evaluate_manifest_compliance,
)

UTC = timezone.utc
MAX_RECENT_CYCLES = 64


def load_json_registry(path: str | Path | None, *, default: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    if path is None:
        return dict(default or {})

    target_path = Path(path)
    if not target_path.exists():
        return dict(default or {})

    try:
        with target_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except Exception:
        return dict(default or {})

    if isinstance(payload, dict):
        return payload
    return dict(default or {})


def persist_json_registry(path: str | Path | None, payload: Mapping[str, Any]) -> None:
    if path is None:
        return

    target_path = Path(path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = target_path.with_suffix(".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
    tmp_path.replace(target_path)


def normalize_uid_key_registry(
    registry: MutableMapping[Any, Any],
) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for uid in sorted(registry, key=lambda value: int(value)):
        normalized[str(int(uid))] = registry[uid]
    return normalized


def remove_uid_from_model_manifest_registry(
    registry: MutableMapping[Any, Any],
    uid: int,
) -> bool:
    return registry.pop(str(int(uid)), None) is not None


def remove_uid_from_suspicion_registry(
    registry: MutableMapping[str, Any],
    uid: int,
) -> bool:
    miners = registry.setdefault("miners", {})
    removed = miners.pop(str(int(uid)), None) is not None
    registry["summary"] = {
        "tracked_miners": len(miners),
        "last_forward_count": int(registry.get("summary", {}).get("last_forward_count", 0)),
    }
    return removed


def remove_uid_from_compliance_registry(
    registry: MutableMapping[str, Any],
    uid: int,
) -> bool:
    miners = registry.setdefault("miners", {})
    removed = miners.pop(str(int(uid)), None) is not None
    transparent_count = sum(1 for item in miners.values() if item.get("status") == "transparent")
    opaque_count = sum(1 for item in miners.values() if item.get("status") == "opaque")
    registry["summary"] = {
        "tracked_miners": len(miners),
        "transparent_miners": transparent_count,
        "opaque_miners": opaque_count,
        "last_forward_count": int(registry.get("summary", {}).get("last_forward_count", 0)),
    }
    return removed


def chunk_fingerprint(chunk: Iterable[Mapping[str, Any]]) -> str:
    encoded = json.dumps(
        list(chunk),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def record_served_chunks(
    registry: MutableMapping[str, Any],
    *,
    chunk_hashes: List[str],
    forward_count: int,
    dataset_hash: str,
) -> Dict[str, Any]:
    chunk_index = registry.setdefault("chunk_index", {})
    recent_cycles = registry.setdefault("recent_cycles", [])
    now_iso = datetime.now(tz=UTC).isoformat()
    repeated_hashes: List[str] = []

    for chunk_hash in chunk_hashes:
        entry = chunk_index.get(chunk_hash)
        if entry is None:
            entry = {
                "times_served": 0,
                "first_forward_count": int(forward_count),
                "first_dataset_hash": dataset_hash,
                "first_seen_at": now_iso,
            }
            chunk_index[chunk_hash] = entry
        else:
            repeated_hashes.append(chunk_hash)

        entry["times_served"] = int(entry.get("times_served", 0)) + 1
        entry["last_forward_count"] = int(forward_count)
        entry["last_dataset_hash"] = dataset_hash
        entry["last_seen_at"] = now_iso

    recent_cycles.append(
        {
            "forward_count": int(forward_count),
            "timestamp": now_iso,
            "dataset_hash": dataset_hash,
            "chunk_count": len(chunk_hashes),
            "repeated_chunk_count": len(repeated_hashes),
            "chunk_hashes": list(chunk_hashes),
        }
    )
    if len(recent_cycles) > MAX_RECENT_CYCLES:
        del recent_cycles[:-MAX_RECENT_CYCLES]

    registry["summary"] = {
        "unique_chunk_count": len(chunk_index),
        "recent_cycle_count": len(recent_cycles),
        "last_forward_count": int(forward_count),
        "last_dataset_hash": dataset_hash,
        "last_repeated_chunk_count": len(repeated_hashes),
    }
    return {
        "repeated_hashes": repeated_hashes,
        "repeated_count": len(repeated_hashes),
        "unique_count": len(chunk_index),
    }


def evaluate_manifest_suspicion(manifest: Optional[Mapping[str, Any]]) -> List[str]:
    if not manifest:
        return ["missing_model_manifest"]

    reasons: List[str] = []
    if not bool(manifest.get("open_source", False)):
        reasons.append("manifest_not_open_source")
    if not str(manifest.get("repo_url", "")).strip():
        reasons.append("manifest_missing_repo_url")
    if not str(manifest.get("repo_commit", "")).strip():
        reasons.append("manifest_missing_repo_commit")
    if not str(manifest.get("training_data_statement", "")).strip():
        reasons.append("manifest_missing_training_data_statement")
    if not str(manifest.get("private_data_attestation", "")).strip():
        reasons.append("manifest_missing_private_data_attestation")
    return reasons


def update_suspicion_registry(
    registry: MutableMapping[str, Any],
    *,
    uid: int,
    reasons: List[str],
    forward_count: int,
    dataset_hash: str,
) -> Optional[Dict[str, Any]]:
    if not reasons:
        return None

    miners = registry.setdefault("miners", {})
    key = str(uid)
    entry = miners.get(key)
    now_iso = datetime.now(tz=UTC).isoformat()
    if entry is None:
        entry = {
            "uid": int(uid),
            "reason_counts": {},
            "events": [],
        }
        miners[key] = entry

    reason_counts = entry.setdefault("reason_counts", {})
    for reason in reasons:
        reason_counts[reason] = int(reason_counts.get(reason, 0)) + 1

    event = {
        "timestamp": now_iso,
        "forward_count": int(forward_count),
        "dataset_hash": dataset_hash,
        "reasons": list(reasons),
    }
    events = entry.setdefault("events", [])
    events.append(event)
    if len(events) > MAX_RECENT_CYCLES:
        del events[:-MAX_RECENT_CYCLES]

    entry["last_event"] = event
    entry["last_forward_count"] = int(forward_count)
    registry["summary"] = {
        "tracked_miners": len(miners),
        "last_forward_count": int(forward_count),
    }
    return event


def update_compliance_registry(
    registry: MutableMapping[str, Any],
    *,
    uid: int,
    compliance: Mapping[str, Any],
    manifest_digest: str,
    forward_count: int,
    dataset_hash: str,
) -> Dict[str, Any]:
    miners = registry.setdefault("miners", {})
    key = str(uid)
    now_iso = datetime.now(tz=UTC).isoformat()
    previous = miners.get(key, {})

    entry = {
        "uid": int(uid),
        "status": str(compliance.get("status", "opaque")),
        "missing_fields": list(compliance.get("missing_fields", [])),
        "policy_violations": list(compliance.get("policy_violations", [])),
        "required_fields": list(compliance.get("required_fields", MIN_REQUIRED_MANIFEST_FIELDS)),
        "open_source": bool(compliance.get("open_source", False)),
        "manifest_digest": manifest_digest,
        "updated_at": now_iso,
        "forward_count": int(forward_count),
        "dataset_hash": dataset_hash,
        "status_changed": previous.get("status") != compliance.get("status"),
    }
    miners[key] = entry

    transparent_count = sum(1 for item in miners.values() if item.get("status") == "transparent")
    opaque_count = sum(1 for item in miners.values() if item.get("status") == "opaque")
    registry["summary"] = {
        "tracked_miners": len(miners),
        "transparent_miners": transparent_count,
        "opaque_miners": opaque_count,
        "last_forward_count": int(forward_count),
    }
    return entry
