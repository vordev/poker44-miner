"""Validator audit lane with optional Verathos-backed verification."""

from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, MutableMapping, Optional

import requests
from Crypto.Cipher import AES, PKCS1_OAEP
from Crypto.PublicKey import RSA
from Crypto.Random import get_random_bytes

from poker44.validator.integrity import load_json_registry, persist_json_registry

UTC = timezone.utc
DEFAULT_VERATHOS_BASE_URL = "https://api.verathos.ai/v1"
DEFAULT_RECENT_REPORTS = 32
DEFAULT_POKER44_AUDIT_PUBLIC_KEY_PEM = """-----BEGIN PUBLIC KEY-----
MIICIjANBgkqhkiG9w0BAQEFAAOCAg8AMIICCgKCAgEA+QF4Cr+x6GnUW+fJGvzp
L7LFpFGpWzqaQNLrjp1khPSIMJETSovGD/hHI2RZFv49+DKP5pzT+oN4k8aVbq9H
oWxH6jdAe+1Eh1Cup5q6x/YQjo6qy1qfV+fR0Mv4RR7mV1+kYj7n5VsCZ09gx1BS
Puy+yookhH2LBi5Om/x3+Uw0rS1rB1lBAGD0OezWoM1DCc9HvWN3w1QN/D8ulDUa
euflk6MTBIq0Tj4SqVTxBL79FtYcMTdnQ3cxkIZkCVl00pFmk3LmlkwIeeeYbf/g
TzD4Usr0FbisvYEzt98TgFVTkoD5rlrAWduMYm02bBiNdcXyoJ1/TFXiT/sU2CIT
w0bQp9EAhPGSgHkUrIziZURThuz2VRCYtWbq+9UGS+4ApvAAwQoWqHgeXKzxt1ct
igDyefBwQ+cClPVJ5zoOKCdn5ms8KBbwk0PFWPDYcagpRprwz7Banckm1/0CkZ19
3Gy/qkCZB8ilwueKS72Nt1X1gWWWhY9MNXB15748FXQNRkbChTi27cKYOvBeKlPs
QXgtks41m0Wpciy8M04e3KCoNi/PYy152OrOL5Yb+IULea4H/zbC67L3T1bOkP7c
asumtu5xcdvJw9o5grZp2SjrLn9NKXpivoMWc9KUOoxSaam3cHu9QPzo1rwBtxfj
gxDz6y08h3vD4eqZC7+Glw8CAwEAAQ==
-----END PUBLIC KEY-----"""


def _now_iso() -> str:
    return datetime.now(tz=UTC).isoformat()


def _sha256_json(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def _truncate_text(value: Any, *, limit: int = 800) -> str:
    text = str(value or "").strip()
    if len(text) <= limit:
        return text
    return f"{text[:limit]}..."


def _normalize_json_content(raw: str) -> Dict[str, Any]:
    text = str(raw or "").strip()
    if not text:
        return {"summary": "", "findings": [], "next_steps": []}

    try:
        payload = json.loads(text)
    except Exception:
        return {
            "summary": text,
            "findings": [],
            "next_steps": [],
        }

    if isinstance(payload, dict):
        return payload
    return {
        "summary": text,
        "findings": [],
        "next_steps": [],
    }


def _public_key_pem_from_env_or_default() -> str:
    pem = str(os.getenv("POKER44_AUDIT_PUBLIC_KEY_PEM", "")).strip()
    return pem or DEFAULT_POKER44_AUDIT_PUBLIC_KEY_PEM


def _encrypt_audit_payload(payload: Mapping[str, Any], *, public_key_pem: str) -> Dict[str, Any]:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    ).encode("utf-8")
    session_key = get_random_bytes(32)
    cipher_rsa = PKCS1_OAEP.new(RSA.import_key(public_key_pem))
    encrypted_session_key = cipher_rsa.encrypt(session_key)
    cipher_aes = AES.new(session_key, AES.MODE_GCM)
    ciphertext, tag = cipher_aes.encrypt_and_digest(encoded)
    return {
        "schema": "poker44.audit.encrypted.v1",
        "cipher": "AES-256-GCM",
        "key_wrap": "RSA-OAEP",
        "public_key_fingerprint": hashlib.sha256(public_key_pem.encode("utf-8")).hexdigest(),
        "created_at": _now_iso(),
        "payload_sha256": hashlib.sha256(encoded).hexdigest(),
        "encrypted_session_key_b64": base64.b64encode(encrypted_session_key).decode("ascii"),
        "nonce_b64": base64.b64encode(cipher_aes.nonce).decode("ascii"),
        "tag_b64": base64.b64encode(tag).decode("ascii"),
        "ciphertext_b64": base64.b64encode(ciphertext).decode("ascii"),
    }


@dataclass(frozen=True)
class VerathosClientConfig:
    base_url: str
    api_key: str
    model: str
    timeout_seconds: float

    @classmethod
    def from_env(cls) -> Optional["VerathosClientConfig"]:
        api_key = str(os.getenv("POKER44_VERATHOS_API_KEY", "")).strip()
        model = str(os.getenv("POKER44_VERATHOS_MODEL", "")).strip()
        if not api_key or not model:
            return None

        base_url = str(
            os.getenv("POKER44_VERATHOS_BASE_URL", DEFAULT_VERATHOS_BASE_URL)
        ).strip().rstrip("/")
        timeout_seconds = float(
            os.getenv("POKER44_VERATHOS_TIMEOUT_SECONDS", "20")
        )
        return cls(
            base_url=base_url or DEFAULT_VERATHOS_BASE_URL,
            api_key=api_key,
            model=model,
            timeout_seconds=timeout_seconds,
        )


class VerathosAuditClient:
    """Thin OpenAI-compatible client for Verathos audit calls."""

    def __init__(self, config: VerathosClientConfig) -> None:
        self.config = config

    def run_audit(self, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        payload = {
            "model": self.config.model,
            "temperature": 0,
            "response_format": {"type": "json_object"},
            "messages": [
                {
                    "role": "system",
                    "content": (
                        "You are producing a structured audit summary for Poker44, "
                        "an online poker bot-detection validator. Review the evidence "
                        "and return compact JSON with keys: summary, findings, next_steps, "
                        "confidence, and integrity_flags."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(evidence, sort_keys=True),
                },
            ],
        }
        started_at = time.time()
        response = requests.post(
            f"{self.config.base_url}/chat/completions",
            headers={
                "Authorization": f"Bearer {self.config.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.config.timeout_seconds,
        )
        latency_seconds = time.time() - started_at
        response.raise_for_status()

        body = response.json()
        message_content = ""
        try:
            message_content = (
                body.get("choices", [{}])[0]
                .get("message", {})
                .get("content", "")
            )
        except Exception:
            message_content = ""

        parsed = _normalize_json_content(message_content)
        verification_metadata = {
            "proof_verified": body.get("proof_verified"),
            "verification": body.get("verification"),
            "proof": body.get("proof"),
            "attestation": body.get("attestation"),
        }
        # Drop empty keys to keep snapshots compact.
        verification_metadata = {
            key: value
            for key, value in verification_metadata.items()
            if value not in (None, "", {}, [])
        }

        proof_verified = bool(
            body.get("proof_verified")
            or (body.get("verification") or {}).get("verified")
            or (body.get("attestation") or {}).get("verified")
        )

        return {
            "provider": "verathos",
            "provider_request_id": body.get("id"),
            "provider_model": body.get("model") or self.config.model,
            "proof_verified": proof_verified,
            "verification_metadata": verification_metadata,
            "latency_seconds": latency_seconds,
            "usage": body.get("usage") or {},
            "audit_output": parsed,
            "raw_response_hash": hashlib.sha256(
                json.dumps(body, sort_keys=True).encode("utf-8")
            ).hexdigest(),
        }


class ValidatorAuditLane:
    """Best-effort audit trail with optional external verification."""

    def __init__(
        self,
        *,
        path: str | Path | None,
        summary_path: str | Path | None,
        provider: str,
        mode: str,
        recent_limit: int = DEFAULT_RECENT_REPORTS,
        verathos_client: Optional[VerathosAuditClient] = None,
    ) -> None:
        self.path = Path(path) if path is not None else None
        self.summary_path = Path(summary_path) if summary_path is not None else None
        self.provider = provider
        self.mode = mode
        self.recent_limit = max(1, int(recent_limit))
        self.verathos_client = verathos_client
        self.registry = load_json_registry(
            self.summary_path,
            default={"latest": {}, "recent_reports": [], "summary": {}},
        )
        self._sanitize_summary_registry()
        self._migrate_legacy_plaintext_if_present()
        self._recompute_summary()

    @classmethod
    def from_env(cls, *, path: str | Path | None) -> "ValidatorAuditLane":
        provider = str(os.getenv("POKER44_AUDIT_PROVIDER", "none")).strip().lower()
        mode = str(os.getenv("POKER44_AUDIT_MODE", "shadow")).strip().lower()
        recent_limit = int(
            os.getenv("POKER44_AUDIT_RECENT_REPORT_LIMIT", str(DEFAULT_RECENT_REPORTS))
        )

        verathos_client = None
        if provider == "verathos":
            client_cfg = VerathosClientConfig.from_env()
            if client_cfg is not None:
                verathos_client = VerathosAuditClient(client_cfg)

        encrypted_path = Path(path) if path is not None else None
        summary_path = None
        if encrypted_path is not None:
            if encrypted_path.name.endswith(".json.enc"):
                summary_name = encrypted_path.name.replace(".json.enc", ".summary.json")
            else:
                summary_name = f"{encrypted_path.name}.summary.json"
            summary_path = encrypted_path.with_name(summary_name)

        return cls(
            path=encrypted_path,
            summary_path=summary_path,
            provider=provider,
            mode=mode,
            recent_limit=recent_limit,
            verathos_client=verathos_client,
        )

    def public_summary(self) -> Dict[str, Any]:
        return dict(self.registry.get("summary", {}) or {})

    def latest_report(self) -> Dict[str, Any]:
        latest = self.registry.get("latest", {}) or {}
        if isinstance(latest, dict):
            return dict(latest)
        return {}

    def record_cycle(self, *, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        record = self._build_base_record(evidence=evidence)

        if self.provider == "none":
            record["status"] = "local_only"
            record["status_reason"] = "audit provider disabled"
        elif self.provider != "verathos":
            record["status"] = "provider_unsupported"
            record["status_reason"] = f"unsupported provider: {self.provider}"
        elif self.mode == "disabled":
            record["status"] = "provider_disabled"
            record["status_reason"] = "audit mode disabled"
        elif self.verathos_client is None:
            record["status"] = "provider_unconfigured"
            record["status_reason"] = "missing Verathos API configuration"
        else:
            try:
                result = self.verathos_client.run_audit(evidence)
                record.update(result)
                record["status"] = (
                    "verified" if bool(result.get("proof_verified")) else "provider_completed"
                )
                record["status_reason"] = ""
            except Exception as exc:
                record["status"] = "provider_failed"
                record["status_reason"] = _truncate_text(exc, limit=300)

        self._persist_encrypted(record)
        self._append_record(record)
        self._persist()
        return dict(record)

    def _build_base_record(self, *, evidence: Mapping[str, Any]) -> Dict[str, Any]:
        latest = self.latest_report()
        provider_stats = evidence.get("provider_stats", {}) if isinstance(evidence, Mapping) else {}
        top_rows = evidence.get("top_rows", []) if isinstance(evidence, Mapping) else []
        return {
            "created_at": _now_iso(),
            "provider": self.provider,
            "mode": self.mode,
            "status": "pending",
            "status_reason": "",
            "competition_epoch_id": provider_stats.get("competition_epoch_id"),
            "active_chunk_id": provider_stats.get("active_chunk_id"),
            "active_chunk_hash": provider_stats.get("active_chunk_hash"),
            "dataset_hash": evidence.get("dataset_hash"),
            "validator_uid": evidence.get("validator_uid"),
            "validator_hotkey": evidence.get("validator_hotkey"),
            "forward_count": evidence.get("forward_count"),
            "top_uid": top_rows[0].get("uid") if top_rows else None,
            "top_reward": top_rows[0].get("reward") if top_rows else None,
            "input_hash": _sha256_json(dict(evidence)),
            "evidence": dict(evidence),
            "previous_status": latest.get("status"),
        }

    def _append_record(self, record: MutableMapping[str, Any]) -> None:
        recent_reports = self.registry.setdefault("recent_reports", [])
        latest_payload = self._build_public_report_summary(record)
        self.registry["latest"] = latest_payload
        recent_reports.append(dict(latest_payload))
        if len(recent_reports) > self.recent_limit:
            del recent_reports[:-self.recent_limit]
        self._recompute_summary()

    def _recompute_summary(self) -> None:
        recent_reports = list(self.registry.get("recent_reports", []) or [])
        verified_reports = sum(1 for item in recent_reports if item.get("proof_verified"))
        failed_reports = sum(
            1
            for item in recent_reports
            if str(item.get("status", "")).startswith("provider_")
            and item.get("status") not in {"provider_completed"}
        )
        latest = self.latest_report()
        self.registry["summary"] = {
            "provider": self.provider,
            "mode": self.mode,
            "recent_report_count": len(recent_reports),
            "verified_report_count": verified_reports,
            "provider_failure_count": failed_reports,
            "last_status": latest.get("status"),
            "last_created_at": latest.get("created_at"),
            "last_proof_verified": bool(latest.get("proof_verified")),
            "last_provider_model": latest.get("provider_model"),
            "latest": latest,
        }

    def _persist(self) -> None:
        persist_json_registry(self.summary_path, self.registry)

    def _persist_encrypted(self, record: Mapping[str, Any]) -> None:
        if self.path is None:
            return
        envelope = _encrypt_audit_payload(
            record,
            public_key_pem=_public_key_pem_from_env_or_default(),
        )
        persist_json_registry(self.path, envelope)

    def _migrate_legacy_plaintext_if_present(self) -> None:
        if self.path is None:
            return
        if self.path.name.endswith(".json.enc"):
            legacy_path = self.path.with_suffix("")
        else:
            legacy_path = None

        if legacy_path is None or not legacy_path.exists():
            return

        legacy_payload = load_json_registry(legacy_path, default={})
        if legacy_payload and not self.registry.get("latest"):
            latest = legacy_payload.get("latest", {})
            recent_reports = legacy_payload.get("recent_reports", [])
            summary = legacy_payload.get("summary", {})
            if isinstance(latest, dict):
                self.registry["latest"] = self._build_public_report_summary(latest)
            if isinstance(recent_reports, list):
                self.registry["recent_reports"] = [
                    self._build_public_report_summary(item)
                    for item in recent_reports[-self.recent_limit :]
                    if isinstance(item, dict)
                ]
            if isinstance(summary, dict):
                self.registry["summary"] = summary
            persist_json_registry(self.summary_path, self.registry)

        try:
            legacy_path.unlink()
        except Exception:
            pass

    def _sanitize_summary_registry(self) -> None:
        latest = self.registry.get("latest", {})
        if isinstance(latest, dict):
            self.registry["latest"] = self._build_public_report_summary(latest)

        recent_reports = self.registry.get("recent_reports", [])
        if isinstance(recent_reports, list):
            self.registry["recent_reports"] = [
                self._build_public_report_summary(item)
                for item in recent_reports
                if isinstance(item, dict)
            ][-self.recent_limit :]

        persist_json_registry(self.summary_path, self.registry)

    @staticmethod
    def _build_public_report_summary(record: Mapping[str, Any]) -> Dict[str, Any]:
        return {
            "created_at": record.get("created_at"),
            "provider": record.get("provider"),
            "mode": record.get("mode"),
            "status": record.get("status"),
            "status_reason": record.get("status_reason"),
            "competition_epoch_id": record.get("competition_epoch_id"),
            "active_chunk_id": record.get("active_chunk_id"),
            "active_chunk_hash": record.get("active_chunk_hash"),
            "dataset_hash": record.get("dataset_hash"),
            "top_uid": record.get("top_uid"),
            "top_reward": record.get("top_reward"),
            "proof_verified": bool(record.get("proof_verified")),
            "provider_model": record.get("provider_model"),
            "provider_request_id": record.get("provider_request_id"),
            "latency_seconds": record.get("latency_seconds"),
            "audit_output": record.get("audit_output"),
            "input_hash": record.get("input_hash"),
        }


def build_validator_audit_evidence(
    *,
    validator_uid: Optional[int],
    validator_hotkey: str,
    forward_count: int,
    dataset_hash: str,
    provider_stats: Mapping[str, Any],
    competition_rows: list[Mapping[str, Any]],
    chunk_count: int,
    total_hands: int,
    human_chunk_count: int,
    bot_chunk_count: int,
    suspicion_summary: Mapping[str, Any],
    compliance_summary: Mapping[str, Any],
    served_chunk_summary: Mapping[str, Any],
    max_rows: int,
) -> Dict[str, Any]:
    sorted_rows = sorted(
        [dict(row) for row in competition_rows],
        key=lambda row: (-float(row.get("reward", 0.0) or 0.0), int(row.get("uid", 0) or 0)),
    )
    trimmed_rows = []
    for row in sorted_rows[: max(1, int(max_rows))]:
        trimmed_rows.append(
            {
                "uid": row.get("uid"),
                "hotkey": row.get("hotkey"),
                "model_name": row.get("model_name"),
                "model_version": row.get("model_version"),
                "repo_url": row.get("repo_url"),
                "repo_commit": row.get("repo_commit"),
                "manifest_digest": row.get("manifest_digest"),
                "reward": row.get("reward"),
                "ap_score": row.get("ap_score"),
                "bot_recall": row.get("bot_recall"),
                "human_safety_penalty": row.get("human_safety_penalty"),
                "coverage_rate": row.get("coverage_rate"),
                "latency_mean_seconds": row.get("latency_mean_seconds"),
                "sample_count": row.get("sample_count"),
            }
        )

    evidence = {
        "validator_uid": validator_uid,
        "validator_hotkey": validator_hotkey,
        "forward_count": int(forward_count),
        "dataset_hash": dataset_hash,
        "provider_stats": {
            "competition_epoch_id": provider_stats.get("competition_epoch_id"),
            "competition_epoch_start": provider_stats.get("competition_epoch_start"),
            "competition_epoch_end": provider_stats.get("competition_epoch_end"),
            "competition_settlement_mode": provider_stats.get("competition_settlement_mode"),
            "active_window_start": provider_stats.get("active_window_start"),
            "active_window_end": provider_stats.get("active_window_end"),
            "active_chunk_id": provider_stats.get("active_chunk_id"),
            "active_chunk_hash": provider_stats.get("active_chunk_hash"),
        },
        "batch_summary": {
            "chunk_count": int(chunk_count),
            "total_hands": int(total_hands),
            "human_chunk_count": int(human_chunk_count),
            "bot_chunk_count": int(bot_chunk_count),
        },
        "integrity_summary": {
            "suspicion": dict(suspicion_summary or {}),
            "compliance": dict(compliance_summary or {}),
            "served_chunks": dict(served_chunk_summary or {}),
        },
        "top_rows": trimmed_rows,
    }
    evidence["evidence_hash"] = _sha256_json(evidence)
    return evidence
