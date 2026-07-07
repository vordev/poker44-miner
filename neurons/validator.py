# The MIT License (MIT)
# Copyright © 2023 Yuma Rao

# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated
# documentation files (the "Software"), to deal in the Software without restriction, including without limitation
# the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software,
# and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all copies or substantial portions of
# the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO
# THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL
# THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
# OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER
# DEALINGS IN THE SOFTWARE.

"""Poker44 validator entrypoint wired into the base Bittensor neuron."""
# neuron/validator.py

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Optional

import bittensor as bt
from dotenv import load_dotenv

from poker44 import __version__, VALIDATOR_DEPLOY_VERSION
from poker44.base.validator import BaseValidatorNeuron
from poker44.utils.config import config
from poker44.utils.network_snapshot import collect_network_snapshot
from poker44.utils.runtime_info import (
    build_signed_runtime_request,
    collect_runtime_info,
    post_runtime_snapshot,
    write_runtime_snapshot,
)
from poker44.utils.wandb_helper import ValidatorWandbHelper
from poker44.validator.forward import forward as forward_cycle
from poker44.validator.integrity import load_json_registry
from poker44.validator.audit import ValidatorAuditLane, build_validator_audit_evidence
from poker44.validator.runtime_provider import (
    ProviderRuntimeConfig,
    ProviderRuntimeDatasetProvider,
)

load_dotenv()
os.makedirs("./logs", exist_ok=True)
bt.logging.set_trace()
bt.logging(debug=True, trace=False, logging_dir="./logs", record_log=True)

DEFAULT_VALIDATOR_RUNTIME_REPORT_URL = (
    "https://api.poker44.net/internal/validators/runtime"
)
DEFAULT_NETWORK_SNAPSHOT_REPORT_URL = (
    "https://api.poker44.net/internal/network/snapshots"
)
DEFAULT_COMPETITION_SCORE_REPORT_URL = (
    "https://api.poker44.net/internal/competition/report-scores"
)


class Validator(BaseValidatorNeuron):
    """Poker44 validator neuron wired into the BaseValidator scaffold."""

    def __init__(self):
        cfg = config(Validator)
        self.poll_interval = int(
            os.getenv("POKER44_POLL_INTERVAL_SECONDS", str(getattr(cfg, "poll_interval_seconds", 300)))
        )
        self.reward_window = 0
        self.runtime_mode = str(
            os.getenv("POKER44_RUNTIME_MODE", "provider_runtime")
        ).strip().lower()
        if self.runtime_mode != "provider_runtime":
            raise RuntimeError(
                "Legacy local runtime has been removed. "
                "Use POKER44_RUNTIME_MODE=provider_runtime."
            )
        super().__init__(config=cfg)
        bt.logging.info(f"🚀 Poker44 Validator v{__version__} started")

        self.forward_count = 0
        self.settings = cfg
        refresh_seconds = int(os.getenv("POKER44_DATASET_REFRESH_SECONDS", str(60 * 60)))
        self.chunk_batch_size = None

        provider_runtime_cfg = ProviderRuntimeConfig.from_env(
            default_validator_id=self.wallet.hotkey.ss58_address
        )
        self.dataset_cfg = provider_runtime_cfg.public_summary()
        self.provider = ProviderRuntimeDatasetProvider(
            provider_runtime_cfg,
            wallet=self.wallet,
        )
        bt.logging.info(
            "🎯 Using provider runtime dataset provider | "
            f"api={provider_runtime_cfg.api_base_url} "
            f"validator_id={provider_runtime_cfg.validator_id}"
        )
        configured_poll_interval = getattr(cfg, "poll_interval_seconds", 30)
        self.poll_interval = int(
            os.getenv("POKER44_POLL_INTERVAL_SECONDS", str(configured_poll_interval))
        )
        self.reward_window = 0
        self.prediction_buffer = {}
        self.label_buffer = {}
        self.coverage_buffer = {}
        self.latency_buffer = {}
        self.competition_scores_payload = []
        state_dir = Path(self.config.neuron.full_path)
        self.model_manifest_path = state_dir / "model_manifests.json"
        self.compliance_registry_path = state_dir / "compliance_registry.json"
        self.suspicion_registry_path = state_dir / "suspicion_registry.json"
        self.served_chunk_registry_path = state_dir / "served_chunk_registry.json"
        self.audit_registry_path = state_dir / "audit_reports.json.enc"
        self.model_manifest_registry = load_json_registry(self.model_manifest_path)
        self.compliance_registry = load_json_registry(
            self.compliance_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.suspicion_registry = load_json_registry(
            self.suspicion_registry_path,
            default={"miners": {}, "summary": {}},
        )
        self.served_chunk_registry = load_json_registry(
            self.served_chunk_registry_path,
            default={"chunk_index": {}, "recent_cycles": [], "summary": {}},
        )
        self.audit_lane = ValidatorAuditLane.from_env(path=self.audit_registry_path)
        self.audit_summary = self.audit_lane.public_summary()
        self.wandb_helper = ValidatorWandbHelper(
            config=cfg,
            validator_uid=self.resolve_uid(self.wallet.hotkey.ss58_address),
            hotkey=self.wallet.hotkey.ss58_address,
            version=__version__,
            netuid=cfg.netuid,
        )
        self.wandb_helper.log_validator_startup(
            dataset_cfg=self.dataset_cfg,
            poll_interval=self.poll_interval,
            reward_window=self.reward_window,
        )
        self.deploy_version = VALIDATOR_DEPLOY_VERSION
        self.version = __version__
        self._write_runtime_snapshot(status="started")

    def resolve_uid(self, hotkey: str) -> Optional[int]:
        try:
            return self.metagraph.hotkeys.index(hotkey)
        except ValueError:
            return None

    @property
    def runtime_snapshot_path(self) -> Path:
        return Path(self.config.neuron.full_path) / "validator_runtime.json"

    @property
    def network_snapshot_path(self) -> Path:
        return Path(self.config.neuron.full_path) / "network_snapshot.json"

    @property
    def runtime_info(self) -> dict[str, Any]:
        info = getattr(self, "_runtime_info", None)
        if info is None:
            info = collect_runtime_info()
            self._runtime_info = info
        return info

    def _write_runtime_snapshot(self, *, status: str, extra: Optional[dict] = None) -> None:
        provider_stats = (
            getattr(self.provider, "stats", {})
            if hasattr(self, "provider") and hasattr(self.provider, "stats")
            else {}
        )
        payload = {
            "status": status,
            "validator_uid": self.resolve_uid(self.wallet.hotkey.ss58_address),
            "hotkey": self.wallet.hotkey.ss58_address,
            "version": __version__,
            "deploy_version": VALIDATOR_DEPLOY_VERSION,
            "netuid": self.config.netuid,
            "runtime_mode": getattr(self, "runtime_mode", "initializing"),
            "poll_interval": getattr(self, "poll_interval", None),
            "reward_window": 0,
            "chunk_batch_size": 0,
            "step": int(getattr(self, "step", 0)),
            "score_slots": int(len(getattr(self, "scores", []))) if hasattr(self, "scores") else 0,
            "nonzero_scores": int((getattr(self, "scores", []) != 0).sum())
            if hasattr(self, "scores")
            else 0,
            "dataset_stats": provider_stats,
            "competition_epoch_id": provider_stats.get("competition_epoch_id"),
            "competition_epoch_start": provider_stats.get("competition_epoch_start"),
            "competition_epoch_end": provider_stats.get("competition_epoch_end"),
            "competition_settlement_mode": provider_stats.get("competition_settlement_mode"),
            "active_window_start": provider_stats.get("active_window_start"),
            "active_window_end": provider_stats.get("active_window_end"),
            "active_chunk_id": provider_stats.get("active_chunk_id"),
            "active_chunk_hash": provider_stats.get("active_chunk_hash"),
            "competition_scores": list(getattr(self, "competition_scores_payload", []) or []),
            "audit": dict(getattr(self, "audit_summary", {}) or {}),
            "runtime": self.runtime_info,
        }
        if extra:
            payload.update(extra)

        write_runtime_snapshot(self.runtime_snapshot_path, payload)
        report_url = str(
            os.getenv(
                "POKER44_VALIDATOR_RUNTIME_REPORT_URL",
                DEFAULT_VALIDATOR_RUNTIME_REPORT_URL,
            )
        ).strip()
        if report_url:
            timeout_seconds = float(
                os.getenv("POKER44_VALIDATOR_RUNTIME_REPORT_TIMEOUT_SECONDS", "5")
            )
            signed_request = build_signed_runtime_request(
                wallet=self.wallet,
                url=report_url,
                payload=payload,
            )
            ok, message = post_runtime_snapshot(
                url=report_url,
                payload=payload,
                timeout_seconds=timeout_seconds,
                **signed_request,
            )
            if ok:
                bt.logging.debug(
                    f"Validator runtime snapshot reported successfully to collector: {report_url}"
                )
            else:
                bt.logging.warning(
                    "Validator runtime snapshot report failed | "
                    f"url={report_url} message={message}"
                )

        network_snapshot = collect_network_snapshot(self)
        write_runtime_snapshot(self.network_snapshot_path, network_snapshot)
        network_report_url = str(
            os.getenv(
                "POKER44_VALIDATOR_NETWORK_SNAPSHOT_REPORT_URL",
                DEFAULT_NETWORK_SNAPSHOT_REPORT_URL,
            )
        ).strip()
        if network_report_url:
            timeout_seconds = float(
                os.getenv("POKER44_VALIDATOR_NETWORK_SNAPSHOT_TIMEOUT_SECONDS", "5")
            )
            signed_request = build_signed_runtime_request(
                wallet=self.wallet,
                url=network_report_url,
                payload=network_snapshot,
            )
            ok, message = post_runtime_snapshot(
                url=network_report_url,
                payload=network_snapshot,
                timeout_seconds=timeout_seconds,
                **signed_request,
            )
            if ok:
                bt.logging.debug(
                    f"Validator network snapshot reported successfully to collector: {network_report_url}"
                )
            else:
                bt.logging.warning(
                    "Validator network snapshot report failed | "
                    f"url={network_report_url} message={message}"
                )

    def _report_competition_scores(self) -> None:
        rows = list(getattr(self, "competition_scores_payload", []) or [])
        if not rows:
            return

        provider_stats = (
            getattr(self.provider, "stats", {})
            if hasattr(self, "provider") and hasattr(self.provider, "stats")
            else {}
        )
        payload = {
            "hotkey": self.wallet.hotkey.ss58_address,
            "validator_uid": self.resolve_uid(self.wallet.hotkey.ss58_address),
            "competition_epoch_id": provider_stats.get("competition_epoch_id"),
            "competition_epoch_start": provider_stats.get("competition_epoch_start"),
            "competition_epoch_end": provider_stats.get("competition_epoch_end"),
            "active_chunk_id": provider_stats.get("active_chunk_id"),
            "active_chunk_hash": provider_stats.get("active_chunk_hash"),
            "active_window_start": provider_stats.get("active_window_start"),
            "active_window_end": provider_stats.get("active_window_end"),
            "competition_scores": rows,
        }
        epoch_id = str(payload.get("competition_epoch_id") or "").strip()
        if not epoch_id:
            bt.logging.warning(
                "Skipping competition score report because provider stats do not contain a competition epoch id."
            )
            return

        report_url = str(os.getenv("POKER44_COMPETITION_SCORE_REPORT_URL", "")).strip()
        if not report_url and self.runtime_mode == "provider_runtime":
            api_base_url = str(
                os.getenv(
                    "POKER44_EVAL_API_BASE_URL",
                    os.getenv("POKER44_PROVIDER_API_BASE_URL", ""),
                )
            ).strip().rstrip("/")
            if api_base_url:
                report_url = f"{api_base_url}/internal/competition/report-scores"
        if not report_url:
            report_url = DEFAULT_COMPETITION_SCORE_REPORT_URL
        if not report_url:
            return

        timeout_seconds = float(
            os.getenv("POKER44_COMPETITION_SCORE_REPORT_TIMEOUT_SECONDS", "5")
        )
        signed_request = build_signed_runtime_request(
            wallet=self.wallet,
            url=report_url,
            payload=payload,
        )
        ok, message = post_runtime_snapshot(
            url=report_url,
            payload=payload,
            timeout_seconds=timeout_seconds,
            **signed_request,
        )
        if ok:
            bt.logging.debug(
                f"Competition score report delivered successfully: {report_url}"
            )
        else:
            bt.logging.warning(
                "Competition score report failed | "
                f"url={report_url} message={message}"
            )

    def _record_audit_report(
        self,
        *,
        total_hands: int,
        chunk_count: int,
        human_chunk_count: int,
        bot_chunk_count: int,
    ) -> None:
        audit_lane = getattr(self, "audit_lane", None)
        if audit_lane is None:
            return

        provider_stats = (
            getattr(self.provider, "stats", {})
            if hasattr(self, "provider") and hasattr(self.provider, "stats")
            else {}
        )
        evidence = build_validator_audit_evidence(
            validator_uid=self.resolve_uid(self.wallet.hotkey.ss58_address),
            validator_hotkey=self.wallet.hotkey.ss58_address,
            forward_count=int(getattr(self, "forward_count", 0)),
            dataset_hash=str(getattr(self.provider, "dataset_hash", "") or ""),
            provider_stats=provider_stats,
            competition_rows=list(getattr(self, "competition_scores_payload", []) or []),
            chunk_count=int(chunk_count),
            total_hands=int(total_hands),
            human_chunk_count=int(human_chunk_count),
            bot_chunk_count=int(bot_chunk_count),
            suspicion_summary=(getattr(self, "suspicion_registry", {}) or {}).get("summary", {}),
            compliance_summary=(getattr(self, "compliance_registry", {}) or {}).get("summary", {}),
            served_chunk_summary=(getattr(self, "served_chunk_registry", {}) or {}).get("summary", {}),
            max_rows=int(os.getenv("POKER44_AUDIT_TOP_ROWS", "8")),
        )
        record = audit_lane.record_cycle(evidence=evidence)
        self.audit_summary = audit_lane.public_summary()
        bt.logging.info(
            "Audit lane updated | "
            f"provider={record.get('provider')} status={record.get('status')} "
            f"epoch={record.get('competition_epoch_id')} top_uid={record.get('top_uid')} "
            f"proof_verified={record.get('proof_verified')}"
        )

    async def forward(self, synapse=None):  # type: ignore[override]
        return await forward_cycle(self)

    def __del__(self) -> None:
        wandb_helper = getattr(self, "wandb_helper", None)
        if wandb_helper is not None:
            try:
                wandb_helper.finish()
            except Exception:
                pass


if __name__ == "__main__":  # pragma: no cover - manual execution
    validator = Validator()
    validator.run()
