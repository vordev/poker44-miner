"""Optional Weights & Biases helper for validator telemetry."""

from __future__ import annotations

import os
import subprocess
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import bittensor as bt


UTC = timezone.utc
REPO_ROOT = Path(__file__).resolve().parents[2]


def _get_nested_attr(obj: Any, path: str, default: Any = None) -> Any:
    current = obj
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return default
    return current


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _flatten_metrics(prefix: str, payload: Mapping[str, Any]) -> Dict[str, Any]:
    flattened: Dict[str, Any] = {}
    for key, value in payload.items():
        metric_key = f"{prefix}/{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            flattened.update(_flatten_metrics(metric_key, value))
            continue
        if isinstance(value, bool):
            flattened[metric_key] = int(value)
            continue
        if isinstance(value, (int, float, str)):
            flattened[metric_key] = value
            continue
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            if all(isinstance(item, (int, float, bool, str)) for item in value):
                flattened[metric_key] = list(value)
            else:
                flattened[f"{metric_key}_count"] = len(value)
            continue
        flattened[metric_key] = str(value)
    return flattened


def _git_commit_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _git_branch_name() -> str:
    try:
        result = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "branch", "--show-current"],
            check=True,
            capture_output=True,
            text=True,
        )
        return result.stdout.strip()
    except Exception:
        return ""


class ValidatorWandbHelper:
    """Best-effort validator telemetry logger for W&B."""

    def __init__(
        self,
        *,
        config: Any,
        validator_uid: Optional[int],
        hotkey: str,
        version: str,
        netuid: int,
    ) -> None:
        self.config = config
        self.validator_uid = validator_uid if validator_uid is not None else -1
        self.hotkey = hotkey
        self.version = version
        self.netuid = netuid
        self.enabled = False
        self.run = None
        self._wandb = None
        self._init_wandb()

    def _init_wandb(self) -> None:
        if bool(_get_nested_attr(self.config, "wandb.off", False)):
            bt.logging.info("W&B disabled via --wandb.off")
            return

        offline = bool(_get_nested_attr(self.config, "wandb.offline", False))
        api_key = os.getenv("POKER44_WANDB_API_KEY") or os.getenv("WANDB_API_KEY")
        if not offline and not api_key:
            bt.logging.debug("No W&B API key configured; validator telemetry logging disabled")
            return

        try:
            import wandb  # Imported lazily so validator still runs without the package.
        except Exception as exc:
            bt.logging.warning(f"W&B import failed; continuing without telemetry: {exc}")
            return

        try:
            project_name = (
                os.getenv("POKER44_WANDB_PROJECT")
                or _get_nested_attr(self.config, "wandb.project_name", "poker44-validators")
            )
            entity_name = (
                os.getenv("POKER44_WANDB_ENTITY")
                or _get_nested_attr(self.config, "wandb.entity", "")
                or None
            )
            notes = _get_nested_attr(self.config, "wandb.notes", "")

            os.environ.setdefault("WANDB_SILENT", "true")
            os.environ.setdefault("WANDB_QUIET", "true")
            if api_key:
                os.environ["WANDB_API_KEY"] = api_key
            if offline:
                os.environ["WANDB_MODE"] = "offline"

            ts = datetime.now(tz=UTC).strftime("%Y%m%d-%H%M%S")
            run_name = f"validator-{self.validator_uid}-{ts}"
            init_kwargs: Dict[str, Any] = {
                "project": project_name,
                "name": run_name,
                "reinit": True,
                "config": {
                    "validator_uid": self.validator_uid,
                    "validator_hotkey": self.hotkey,
                    "subnet_netuid": self.netuid,
                    "subnet_version": self.version,
                    "git_commit": _git_commit_sha(),
                    "wandb_offline": offline,
                },
                "settings": wandb.Settings(quiet=True),
            }
            if entity_name:
                init_kwargs["entity"] = entity_name
            if notes:
                init_kwargs["notes"] = notes

            self._wandb = wandb
            self.run = wandb.init(**init_kwargs)
            self.enabled = self.run is not None
            if self.enabled:
                bt.logging.info(
                    f"W&B telemetry enabled: project={project_name} run={self.run.name}"
                )
        except Exception as exc:
            bt.logging.warning(f"W&B initialization failed; continuing without telemetry: {exc}")
            self.enabled = False
            self.run = None
            self._wandb = None

    def log_payload(self, payload: Mapping[str, Any]) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.log(dict(payload))
        except Exception as exc:
            bt.logging.debug(f"W&B log skipped: {exc}")

    def log_validator_startup(
        self,
        *,
        dataset_cfg: Any,
        poll_interval: int,
        reward_window: int,
        runtime_info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        cfg_dict = asdict(dataset_cfg) if is_dataclass(dataset_cfg) else dict(dataset_cfg)
        safe_cfg = {
            "chunk_count": cfg_dict.get("chunk_count"),
            "min_hands_per_chunk": cfg_dict.get("min_hands_per_chunk"),
            "max_hands_per_chunk": cfg_dict.get("max_hands_per_chunk"),
            "human_ratio": cfg_dict.get("human_ratio"),
            "refresh_seconds": cfg_dict.get("refresh_seconds"),
            "seed_configured": bool(cfg_dict.get("seed") is not None),
        }
        payload = _flatten_metrics("validator_startup", safe_cfg)
        payload.update(
            {
                "validator_startup/poll_interval_seconds": _safe_int(poll_interval),
                "validator_startup/reward_window": _safe_int(reward_window),
                "validator_startup/timestamp": datetime.now(tz=UTC).isoformat(),
                "validator_startup/git_commit": _git_commit_sha(),
                "validator_startup/git_branch": _git_branch_name(),
            }
        )
        if runtime_info:
            payload.update(_flatten_metrics("validator_runtime", runtime_info))
        self.log_payload(payload)

    def log_dataset_state(self, *, dataset_hash: str, stats: Mapping[str, Any]) -> None:
        payload = {
            "dataset/hash": dataset_hash,
            "dataset/hash_prefix": dataset_hash[:12] if dataset_hash else "",
        }
        payload.update(_flatten_metrics("dataset", stats))
        self.log_payload(payload)

    def log_forward_summary(
        self,
        *,
        forward_count: int,
        chunk_count: int,
        total_hands: int,
        miner_count: int,
        responded_count: int,
        successful_miners: int,
        dataset_hash: str,
        dataset_stats: Mapping[str, Any],
        extra: Optional[Mapping[str, Any]] = None,
    ) -> None:
        payload: Dict[str, Any] = {
            "forward/count": _safe_int(forward_count),
            "forward/chunk_count": _safe_int(chunk_count),
            "forward/total_hands": _safe_int(total_hands),
            "forward/miner_count": _safe_int(miner_count),
            "forward/responded_count": _safe_int(responded_count),
            "forward/successful_miners": _safe_int(successful_miners),
            "forward/response_rate": (
                _safe_float(successful_miners) / max(1.0, _safe_float(miner_count))
            ),
            "forward/dataset_hash_prefix": dataset_hash[:12] if dataset_hash else "",
        }
        for key in (
            "chunk_count",
            "human_chunks",
            "bot_chunks",
            "total_hands",
            "human_hands",
            "bot_hands",
            "shortcut_rule_accuracy",
            "avg_streets_gap",
        ):
            if key in dataset_stats:
                payload[f"forward/dataset_{key}"] = dataset_stats[key]
        if extra:
            payload.update(_flatten_metrics("", extra))
        self.log_payload(payload)

    def log_reward_summary(
        self,
        *,
        reward_map: Mapping[int, float],
        metrics_map: Mapping[int, Mapping[str, Any]],
        winner_uids: Sequence[int],
        winner_rewards: Sequence[float],
    ) -> None:
        reward_values = [float(v) for v in reward_map.values()]
        ap_scores = [_safe_float(metric.get("ap_score")) for metric in metrics_map.values()]
        recalls = [_safe_float(metric.get("bot_recall")) for metric in metrics_map.values()]
        fprs = [_safe_float(metric.get("fpr")) for metric in metrics_map.values()]
        payload: Dict[str, Any] = {
            "rewards/min": min(reward_values) if reward_values else 0.0,
            "rewards/max": max(reward_values) if reward_values else 0.0,
            "rewards/mean": (
                sum(reward_values) / len(reward_values) if reward_values else 0.0
            ),
            "rewards/nonzero_count": sum(1 for value in reward_values if value > 0.0),
            "rewards/winner_count": len(winner_uids),
            "rewards/winner_uids": list(winner_uids),
            "rewards/winner_weights": [float(v) for v in winner_rewards],
            "metrics/ap_mean": sum(ap_scores) / len(ap_scores) if ap_scores else 0.0,
            "metrics/bot_recall_mean": sum(recalls) / len(recalls) if recalls else 0.0,
            "metrics/fpr_mean": sum(fprs) / len(fprs) if fprs else 0.0,
        }
        self.log_payload(payload)

    def log_set_weights_result(
        self,
        *,
        success: bool,
        message: str,
        wait_for_inclusion: bool,
        wait_for_finalization: bool,
    ) -> None:
        self.log_payload(
            {
                "set_weights/success": int(success),
                "set_weights/message": str(message)[:512],
                "set_weights/wait_for_inclusion": int(wait_for_inclusion),
                "set_weights/wait_for_finalization": int(wait_for_finalization),
            }
        )

    def log_error(self, error_type: str, error_message: str) -> None:
        self.log_payload(
            {
                "errors/type": error_type,
                "errors/message": str(error_message)[:2048],
                "errors/count": 1,
            }
        )

    def finish(self) -> None:
        if not self.enabled or self.run is None:
            return
        try:
            self.run.finish(quiet=True)
        except Exception:
            pass
