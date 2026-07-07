"""Helpers to build a public network snapshot from validator metagraph state."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from poker44.utils.misc import ttl_get_block


def _scalar(value: Any, default: Any = 0) -> Any:
    try:
        if hasattr(value, "item"):
            return value.item()
    except Exception:
        pass
    return value if value is not None else default


def _number_string(value: Any) -> str:
    raw = _scalar(value, 0)
    try:
        return str(float(raw))
    except Exception:
        return "0"


def _int_or_none(value: Any) -> int | None:
    raw = _scalar(value, None)
    if raw is None:
        return None
    try:
        return int(raw)
    except Exception:
        return None


def _bool(value: Any) -> bool:
    raw = _scalar(value, False)
    try:
        return bool(raw)
    except Exception:
        return False


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def collect_network_snapshot(validator: Any) -> dict[str, Any]:
    metagraph = validator.metagraph
    current_block = int(ttl_get_block(validator))
    burn_uid = 0

    rows: list[dict[str, Any]] = []
    active_neurons = 0
    validators_with_permit = 0
    active_validators = 0
    active_miners = 0

    total_neurons = len(getattr(metagraph, "hotkeys", []) or [])

    for uid in range(total_neurons):
        active = _bool(getattr(metagraph, "active", [False] * total_neurons)[uid])
        validator_permit = _bool(
            getattr(metagraph, "validator_permit", [False] * total_neurons)[uid]
        )
        updated_at = _int_or_none(
            getattr(metagraph, "last_update", [None] * total_neurons)[uid]
        )
        updated_blocks = None
        if updated_at is not None:
            updated_blocks = max(current_block - updated_at, 0)

        axon = None
        try:
            axon_info = metagraph.axons[uid]
            axon = {
                "ip": _text(getattr(axon_info, "ip", None)) or None,
                "port": _int_or_none(getattr(axon_info, "port", None)),
            }
        except Exception:
            axon = {"ip": None, "port": None}

        row = {
            "uid": uid,
            "hotkey": _text(metagraph.hotkeys[uid]) if uid < len(metagraph.hotkeys) else "",
            "coldkey": _text(metagraph.coldkeys[uid]) if uid < len(getattr(metagraph, "coldkeys", [])) else "",
            "active": active,
            "validator_permit": validator_permit,
            "updated_blocks": updated_blocks,
            "rank": _scalar(getattr(metagraph, "R", [None] * total_neurons)[uid], None),
            "emission": _number_string(getattr(metagraph, "E", [0] * total_neurons)[uid]),
            "incentive": _number_string(getattr(metagraph, "I", [0] * total_neurons)[uid]),
            "dividends": _number_string(getattr(metagraph, "D", [0] * total_neurons)[uid]),
            "validator_trust": _number_string(getattr(metagraph, "Tv", [0] * total_neurons)[uid]),
            "consensus": _number_string(getattr(metagraph, "C", [0] * total_neurons)[uid]),
            "total_alpha_stake": _number_string(getattr(metagraph, "S", [0] * total_neurons)[uid]),
            "stake": _number_string(getattr(metagraph, "S", [0] * total_neurons)[uid]),
            "axon": axon,
        }
        rows.append(row)

        if active:
            active_neurons += 1
            if validator_permit:
                active_validators += 1
            elif uid != burn_uid:
                active_miners += 1
        if validator_permit:
            validators_with_permit += 1

    burn_row = next((row for row in rows if row["uid"] == burn_uid), None)

    return {
        "status": "running",
        "hotkey": validator.wallet.hotkey.ss58_address,
        "validator_uid": validator.resolve_uid(validator.wallet.hotkey.ss58_address),
        "version": getattr(validator, "version", ""),
        "deploy_version": getattr(validator, "deploy_version", ""),
        "netuid": int(validator.config.netuid),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "runtime": getattr(validator, "runtime_info", {}),
        "subnet": {
            "netuid": int(validator.config.netuid),
            "block_number": current_block,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "total_neurons": total_neurons,
            "active_neurons": active_neurons,
            "validators_with_permit": validators_with_permit,
            "active_validators": active_validators,
            "active_miners": active_miners,
            "burn_uid": burn_uid,
            "incentive_burn": burn_row["incentive"] if burn_row else "0",
        },
        "neurons": rows,
    }
