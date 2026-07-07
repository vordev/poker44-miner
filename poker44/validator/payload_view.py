"""Utilities for producing miner-visible Poker44 payloads."""

from __future__ import annotations

import hashlib
from typing import Any, Dict, List, Tuple

_LEAKAGE_KEYS = {
    "label",
    "label_flag",
    "is_bot",
    "bot_family_id",
    "bot_version",
}
_MINER_ACTION_WINDOW = 12
_MINER_ACTION_WINDOW_MIN = 5
_MINER_ACTION_WINDOW_MAX = 8
_DEFAULT_MAX_SEATS = 6
_VISIBLE_SB = 0.01
_VISIBLE_BB = 0.02
_VISIBLE_ANTE = 0.0
_MAX_NORMALIZED_STACK_BB = 500.0
_MAX_NORMALIZED_ACTION_BB = 200.0
_MAX_NORMALIZED_POT_BB = 1000.0
_VISIBLE_BB_BUCKETS = (
    0.0,
    0.5,
    1.0,
    1.5,
    2.0,
    3.0,
    4.0,
    6.0,
    8.0,
    12.0,
    16.0,
    24.0,
    36.0,
    56.0,
    84.0,
    126.0,
)
_ALLOWED_ACTION_TYPES = {
    "small_blind",
    "big_blind",
    "ante",
    "check",
    "call",
    "bet",
    "raise",
    "fold",
}


def _round_bounded(value: float, *, lower: float = 0.0, upper: float) -> float:
    return round(max(lower, min(upper, float(value))), 2)


def _to_bb_units(value: Any, bb: float, *, upper: float) -> float:
    try:
        numeric = float(value or 0.0)
    except (TypeError, ValueError):
        numeric = 0.0
    if bb <= 0:
        return 0.0
    return _round_bounded(numeric / bb, upper=upper)


def _from_bb_units(bb_value: float, *, visible_bb: float = _VISIBLE_BB) -> float:
    return round(max(0.0, float(bb_value)) * visible_bb, 4)


def _sanitize_seat(value: Any, *, max_seats: int) -> int:
    try:
        seat = int(value)
    except (TypeError, ValueError):
        return 0
    return seat if 1 <= seat <= max_seats else 0


def _sanitize_action_type(value: Any) -> str:
    action_type = str(value or "").strip().lower()
    if action_type in _ALLOWED_ACTION_TYPES:
        return action_type
    if "raise" in action_type:
        return "raise"
    if "bet" in action_type:
        return "bet"
    if "call" in action_type:
        return "call"
    if "check" in action_type:
        return "check"
    if "fold" in action_type or action_type == "muck":
        return "fold"
    return ""


def _stable_bucket_noise(seed_parts: List[str]) -> float:
    digest = hashlib.sha256("|".join(seed_parts).encode("utf-8", errors="ignore")).digest()
    return ((digest[1] / 255.0) * 2.0) - 1.0


def _coarse_bb_value(bb_value: float, *, seed_parts: List[str]) -> float:
    value = max(0.0, float(bb_value))
    if value <= 0:
        return 0.0
    nearest = min(_VISIBLE_BB_BUCKETS[1:], key=lambda bucket: abs(bucket - value))
    noise = _stable_bucket_noise(seed_parts)
    if nearest <= 1.5:
        adjusted = nearest + noise * 0.08
    elif nearest <= 8.0:
        adjusted = nearest + noise * 0.22
    else:
        adjusted = nearest + noise * max(0.35, nearest * 0.05)
    return round(max(0.0, min(_VISIBLE_BB_BUCKETS[-1], adjusted)), 2)


def _build_seat_alias_map(
    players_raw: List[Dict[str, Any]],
    actions_raw: List[Dict[str, Any]],
    *,
    max_seats: int,
) -> Dict[int, int]:
    seat_order: List[int] = []

    def _push(raw: Any) -> None:
        seat = _sanitize_seat(raw, max_seats=max_seats)
        if seat > 0 and seat not in seat_order:
            seat_order.append(seat)

    for action in actions_raw:
        if isinstance(action, dict):
            _push(action.get("actor_seat"))
    for player in players_raw:
        if isinstance(player, dict):
            _push(player.get("seat"))
    for seat in range(1, max_seats + 1):
        _push(seat)

    return {seat: idx + 1 for idx, seat in enumerate(seat_order)}


def _resolve_action_type(
    value: Any,
    *,
    amount_bb: float,
    raise_to_bb: float,
    call_to_bb: float,
    pot_before_bb: float,
    pot_after_bb: float,
) -> str:
    direct = _sanitize_action_type(value)
    if direct:
        return direct
    if raise_to_bb > 0:
        return "raise"
    if call_to_bb > 0:
        return "call"
    if amount_bb > 0:
        return "bet" if pot_after_bb > pot_before_bb else "call"
    if pot_after_bb <= pot_before_bb:
        return "check"
    return "call"


def strip_private_fields(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: Dict[str, Any] = {}
        for key, item in value.items():
            if key in _LEAKAGE_KEYS:
                continue
            cleaned[key] = strip_private_fields(item)
        return cleaned
    if isinstance(value, list):
        return [strip_private_fields(item) for item in value]
    return value


def _deterministic_window_size(seed_parts: List[str]) -> int:
    seed = "|".join(seed_parts).encode("utf-8", errors="ignore")
    digest = hashlib.sha256(seed).digest()
    span = _MINER_ACTION_WINDOW_MAX - _MINER_ACTION_WINDOW_MIN + 1
    return _MINER_ACTION_WINDOW_MIN + (digest[0] % span)


def _sample_visible_indices(
    total: int,
    *,
    window_size: int,
    seed_parts: List[str],
    actions: Optional[List[Dict[str, Any]]] = None,
) -> List[int]:
    if total <= 1:
        return [0] * max(1, window_size)
    if total <= window_size:
        return list(range(total))

    seed = "|".join(seed_parts).encode("utf-8", errors="ignore")

    def _sort_key(index: int, extra: str = "") -> bytes:
        return hashlib.sha256(seed + f":{index}:{extra}".encode("utf-8")).digest()

    picked = {0, total - 1}

    if actions:
        street_buckets: Dict[str, List[int]] = {}
        for idx in range(1, total - 1):
            action = actions[idx] if idx < len(actions) else {}
            street = str(action.get("street", "") or "preflop").lower()
            street_buckets.setdefault(street, []).append(idx)
        for street in sorted(street_buckets.keys()):
            if len(picked) >= window_size:
                break
            ordered = sorted(street_buckets[street], key=lambda idx: _sort_key(idx, street))
            if ordered:
                picked.add(ordered[0])

    middle = [idx for idx in range(1, total - 1) if idx not in picked]
    ordered_middle = sorted(middle, key=_sort_key)
    for idx in ordered_middle:
        if len(picked) >= window_size:
            break
        picked.add(idx)
    return sorted(picked)


def _collapse_visible_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    collapsed: List[Dict[str, Any]] = []
    previous: Optional[Dict[str, Any]] = None
    for action in actions:
        action_type = _sanitize_action_type(action.get("action_type"))
        zero_money = float(action.get("normalized_amount_bb", 0.0) or 0.0) <= 0.0
        pot_before = float(action.get("pot_before", 0.0) or 0.0)
        pot_after = float(action.get("pot_after", 0.0) or 0.0)
        duplicate_noop = (
            previous is not None
            and zero_money
            and pot_after <= pot_before
            and action_type in {"check", "fold"}
            and _sanitize_action_type(previous.get("action_type")) == action_type
            and int(previous.get("actor_seat", 0) or 0) == int(action.get("actor_seat", 0) or 0)
            and str(previous.get("street", "")) == str(action.get("street", ""))
            and float(previous.get("normalized_amount_bb", 0.0) or 0.0) <= 0.0
            and float(previous.get("pot_after", 0.0) or 0.0)
            <= float(previous.get("pot_before", 0.0) or 0.0)
        )
        if duplicate_noop:
            continue
        collapsed.append(action)
        previous = action
    return collapsed


def build_miner_payload_hand(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Keep behaviorally useful structure while suppressing direct identity fields."""
    cleaned = strip_private_fields(hand_payload)
    if not isinstance(cleaned, dict):
        return {}

    metadata = cleaned.get("metadata") if isinstance(cleaned.get("metadata"), dict) else {}
    players_raw = cleaned.get("players") if isinstance(cleaned.get("players"), list) else []
    actions_raw = cleaned.get("actions") if isinstance(cleaned.get("actions"), list) else []
    outcome = cleaned.get("outcome") if isinstance(cleaned.get("outcome"), dict) else {}

    max_seats = max(
        _DEFAULT_MAX_SEATS,
        _sanitize_seat(metadata.get("max_seats"), max_seats=10),
    )
    source_bb = float(metadata.get("bb", 0.0) or 0.0)
    seat_alias_map = _build_seat_alias_map(players_raw, actions_raw, max_seats=max_seats)
    alias_max_seats = max(2, len(seat_alias_map))

    seat_to_stack_bb: Dict[int, float] = {}
    for player in players_raw:
        if not isinstance(player, dict):
            continue
        seat_i = _sanitize_seat(player.get("seat"), max_seats=max_seats)
        if seat_i == 0:
            continue
        seat_to_stack_bb[seat_alias_map.get(seat_i, seat_i)] = _to_bb_units(
            player.get("starting_stack", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_STACK_BB,
        )

    visible_players: List[Dict[str, Any]] = [
        {
            "player_uid": f"seat_{seat_i}",
            "seat": seat_i,
            "starting_stack": _from_bb_units(starting_stack_bb),
            "hole_cards": None,
            "showed_hand": False,
        }
        for seat_i, starting_stack_bb in sorted(seat_to_stack_bb.items())
    ]

    raw_actions: List[Dict[str, Any]] = []
    for action in actions_raw:
        if not isinstance(action, dict):
            continue
        amount_bb_raw = _to_bb_units(
            action.get("amount", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_ACTION_BB,
        )
        raise_to_bb_raw = _to_bb_units(
            action.get("raise_to"),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        call_to_bb_raw = _to_bb_units(
            action.get("call_to"),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        pot_before_bb_raw = _to_bb_units(
            action.get("pot_before", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        pot_after_bb_raw = _to_bb_units(
            action.get("pot_after", 0.0),
            source_bb,
            upper=_MAX_NORMALIZED_POT_BB,
        )
        seed_base = [
            str(metadata.get("hero_seat", "")),
            str(metadata.get("max_seats", "")),
            str(action.get("street", "")),
            str(action.get("actor_seat", "")),
            str(action.get("action_id", "")),
        ]
        amount_bb = _coarse_bb_value(amount_bb_raw, seed_parts=[*seed_base, "amount"])
        raise_to_bb = _coarse_bb_value(raise_to_bb_raw, seed_parts=[*seed_base, "raise_to"])
        call_to_bb = _coarse_bb_value(call_to_bb_raw, seed_parts=[*seed_base, "call_to"])
        pot_before_bb = _coarse_bb_value(
            pot_before_bb_raw,
            seed_parts=[*seed_base, "pot_before"],
        )
        pot_after_bb = _coarse_bb_value(
            max(pot_before_bb, pot_after_bb_raw),
            seed_parts=[*seed_base, "pot_after"],
        )
        direct_action_type = _sanitize_action_type(action.get("action_type"))
        action_type = direct_action_type or _resolve_action_type(
            action.get("action_type"),
            amount_bb=amount_bb,
            raise_to_bb=raise_to_bb,
            call_to_bb=call_to_bb,
            pot_before_bb=pot_before_bb,
            pot_after_bb=pot_after_bb,
        )
        if action_type in {"small_blind", "big_blind", "ante"}:
            continue
        if direct_action_type == "" and (
            not action_type
            or (
                amount_bb <= 0
                and raise_to_bb <= 0
                and call_to_bb <= 0
                and pot_after_bb <= pot_before_bb
            )
        ):
            continue
        raw_actions.append(
            {
                "action_id": "",
                "street": str(action.get("street", "")),
                "actor_seat": seat_alias_map.get(
                    _sanitize_seat(action.get("actor_seat"), max_seats=max_seats),
                    0,
                ),
                "action_type": action_type,
                "amount": _from_bb_units(amount_bb),
                "raise_to": None if raise_to_bb <= 0 else _from_bb_units(raise_to_bb),
                "call_to": None if call_to_bb <= 0 else _from_bb_units(call_to_bb),
                "normalized_amount_bb": amount_bb,
                "pot_before": _from_bb_units(pot_before_bb),
                "pot_after": _from_bb_units(pot_after_bb),
            }
        )

    visible_actions: List[Dict[str, Any]] = []
    raw_actions = _collapse_visible_actions(raw_actions)
    if raw_actions:
        last_idx = len(raw_actions) - 1
        if len(raw_actions) == 1:
            window_size = _MINER_ACTION_WINDOW
            indices = [0] * window_size
        else:
            window_size = min(
                len(raw_actions),
                _deterministic_window_size(
                    [
                        str(metadata.get("hero_seat", "")),
                        str(metadata.get("max_seats", "")),
                        str(raw_actions[0].get("street", "")),
                        str(len(raw_actions)),
                    ]
                ),
            )
            indices = _sample_visible_indices(
                len(raw_actions),
                window_size=window_size,
                seed_parts=[
                    str(metadata.get("hero_seat", "")),
                    str(metadata.get("max_seats", "")),
                    str(raw_actions[0].get("street", "")),
                    str(len(raw_actions)),
                ],
                actions=raw_actions,
            )
        visible_actions = [dict(raw_actions[i]) for i in indices]

    for idx, action in enumerate(visible_actions, start=1):
        action["action_id"] = str(idx)

    return {
        "metadata": {
            "game_type": str(metadata.get("game_type", "")),
            "limit_type": str(metadata.get("limit_type", "")),
            "max_seats": alias_max_seats,
            "hero_seat": seat_alias_map.get(
                _sanitize_seat(metadata.get("hero_seat"), max_seats=max_seats),
                0,
            ),
            "hand_ended_on_street": "",
            "button_seat": 0,
            "sb": _VISIBLE_SB,
            "bb": _VISIBLE_BB,
            "ante": _VISIBLE_ANTE,
            "rng_seed_commitment": None,
        },
        "players": visible_players,
        "streets": [
            {
                "street": str(street.get("street", "")),
                "board_cards": [],
            }
            for street in (cleaned.get("streets") or [])
            if isinstance(street, dict)
        ],
        "actions": visible_actions,
        "outcome": {
            "winners": [],
            "payouts": {},
            "total_pot": 0.0,
            "rake": 0.0,
            "result_reason": "",
            "showdown": False,
        },
    }


def prepare_hand_for_miner(hand_payload: Dict[str, Any]) -> Dict[str, Any]:
    """Project all hands through the same miner-visible canonicalizer."""
    return build_miner_payload_hand(hand_payload)


def payload_chunk_signature(
    hands: List[Dict[str, Any]],
) -> Tuple[float, float, float, float, float, float, float, float, float]:
    """Coarse miner-visible behavior signature for chunk analysis and matching."""
    if not hands:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

    visible_hands = [build_miner_payload_hand(hand) for hand in hands]
    total_calls = 0
    total_checks = 0
    total_raises = 0
    total_folds = 0
    total_actions = 0
    total_streets = 0
    total_players = 0
    total_action_amount = 0.0
    total_action_pot_after = 0.0
    for hand in visible_hands:
        players = hand.get("players") or []
        actions = hand.get("actions") or []
        total_players += len(players)
        total_actions += len(actions)
        for action in actions:
            action_type = action.get("action_type")
            total_action_amount += float(action.get("normalized_amount_bb", 0.0) or 0.0)
            total_action_pot_after += float(action.get("pot_after", 0.0) or 0.0)
            if action_type == "call":
                total_calls += 1
            elif action_type == "check":
                total_checks += 1
            elif action_type == "raise":
                total_raises += 1
            elif action_type == "fold":
                total_folds += 1
        total_streets += len(hand.get("streets") or [])

    n = float(len(visible_hands))
    return (
        total_calls / n,
        total_checks / n,
        total_raises / n,
        total_folds / n,
        total_actions / n,
        total_streets / n,
        total_players / n,
        total_action_amount / n,
        total_action_pot_after / n,
    )
