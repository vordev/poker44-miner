"""
Core data models used within the Poker44 subnet.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Mapping, Optional, Sequence


@dataclass(frozen=True)
class PlayerProfile:
    """Represents a player in the Poker44 dataset."""

    player_uid: str
    seat: Optional[int] = None
    starting_stack: Optional[float] = None
    hole_cards: Optional[Sequence[str]] = None
    showed_hand: Optional[bool] = None
    ending_stack: Optional[float] = None
    is_bot: Optional[bool] = None
    bot_family_id: Optional[str] = None
    bot_version: Optional[str] = None
    human_verification_tier: Optional[int] = None
    device_fingerprint_hash: Optional[str] = None
    ip_prefix_bucket: Optional[str] = None
    user_agent_family: Optional[str] = None
    timezone_offset_bucket: Optional[str] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "PlayerProfile":
        return cls(
            player_uid=str(payload.get("player_uid")),
            seat=payload.get("seat"),
            starting_stack=payload.get("starting_stack"),
            hole_cards=payload.get("hole_cards"),
            showed_hand=payload.get("showed_hand"),
            ending_stack=payload.get("ending_stack"),
            is_bot=payload.get("is_bot"),
            bot_family_id=payload.get("bot_family_id"),
            bot_version=payload.get("bot_version"),
            human_verification_tier=payload.get("human_verification_tier"),
            device_fingerprint_hash=payload.get("device_fingerprint_hash"),
            ip_prefix_bucket=payload.get("ip_prefix_bucket"),
            user_agent_family=payload.get("user_agent_family"),
            timezone_offset_bucket=payload.get("timezone_offset_bucket"),
        )

    def to_payload(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "player_uid": self.player_uid,
            "seat": self.seat,
            "starting_stack": self.starting_stack,
            "hole_cards": self.hole_cards,
            "showed_hand": self.showed_hand,
        }
        if self.ending_stack is not None:
            data["ending_stack"] = self.ending_stack
        if self.is_bot is not None:
            data["is_bot"] = self.is_bot
        return data


@dataclass(frozen=True)
class ActionEvent:
    """Canonical append-only event for a poker action."""

    action_id: str
    hand_id: str
    street: str
    actor_seat: int
    action_type: str
    amount: float
    raise_to: Optional[float]
    call_to: Optional[float]
    normalized_amount_bb: float
    pot_before: float
    pot_after: float
    timestamp_action: datetime
    decision_time_ms: Optional[int] = None
    decision_start_ts: Optional[datetime] = None
    action_ts: Optional[datetime] = None

    @classmethod
    def from_payload(cls, payload: Mapping[str, object], hand_id: str) -> "ActionEvent":
        ts = payload.get("timestamp_action")
        try:
            parsed_ts = datetime.fromisoformat(ts) if isinstance(ts, str) else datetime.utcfromtimestamp(float(ts))
        except Exception:
            parsed_ts = datetime.utcnow()
        def _parse_dt(value):
            if isinstance(value, str):
                try:
                    return datetime.fromisoformat(value)
                except Exception:
                    return None
            return None

        return cls(
            action_id=str(payload.get("action_id") or payload.get("id")),
            hand_id=hand_id,
            street=str(payload.get("street") or "preflop"),
            actor_seat=int(payload.get("actor_seat") or 0),
            action_type=str(payload.get("action_type") or ""),
            amount=float(payload.get("amount") or 0.0),
            raise_to=None if payload.get("raise_to") is None else float(payload.get("raise_to")),
            call_to=None if payload.get("call_to") is None else float(payload.get("call_to")),
            normalized_amount_bb=float(payload.get("normalized_amount_bb") or 0.0),
            pot_before=float(payload.get("pot_before") or 0.0),
            pot_after=float(payload.get("pot_after") or 0.0),
            timestamp_action=parsed_ts,
            decision_time_ms=payload.get("decision_time_ms"),
            decision_start_ts=_parse_dt(payload.get("decision_start_ts")),
            action_ts=_parse_dt(payload.get("action_ts")),
        )

    def to_payload(self) -> Dict[str, object]:
        return {
            "action_id": self.action_id,
            "street": self.street,
            "actor_seat": self.actor_seat,
            "action_type": self.action_type,
            "amount": self.amount,
            "raise_to": self.raise_to,
            "call_to": self.call_to,
            "normalized_amount_bb": self.normalized_amount_bb,
            "pot_before": self.pot_before,
            "pot_after": self.pot_after,
        }


@dataclass(frozen=True)
class StreetState:
    street: str
    board_cards: Sequence[str]

    def to_payload(self) -> Dict[str, object]:
        return {"street": self.street, "board_cards": list(self.board_cards)}


@dataclass(frozen=True)
class HandIntegrity:
    """Behavioral and context signals that improve bot detection."""

    decision_times_ms: Sequence[int] = field(default_factory=list)
    timebank_used: bool = False
    auto_actions: Sequence[str] = field(default_factory=list)
    disconnect_events: int = 0
    reconnect_events: int = 0
    client_latency_bucket: Optional[str] = None
    session_id: Optional[str] = None
    tables_open_count: Optional[str] = None
    hands_per_minute: Optional[str] = None
    format_flags: Optional[str] = None


@dataclass(frozen=True)
class HandMetadata:
    game_type: str = ""
    limit_type: str = ""
    max_seats: int = 0
    hero_seat: int = 0
    hand_ended_on_street: str = ""
    button_seat: int = 0
    sb: float = 0.0
    bb: float = 0.0
    ante: float = 0.0
    rng_seed_commitment: Optional[str] = None

    def to_payload(self) -> Dict[str, object]:
        return {
            "game_type": self.game_type,
            "limit_type": self.limit_type,
            "max_seats": self.max_seats,
            "hero_seat": self.hero_seat,
            "hand_ended_on_street": self.hand_ended_on_street,
            "button_seat": self.button_seat,
            "sb": self.sb,
            "bb": self.bb,
            "ante": self.ante,
            "rng_seed_commitment": self.rng_seed_commitment,
        }


@dataclass(frozen=True)
class HandOutcome:
    winners: Sequence[str]
    payouts: Mapping[str, float]
    total_pot: float
    rake: float
    result_reason: str
    showdown: bool
    hole_cards: Optional[Mapping[str, Sequence[str]]] = None

    def to_payload(self) -> Dict[str, object]:
        data: Dict[str, object] = {
            "winners": list(self.winners),
            "payouts": dict(self.payouts),
            "total_pot": self.total_pot,
            "rake": self.rake,
            "result_reason": self.result_reason,
            "showdown": self.showdown,
        }
        if self.hole_cards is not None:
            data["hole_cards"] = self.hole_cards
        return data


@dataclass
class HandHistory:
    """Complete description of a poker hand with a ground-truth label."""

    metadata: HandMetadata
    participants: List[PlayerProfile]
    streets: List[StreetState]
    actions: List[ActionEvent]
    outcome: HandOutcome
    integrity: Optional[HandIntegrity] = None
    label_flag: Optional[bool] = None

    @property
    def label(self) -> bool:
        if self.label_flag is not None:
            return bool(self.label_flag)
        # Fallback: infer from participant bot flags if provided.
        for player in self.participants:
            if player.is_bot is not None:
                return bool(player.is_bot)
        return False

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "HandHistory":
        meta_raw = payload.get("metadata") or {}
        metadata = HandMetadata(
            game_type=str(meta_raw.get("game_type") or ""),
            limit_type=str(meta_raw.get("limit_type") or ""),
            max_seats=int(meta_raw.get("max_seats") or 0),
            hero_seat=int(meta_raw.get("hero_seat") or 0),
            hand_ended_on_street=str(meta_raw.get("hand_ended_on_street") or ""),
            button_seat=int(meta_raw.get("button_seat") or 0),
            sb=float(meta_raw.get("sb") or 0.0),
            bb=float(meta_raw.get("bb") or 0.0),
            ante=float(meta_raw.get("ante") or 0.0),
            rng_seed_commitment=meta_raw.get("rng_seed_commitment"),
        )
        participants = [
            PlayerProfile.from_payload(player)
            for player in (payload.get("players") or [])
        ]
        streets = [
            StreetState(
                street=str(street.get("street") or street.get("name")),
                board_cards=street.get("board_cards") or [],
            )
            for street in (payload.get("streets") or [])
        ]
        actions = [
            ActionEvent.from_payload(action, hand_id="")
            for action in (payload.get("actions") or [])
        ]
        outcome_raw = payload.get("outcome") or {}
        outcome = HandOutcome(
            winners=outcome_raw.get("winners") or [],
            payouts=outcome_raw.get("payouts") or {},
            total_pot=float(outcome_raw.get("total_pot") or 0.0),
            rake=float(outcome_raw.get("rake") or 0.0),
            result_reason=str(outcome_raw.get("result_reason") or outcome_raw.get("hand_result_reason") or ""),
            showdown=bool(outcome_raw.get("showdown") or False),
        )
        # Optional explicit label: "human" or "AI"
        label_raw = payload.get("label")
        label_flag = None
        if isinstance(label_raw, str):
            lowered = label_raw.strip().lower()
            if lowered == "human":
                label_flag = False
            elif lowered in ("ai", "bot"):
                label_flag = True
        elif isinstance(label_raw, (bool, int)):
            label_flag = bool(label_raw)

        return cls(
            metadata=metadata,
            participants=participants,
            streets=streets,
            actions=actions,
            outcome=outcome,
            label_flag=label_flag,
        )

    def to_payload(self) -> Dict[str, object]:
        payload: Dict[str, object] = {
            "metadata": self.metadata.to_payload(),
            "players": [p.to_payload() for p in self.participants],
            "streets": [s.to_payload() for s in self.streets],
            "actions": [a.to_payload() for a in self.actions],
            "outcome": self.outcome.to_payload(),
        }
        if self.label_flag is not None:
            payload["label"] = "bot" if self.label_flag else "human"
        return payload


@dataclass
class Score:
    uid: int
    value: float
    debug: Mapping[str, float] = field(default_factory=dict)


@dataclass
class Receipt:
    """Validator attestation summary recorded for auditors."""

    cycle: int
    timestamp: datetime
    scores: Sequence[Score]
    hands_processed: int
    outliers: Sequence[int] = field(default_factory=list)


@dataclass
class LabeledHandBatch:
    """Batch of hands for one player with a ground-truth flag."""

    hands: List[HandHistory]
    is_human: bool  # API sends 0 (human) or 1 (bot); bool(int) -> True when bot
