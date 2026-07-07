"""
Standard JSON structure for Poker44 poker hands.

Validators receive batches of hands already structured in this format; miners
consume arrays of these hand JSON objects (one array per player) to decide if
the player is a bot.
"""

from __future__ import annotations

from typing import Any, Mapping

from poker44.core.models import HandHistory

# Example hand encoded in the Poker44 canonical JSON structure.
V0_JSON_HAND: dict[str, Any] = {
    "metadata": {
        "game_type": "Hold'em",
        "limit_type": "No Limit",
        "max_seats": 6,
        "hero_seat": 1,
        "hand_ended_on_street": "flop",
        "button_seat": 1,
        "sb": 0.02,
        "bb": 0.05,
        "ante": 0.0,
        "rng_seed_commitment": None,
    },
    "players": [
        {"player_uid": "DemoPlayer1", "seat": 1, "starting_stack": 5.0, "hole_cards": ["Ah", "Kh"], "showed_hand": True},
        {"player_uid": "DemoPlayer2", "seat": 2, "starting_stack": 5.16, "hole_cards": ["9c", "9d"], "showed_hand": False},
        {"player_uid": "DemoPlayer3", "seat": 3, "starting_stack": 13.57, "hole_cards": ["8s", "7s"], "showed_hand": None},
        {"player_uid": "DemoPlayer4", "seat": 4, "starting_stack": 5.0, "hole_cards": ["Qd", "Js"], "showed_hand": None},
        {"player_uid": "DemoPlayer5", "seat": 5, "starting_stack": 5.19, "hole_cards": ["Ad", "5h"], "showed_hand": None},
        {"player_uid": "DemoPlayer6", "seat": 6, "starting_stack": 8.86, "hole_cards": ["Tc", "2c"], "showed_hand": None},
    ],
    "streets": [
        {"street": "flop", "board_cards": ["7s", "Jd", "Ad"]},
    ],
    "actions": [
        {"action_id": "1", "street": "preflop", "actor_seat": 2, "action_type": "small_blind", "amount": 0.02, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.4, "pot_before": 0.0, "pot_after": 0.02},
        {"action_id": "2", "street": "preflop", "actor_seat": 3, "action_type": "big_blind", "amount": 0.05, "raise_to": None, "call_to": None, "normalized_amount_bb": 1.0, "pot_before": 0.02, "pot_after": 0.07},
        {"action_id": "3", "street": "preflop", "actor_seat": 4, "action_type": "raise", "amount": 0.15, "raise_to": 0.2, "call_to": None, "normalized_amount_bb": 3.0, "pot_before": 0.07, "pot_after": 0.22},
        {"action_id": "4", "street": "preflop", "actor_seat": 1, "action_type": "raise", "amount": 0.45, "raise_to": 0.65, "call_to": None, "normalized_amount_bb": 9.0, "pot_before": 0.22, "pot_after": 0.67},
        {"action_id": "5", "street": "preflop", "actor_seat": 2, "action_type": "fold", "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0, "pot_before": 0.67, "pot_after": 0.67},
        {"action_id": "6", "street": "preflop", "actor_seat": 3, "action_type": "fold", "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0, "pot_before": 0.67, "pot_after": 0.67},
        {"action_id": "7", "street": "preflop", "actor_seat": 4, "action_type": "call", "amount": 0.3, "raise_to": None, "call_to": 0.65, "normalized_amount_bb": 6.0, "pot_before": 0.67, "pot_after": 0.97},
        {"action_id": "8", "street": "flop", "actor_seat": 4, "action_type": "check", "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0, "pot_before": 0.97, "pot_after": 0.97},
        {"action_id": "9", "street": "flop", "actor_seat": 1, "action_type": "bet", "amount": 0.3, "raise_to": None, "call_to": None, "normalized_amount_bb": 6.0, "pot_before": 0.97, "pot_after": 1.27},
        {"action_id": "10", "street": "flop", "actor_seat": 4, "action_type": "fold", "amount": 0.0, "raise_to": None, "call_to": None, "normalized_amount_bb": 0.0, "pot_before": 1.27, "pot_after": 1.27},
    ],
    "outcome": {
        "winners": ["miiguelik"],
        "payouts": {"miiguelik": 0.91},
        "total_pot": 0.97,
        "rake": 0.06,
        "result_reason": "fold",
        "showdown": False,
    },
    "label": "human",
}


def from_standard_json(payload: Mapping[str, Any]) -> HandHistory:
    """
    Convert a structured Poker44 hand JSON payload into a HandHistory object.
    """
    return HandHistory.from_payload(payload)
