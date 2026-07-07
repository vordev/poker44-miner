import unittest

from poker44.validator.payload_view import prepare_hand_for_miner


class SanitizationFocusSeatTests(unittest.TestCase):
    def test_preserves_hero_seat_for_canonical_eval_hands(self):
        payload = {
            "metadata": {
                "game_type": "Hold'em",
                "limit_type": "No Limit",
                "max_seats": 2,
                "hero_seat": 2,
                "hand_ended_on_street": "",
                "button_seat": 0,
                "sb": 0.01,
                "bb": 0.02,
                "ante": 0.0,
                "rng_seed_commitment": None,
            },
            "players": [
                {
                    "player_uid": "seat_1",
                    "seat": 1,
                    "starting_stack": 10.0,
                    "hole_cards": None,
                    "showed_hand": False,
                },
                {
                    "player_uid": "seat_2",
                    "seat": 2,
                    "starting_stack": 10.0,
                    "hole_cards": None,
                    "showed_hand": False,
                },
            ],
            "streets": [],
            "actions": [
                {
                    "action_id": "1",
                    "street": "preflop",
                    "actor_seat": 1,
                    "action_type": "call",
                    "amount": 0.1,
                    "raise_to": None,
                    "call_to": 0.1,
                    "normalized_amount_bb": 5.0,
                    "pot_before": 0.1,
                    "pot_after": 0.2,
                }
            ],
            "outcome": {
                "winners": [],
                "payouts": {},
                "total_pot": 0.2,
                "rake": 0.0,
                "result_reason": "fold",
                "showdown": False,
            },
        }

        prepared = prepare_hand_for_miner(payload)

        self.assertEqual(prepared["metadata"]["hero_seat"], 2)

    def test_canonicalizes_eval_hands_and_drops_noop_other_actions(self):
        payload = {
            "schema": "poker44_eval_hand_v1",
            "label": "bot",
            "metadata": {
                "game_type": "Hold'em",
                "limit_type": "No Limit",
                "max_seats": 2,
                "hero_seat": 1,
                "button_seat": 2,
                "bb": 0.02,
            },
            "players": [
                {"player_uid": "seat_1", "seat": 1, "starting_stack": 8.0},
                {"player_uid": "seat_2", "seat": 2, "starting_stack": 8.0},
            ],
            "streets": [],
            "actions": [
                {
                    "action_id": "1",
                    "street": "preflop",
                    "actor_seat": 1,
                    "action_type": "other",
                    "amount": 0.0,
                    "raise_to": None,
                    "call_to": None,
                    "normalized_amount_bb": 0.0,
                    "pot_before": 0.1,
                    "pot_after": 0.1,
                },
                {
                    "action_id": "2",
                    "street": "preflop",
                    "actor_seat": 2,
                    "action_type": "other",
                    "amount": 0.1,
                    "raise_to": None,
                    "call_to": 0.1,
                    "normalized_amount_bb": 5.0,
                    "pot_before": 0.1,
                    "pot_after": 0.2,
                },
            ],
            "outcome": {"showdown": False},
        }

        prepared = prepare_hand_for_miner(payload)

        self.assertNotIn("label", prepared)
        self.assertEqual(len(prepared["actions"]), 12)
        self.assertEqual(prepared["actions"][0]["action_type"], "call")
        self.assertTrue(all(action["action_type"] != "other" for action in prepared["actions"]))

    def test_drops_forced_actions_and_canonicalizes_all_in(self):
        payload = {
            "metadata": {
                "game_type": "Hold'em",
                "limit_type": "No Limit",
                "max_seats": 2,
                "hero_seat": 1,
                "button_seat": 2,
                "bb": 0.02,
            },
            "players": [
                {"player_uid": "seat_1", "seat": 1, "starting_stack": 8.0},
                {"player_uid": "seat_2", "seat": 2, "starting_stack": 8.0},
            ],
            "streets": [],
            "actions": [
                {
                    "action_id": "1",
                    "street": "preflop",
                    "actor_seat": 1,
                    "action_type": "small_blind",
                    "amount": 0.01,
                    "raise_to": None,
                    "call_to": None,
                    "normalized_amount_bb": 0.5,
                    "pot_before": 0.0,
                    "pot_after": 0.01,
                },
                {
                    "action_id": "2",
                    "street": "preflop",
                    "actor_seat": 2,
                    "action_type": "big_blind",
                    "amount": 0.02,
                    "raise_to": None,
                    "call_to": None,
                    "normalized_amount_bb": 1.0,
                    "pot_before": 0.01,
                    "pot_after": 0.03,
                },
                {
                    "action_id": "3",
                    "street": "preflop",
                    "actor_seat": 1,
                    "action_type": "all_in",
                    "amount": 0.16,
                    "raise_to": 0.16,
                    "call_to": None,
                    "normalized_amount_bb": 8.0,
                    "pot_before": 0.03,
                    "pot_after": 0.19,
                },
            ],
            "outcome": {"showdown": False},
        }

        prepared = prepare_hand_for_miner(payload)

        self.assertTrue(
            all(
                action["action_type"] not in {"small_blind", "big_blind", "ante", "all_in"}
                for action in prepared["actions"]
            )
        )
        self.assertEqual(prepared["actions"][0]["action_type"], "raise")


if __name__ == "__main__":
    unittest.main()
