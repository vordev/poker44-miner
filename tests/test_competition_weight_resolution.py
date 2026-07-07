import unittest

import numpy as np

from poker44.base.validator import (
    BACKEND_BURN_FRACTION,
    BACKEND_KEEP_FRACTION,
    _extract_competition_weight_vector,
)


class _Provider:
    def __init__(self, payload):
        self.payload = payload

    def get_competition_settlement_weights(self):
        return self.payload


class CompetitionWeightResolutionTests(unittest.TestCase):
    def test_uses_settled_backend_vector(self):
        raw_weights, metadata = _extract_competition_weight_vector(
            provider=_Provider(
                {
                    "status": "settled",
                    "epochId": "day_2026-04-20_2000utc",
                    "sourceEpochId": "day_2026-04-19_2000utc",
                    "winnerUid": 42,
                    "weights": [{"uid": 42, "weight": 1}],
                }
            ),
            metagraph_size=128,
        )

        self.assertIsInstance(raw_weights, np.ndarray)
        self.assertEqual(metadata["weights_source"], "competition_settlement")
        self.assertEqual(metadata["settlement_winner_uid"], 42)
        self.assertEqual(int(np.count_nonzero(raw_weights)), 1)
        self.assertAlmostEqual(float(raw_weights[0]), BACKEND_BURN_FRACTION, places=6)
        self.assertAlmostEqual(float(raw_weights[42]), BACKEND_KEEP_FRACTION, places=6)
        self.assertAlmostEqual(float(raw_weights.sum()), 1.0, places=6)

    def test_uses_backend_fallback_vector_before_first_winner(self):
        raw_weights, metadata = _extract_competition_weight_vector(
            provider=_Provider(
                {
                    "status": "fallback",
                    "epochId": "day_2026-04-20_2000utc",
                    "sourceEpochId": None,
                    "winnerUid": 0,
                    "weights": [{"uid": 0, "weight": 1}],
                }
            ),
            metagraph_size=128,
        )

        self.assertIsInstance(raw_weights, np.ndarray)
        self.assertEqual(metadata["weights_source"], "competition_fallback")
        self.assertEqual(metadata["settlement_winner_uid"], 0)
        self.assertEqual(int(np.count_nonzero(raw_weights)), 1)
        self.assertAlmostEqual(float(raw_weights[0]), 1.0, places=6)

    def test_rescales_multiple_backend_winners_into_remaining_fraction(self):
        raw_weights, metadata = _extract_competition_weight_vector(
            provider=_Provider(
                {
                    "status": "runtime",
                    "epochId": "day_2026-04-20_2000utc",
                    "sourceEpochId": "day_2026-04-19_2000utc",
                    "winnerUid": None,
                    "weights": [
                        {"uid": 7, "weight": 2},
                        {"uid": 11, "weight": 1},
                    ],
                }
            ),
            metagraph_size=128,
        )

        self.assertIsInstance(raw_weights, np.ndarray)
        self.assertEqual(metadata["weights_source"], "competition_runtime")
        self.assertAlmostEqual(float(raw_weights[0]), BACKEND_BURN_FRACTION, places=6)
        self.assertAlmostEqual(float(raw_weights[7]), BACKEND_KEEP_FRACTION * (2.0 / 3.0), places=6)
        self.assertAlmostEqual(float(raw_weights[11]), BACKEND_KEEP_FRACTION * (1.0 / 3.0), places=6)
        self.assertAlmostEqual(float(raw_weights.sum()), 1.0, places=6)

    def test_falls_back_to_local_scores_when_backend_vector_is_unusable(self):
        raw_weights, metadata = _extract_competition_weight_vector(
            provider=_Provider(
                {
                    "status": "settled",
                    "epochId": "day_2026-04-20_2000utc",
                    "sourceEpochId": "day_2026-04-19_2000utc",
                    "winnerUid": 42,
                    "weights": [{"uid": 42, "weight": 0}],
                }
            ),
            metagraph_size=128,
        )

        self.assertIsNone(raw_weights)
        self.assertEqual(metadata["weights_source"], "local_scores")
        self.assertEqual(metadata["settlement_winner_uid"], 42)


if __name__ == "__main__":
    unittest.main()
