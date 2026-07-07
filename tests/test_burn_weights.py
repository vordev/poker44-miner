import unittest

from poker44.validator.constants import BURN_FRACTION, KEEP_FRACTION, UID_ZERO
from poker44.validator.forward import _select_weight_targets


class BurnWeightTests(unittest.TestCase):
    def test_burn_configuration_targets_zero_percent(self):
        self.assertAlmostEqual(BURN_FRACTION, 0.00, places=6)
        self.assertAlmostEqual(KEEP_FRACTION, 1.00, places=6)

    def test_winner_take_all_assigns_uid_zero_and_top_reward(self):
        reward_map = {
            1: 0.25,
            2: 0.8,
            3: 0.4,
        }

        uids, weights = _select_weight_targets(reward_map)

        self.assertEqual(uids, [UID_ZERO, 2])
        self.assertAlmostEqual(float(weights[0]), BURN_FRACTION, places=6)
        self.assertAlmostEqual(float(weights[1]), KEEP_FRACTION, places=6)
        self.assertAlmostEqual(float(weights.sum()), 1.0, places=6)

    def test_burns_everything_when_no_positive_miner_rewards_exist(self):
        reward_map = {
            1: 0.0,
            2: -0.1,
            3: 0.0,
        }

        uids, weights = _select_weight_targets(reward_map)

        self.assertEqual(uids, [UID_ZERO])
        self.assertAlmostEqual(float(weights[0]), 1.0, places=6)
        self.assertEqual(len(weights), 1)


if __name__ == "__main__":
    unittest.main()
