import unittest
from datetime import datetime, timezone

from poker44.validator.runtime_provider import _current_competition_epoch


class CompetitionEpochCadenceTests(unittest.TestCase):
    def test_keeps_daily_cadence_before_72_hour_anchor(self):
        epoch = _current_competition_epoch(datetime(2026, 4, 27, 21, 15, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-04-27_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-04-27T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-04-28T20:00:00+00:00")

    def test_uses_72_hour_anchor_after_cutover(self):
        epoch = _current_competition_epoch(datetime(2026, 5, 13, 21, 15, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-05-12_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-05-12T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-05-15T20:00:00+00:00")

    def test_uses_same_72_hour_epoch_before_next_close(self):
        epoch = _current_competition_epoch(datetime(2026, 5, 15, 19, 59, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-05-12_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-05-12T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-05-15T20:00:00+00:00")

    def test_uses_120_hour_anchor_after_5_day_cutover(self):
        epoch = _current_competition_epoch(datetime(2026, 6, 18, 1, 15, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-06-17_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-06-17T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-06-22T20:00:00+00:00")

    def test_uses_same_120_hour_epoch_before_next_close(self):
        epoch = _current_competition_epoch(datetime(2026, 6, 22, 19, 59, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-06-17_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-06-17T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-06-22T20:00:00+00:00")

    def test_uses_shortened_final_epoch_before_v2_start(self):
        epoch = _current_competition_epoch(datetime(2026, 6, 26, 21, 15, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-06-22_2000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-06-22T20:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-06-27T00:00:00+00:00")

    def test_uses_v2_epoch_anchor_after_shortened_final_close(self):
        epoch = _current_competition_epoch(datetime(2026, 6, 27, 0, 15, tzinfo=timezone.utc))

        self.assertEqual(epoch["competition_epoch_id"], "day_2026-06-27_0000utc")
        self.assertEqual(epoch["competition_epoch_start"], "2026-06-27T00:00:00+00:00")
        self.assertEqual(epoch["competition_epoch_end"], "2026-07-02T00:00:00+00:00")


if __name__ == "__main__":
    unittest.main()
