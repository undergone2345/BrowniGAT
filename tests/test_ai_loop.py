import unittest
from datetime import datetime

from utils.ai_loop_utils import build_loop_summary, parse_deadline, should_continue


class AILoopTests(unittest.TestCase):
    def test_parse_tonight_deadline_rolls_to_next_midnight(self):
        now = datetime.fromisoformat("2026-06-03T21:15:00+08:00")
        deadline = parse_deadline("tonight_24", now=now)

        self.assertEqual(deadline.isoformat(), "2026-06-04T00:00:00+08:00")

    def test_parse_clock_deadline_rolls_forward_when_time_has_passed(self):
        now = datetime.fromisoformat("2026-06-03T23:50:00+08:00")
        deadline = parse_deadline("23:30", now=now)

        self.assertEqual(deadline.isoformat(), "2026-06-04T23:30:00+08:00")

    def test_parse_24_clock_deadline_maps_to_next_midnight(self):
        now = datetime.fromisoformat("2026-06-03T21:15:00+08:00")
        deadline = parse_deadline("24:00", now=now)

        self.assertEqual(deadline.isoformat(), "2026-06-04T00:00:00+08:00")

    def test_should_continue_honors_iteration_limit(self):
        now = datetime.fromisoformat("2026-06-03T21:15:00+08:00")
        deadline = datetime.fromisoformat("2026-06-04T00:00:00+08:00")

        self.assertTrue(should_continue(now, deadline, 0, max_iterations=2))
        self.assertFalse(should_continue(now, deadline, 2, max_iterations=2))

    def test_build_loop_summary_is_serializable(self):
        started_at = datetime.fromisoformat("2026-06-03T21:15:00+08:00")
        deadline = datetime.fromisoformat("2026-06-04T00:00:00+08:00")
        summary = build_loop_summary(started_at, deadline, 4, 3, 1)

        self.assertEqual(summary["iteration_count"], 4)
        self.assertEqual(summary["success_count"], 3)
        self.assertEqual(summary["failure_count"], 1)


if __name__ == "__main__":
    unittest.main()
