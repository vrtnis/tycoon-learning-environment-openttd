from __future__ import annotations

import random
import unittest

from openttd_le.research.gym_baselines import select_baseline_action


class GymBaselineTests(unittest.TestCase):
    def test_select_highest_production_uses_action_mask(self) -> None:
        try:
            import numpy as np
        except ImportError:
            self.skipTest("numpy is not installed")

        info = {
            "action_mask": np.array([1, 1, 0], dtype="int8"),
            "candidate_actions": [
                {"route": {"production": 10}},
                {"route": {"production": 50}},
                {"route": {"production": 1000}},
            ],
        }
        action = select_baseline_action("highest_production", {}, info, random.Random(1))
        self.assertEqual(action, 1)


if __name__ == "__main__":
    unittest.main()
