from __future__ import annotations

import unittest


class GymnasiumAdapterTests(unittest.TestCase):
    def test_candidate_index_adapter_steps_when_gymnasium_installed(self) -> None:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            self.skipTest("gymnasium optional dependency is not installed")

        from openttd_le.adapters.gymnasium import OpenTTDLEGymEnv

        env = OpenTTDLEGymEnv(task_id="coal_easy_001", max_candidates=8)
        try:
            obs, info = env.reset(seed=1)
            self.assertIn("candidate_actions", info)
            self.assertIn("action_mask", obs)
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated and truncated)
            self.assertIn("selected_action", info)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
