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
            self.assertEqual(len(env.action_masks()), 8)
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertIsInstance(reward, float)
            self.assertFalse(terminated and truncated)
            self.assertIn("selected_action", info)
        finally:
            env.close()

    def test_registered_toy_env_id_works_after_adapter_import(self) -> None:
        try:
            import gymnasium as gym
        except ImportError:
            self.skipTest("gymnasium optional dependency is not installed")

        import openttd_le.adapters.gymnasium  # noqa: F401

        env = gym.make("OpenTTDLE-Toy-v0")
        try:
            obs, info = env.reset(seed=1)
            self.assertIn("action_mask", obs)
            self.assertIn("candidate_actions", info)
        finally:
            env.close()

    def test_firs_adapter_wraps_native_env_when_gymnasium_installed(self) -> None:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            self.skipTest("gymnasium optional dependency is not installed")

        from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv

        class FakeFIRSEnv:
            def __init__(self) -> None:
                self.last_action = None

            def reset(self, *, seed: int | None = None):
                observation = {
                    "tick": 0,
                    "bank_balance": 500000,
                    "routes": [],
                    "candidate_actions": [
                        {
                            "id": "build_route_1",
                            "kind": "build_cargo_route",
                            "route": {"production": 120},
                            "action": {"type": "build_cargo_route", "source_id": 1, "destination_id": 2, "cargo_id": 3},
                        }
                    ],
                }
                return observation, {"run_dir": "fake", "candidate_actions": observation["candidate_actions"]}

            def step(self, action):
                self.last_action = action
                observation = {
                    "tick": 2220,
                    "bank_balance": 499000,
                    "routes": [{"route_id": "route_1", "delivered": 20, "profit": 100}],
                    "candidate_actions": [{"id": "wait_1", "kind": "wait_months", "action": {"type": "wait_months", "months": 1}}],
                }
                return observation, 25.0, True, False, {"candidate_actions": observation["candidate_actions"]}

            def close(self) -> None:
                return None

        fake = FakeFIRSEnv()
        env = OpenTTDFIRSGymEnv(env=fake, max_candidates=4)
        try:
            obs, info = env.reset(seed=1)
            self.assertEqual(int(obs["action_mask"][0]), 1)
            self.assertEqual(int(env.action_masks()[0]), 1)
            self.assertIn("native_observation", info)
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertEqual(fake.last_action["type"], "build_cargo_route")
            self.assertEqual(reward, 25.0)
            self.assertTrue(terminated)
            self.assertFalse(truncated)
            self.assertEqual(float(obs["cargo_delivered"][0]), 20.0)
        finally:
            env.close()

    def test_firs_adapter_masks_physically_infeasible_candidates(self) -> None:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            self.skipTest("gymnasium optional dependency is not installed")

        from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv

        class FakeFIRSEnv:
            def __init__(self) -> None:
                self.last_action = None

            def reset(self, *, seed: int | None = None):
                observation = {
                    "tick": 0,
                    "bank_balance": 500000,
                    "routes": [],
                    "candidate_actions": [
                        {
                            "id": "build_route_1",
                            "kind": "build_cargo_route",
                            "feasible": False,
                            "diagnostics": ["no_path_between_station_candidates"],
                            "route": {"production": 120},
                            "action": {"type": "build_cargo_route", "source_id": 1, "destination_id": 2, "cargo_id": 3},
                        }
                    ],
                }
                return observation, {"candidate_actions": observation["candidate_actions"]}

            def step(self, action):
                self.last_action = action
                observation = {"tick": 2220, "bank_balance": 499900, "routes": [], "candidate_actions": []}
                return observation, -2.0, False, False, {"candidate_actions": []}

            def close(self) -> None:
                return None

        fake = FakeFIRSEnv()
        env = OpenTTDFIRSGymEnv(env=fake, max_candidates=4, invalid_action_penalty=-5.0)
        try:
            obs, info = env.reset(seed=1)
            self.assertEqual(int(obs["action_mask"][0]), 0)
            obs, reward, terminated, truncated, info = env.step(0)
            self.assertEqual(fake.last_action["type"], "wait_months")
            self.assertTrue(info["invalid_action"])
            self.assertEqual(reward, -7.0)
        finally:
            env.close()

    def test_firs_deterministic_adapter_normalizes_volatile_info(self) -> None:
        try:
            import gymnasium  # noqa: F401
        except ImportError:
            self.skipTest("gymnasium optional dependency is not installed")

        from openttd_le.adapters.gymnasium import FIRS_DETERMINISTIC_GYM_ID, OpenTTDFIRSGymEnv

        class FakeFIRSEnv:
            def __init__(self) -> None:
                self.executed_steps = 0

            def reset(self, *, seed: int | None = None):
                observation = {
                    "tick": 123,
                    "reason": "requested",
                    "bank_balance": 500000,
                    "last_scroll": {"x": 1},
                    "routes": [],
                    "candidate_actions": [
                        {
                            "id": "build_route_1",
                            "kind": "build_cargo_route",
                            "route": {"production": 120},
                            "action": {"type": "build_cargo_route", "source_id": 1, "destination_id": 2, "cargo_id": 3},
                        }
                    ],
                }
                return observation, {
                    "run_dir": "volatile",
                    "candidate_actions": observation["candidate_actions"],
                    "candidate_planning": {"plan_attempts": 1, "proven_feasible": 1, "unknown": 0},
                    "native_observation": observation,
                }

            def step(self, action):
                self.executed_steps += 1
                observation = {
                    "tick": 456,
                    "reason": "after_action",
                    "bank_balance": 499000,
                    "routes": [
                        {
                            "route_id": "route_1",
                            "source_station": 7,
                            "destination_station": 8,
                            "vehicle_details": [{"x": 1, "y": 2}],
                            "delivered": 20,
                            "profit": 100,
                        }
                    ],
                    "candidate_actions": [{"id": "wait_1", "kind": "wait_months", "action": {"type": "wait_months", "months": 1}}],
                }
                return observation, 25.0, True, False, {
                    "run_dir": "volatile",
                    "result": {"action_type": "build_cargo_route", "source_station": 7, "route_id": "route_1"},
                    "candidate_actions": observation["candidate_actions"],
                }

            def close(self) -> None:
                return None

        env = OpenTTDFIRSGymEnv(env=FakeFIRSEnv(), max_candidates=4, deterministic=True)
        try:
            self.assertEqual(env.spec.id, FIRS_DETERMINISTIC_GYM_ID)
            self.assertFalse(env.spec.nondeterministic)
            obs, info = env.reset(seed=1)
            self.assertEqual(float(obs["tick"][0]), 0.0)
            self.assertNotIn("run_dir", info)
            self.assertEqual(info["candidate_planning"]["plan_attempts"], 1)
            self.assertNotIn("tick", info["native_observation"])
            self.assertEqual(info["native_observation"]["decision_step"], 0)

            obs, _reward, _terminated, _truncated, info = env.step(0)
            self.assertEqual(float(obs["tick"][0]), 1.0)
            self.assertNotIn("source_station", info["result"])
            self.assertNotIn("vehicle_details", info["native_observation"]["routes"][0])
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
