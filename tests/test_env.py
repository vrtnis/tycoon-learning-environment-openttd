from __future__ import annotations

import unittest

from openttd_le.agents import GreedyAgent, RandomAgent
from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.scenarios import load_registry
from openttd_le.core.schemas import CANDIDATE_ACTION_SCHEMA, OBSERVATION_SCHEMA


class ScenarioTests(unittest.TestCase):
    def test_registry_loads_lab_play_pack(self) -> None:
        registry = load_registry()
        self.assertGreaterEqual(len(registry.list()), 10)
        self.assertEqual(registry.get("coal_easy_001").goals.cargo, "coal")


class EnvironmentTests(unittest.TestCase):
    def test_observation_exposes_candidate_action_frontier(self) -> None:
        env = OpenTTDLEnv()
        obs, _ = env.reset("coal_easy_001", seed=1)

        candidates = obs["candidate_actions"]

        self.assertEqual(obs["schema"], OBSERVATION_SCHEMA)
        self.assertTrue(candidates)
        self.assertEqual(candidates[0]["schema"], CANDIDATE_ACTION_SCHEMA)
        self.assertIn("action", candidates[0])
        self.assertIn("estimates", candidates[0])
        self.assertTrue(any(item["kind"] == "build_route" for item in candidates))
        env.close()

    def test_preview_and_reward_details_are_available(self) -> None:
        env = OpenTTDLEnv()
        obs, _ = env.reset("coal_easy_001", seed=1)
        action = next(item["action"] for item in obs["candidate_actions"] if item["kind"] == "build_route")

        preview = env.preview(action)
        result = env.step(action)

        self.assertIn("components", preview)
        self.assertIn("reward_details", result.info)
        self.assertIn("components", result.info["reward_details"])
        self.assertIn("route_built", result.info["reward_details"]["milestones"])
        env.close()

    def test_greedy_agent_delivers_cargo(self) -> None:
        env = OpenTTDLEnv()
        obs, _ = env.reset("coal_easy_001", seed=1)
        agent = GreedyAgent()
        for _ in range(obs["time"]["max_steps"]):
            result = env.step(agent.act(obs))
            obs = result.observation
            if result.terminated:
                break
        self.assertGreater(obs["metrics"]["cargo_delivered"], 0)
        self.assertGreater(obs["metrics"]["score"], 50)
        env.close()

    def test_openttd_backend_reports_missing_executable_cleanly(self) -> None:
        from openttd_le.backends.openttd import OpenTTDBackend

        backend = OpenTTDBackend(executable="Z:/missing/openttd.exe")
        self.assertFalse(backend.smoke()["exists"])

    def test_invalid_action_is_counted(self) -> None:
        env = OpenTTDLEnv()
        env.reset("coal_easy_001", seed=1)
        result = env.step({"type": "build_route", "source_id": "missing"})
        self.assertEqual(result.observation["metrics"]["invalid_actions"], 1)
        env.close()

    def test_random_agent_returns_valid_action_shape(self) -> None:
        env = OpenTTDLEnv()
        obs, _ = env.reset("passenger_pair_001", seed=4)
        action = RandomAgent(seed=4).act(obs)
        self.assertIn(action["type"], obs["allowed_actions"])
        env.close()


if __name__ == "__main__":
    unittest.main()
