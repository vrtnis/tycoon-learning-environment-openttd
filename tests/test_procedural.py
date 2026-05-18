from __future__ import annotations

import unittest

from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.procedural import generate_procedural_scenarios, procedural_task_ids
from openttd_le.core.scenarios import load_registry
from openttd_le.research.core_benchmark import CoreBenchmarkConfig, benchmark_task_ids


class ProceduralScenarioTests(unittest.TestCase):
    def test_procedural_generation_is_deterministic_by_split(self) -> None:
        first = generate_procedural_scenarios(split="dev", count_per_family=1)
        second = generate_procedural_scenarios(split="dev", count_per_family=1)
        test = generate_procedural_scenarios(split="test", count_per_family=1)

        self.assertEqual([scenario.id for scenario in first], [scenario.id for scenario in second])
        self.assertEqual(first[0].nodes[0], second[0].nodes[0])
        self.assertNotEqual(first[0].nodes[0], test[0].nodes[0])
        self.assertTrue(all("procedural" in scenario.tags for scenario in first))

    def test_generated_registry_can_drive_environment(self) -> None:
        generated = generate_procedural_scenarios(split="dev", count_per_family=1)
        registry = load_registry().extend(generated)
        env = OpenTTDLEnv(registry=registry)
        obs, _ = env.reset(generated[0].id, seed=1)

        self.assertTrue(obs["candidate_actions"])
        self.assertIn("split:dev", obs["scenario"]["tags"])
        env.close()

    def test_procedural_benchmark_defaults_to_generated_task_ids(self) -> None:
        config = CoreBenchmarkConfig(suite="procedural", split="dev", procedural_count_per_family=1)

        self.assertEqual(benchmark_task_ids(config), tuple(procedural_task_ids(split="dev", count_per_family=1)))


if __name__ == "__main__":
    unittest.main()
