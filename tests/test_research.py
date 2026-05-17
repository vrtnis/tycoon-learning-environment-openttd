from __future__ import annotations

import unittest

from openttd_le.research.api import Prototype, get_cargo_chains, get_routes
from openttd_le.research.benchmarks import aggregate_route_builder_attempts, aggregate_runs, load_benchmark_tasks, select_task
from openttd_le.research.scoring import delivered_cargo_value, score_snapshot


class ResearchApiTests(unittest.TestCase):
    def test_typed_cargo_chain_objects(self) -> None:
        chains = get_cargo_chains(
            {
                "industry_graph": [
                    {
                        "source_id": 1,
                        "source_name": "Fedingworth Port",
                        "destination_id": 2,
                        "destination_name": "Fort Coal Mine",
                        "cargo_id": 3,
                        "cargo_label": "ENSP",
                        "cargo_name": "Engineering Supplies",
                        "distance": 10,
                        "production": 152,
                    }
                ]
            },
            {"ENSP": 1.8},
        )

        self.assertEqual(chains[0].cargo.label, Prototype.Cargo.EngineeringSupplies)
        self.assertEqual(chains[0].source.type, "Port")
        self.assertEqual(chains[0].cargo.value, 1.8)

    def test_cargo_value_score_prefers_processed_cargo(self) -> None:
        score = delivered_cargo_value(
            [
                {"cargo_label": "COAL", "delivered": 10},
                {"cargo_label": "STEL", "delivered": 10},
            ]
        )

        self.assertGreater(score, 20)
        snapshot = score_snapshot({"routes": [{"cargo_label": "STEL", "delivered": 2, "profit": 1000}]})
        self.assertGreater(snapshot["network_value"], snapshot["cargo_score"])

    def test_route_objects_allow_dict_style_get_for_repl_agents(self) -> None:
        route = get_routes(
            {
                "routes": [
                    {
                        "route_id": "route_1",
                        "source_id": 29,
                        "source_name": "Fort Coal Mine",
                        "destination_id": 12,
                        "destination_name": "Steel Mill",
                        "cargo_id": 2,
                        "cargo_label": "COAL",
                        "vehicles": 5,
                        "delivered": 20,
                        "profit": 529,
                        "source_waiting": 164,
                        "destination_waiting": 0,
                    }
                ]
            }
        )[0]

        self.assertEqual(route.get("route_id"), "route_1")
        self.assertEqual(route.get("cargo_label"), Prototype.Cargo.Coal)
        self.assertEqual(route["source_id"], 29)
        self.assertEqual(route.get("destination_waiting"), 0)


class BenchmarkTests(unittest.TestCase):
    def test_load_and_select_default_tasks(self) -> None:
        tasks = load_benchmark_tasks()

        self.assertTrue(tasks)
        self.assertEqual(select_task("lab_supply_mine_short").mode, "lab")

    def test_aggregate_runs(self) -> None:
        aggregate = aggregate_runs(
            [
                {"completed": True, "total_reward": 10, "final_score": {"network_value": 5}, "model": "a", "benchmark_task": "x"},
                {"completed": False, "total_reward": 20, "final_score": {"network_value": 15}, "model": "a", "benchmark_task": "x"},
            ]
        )

        self.assertEqual(aggregate["runs"], 2)
        self.assertEqual(aggregate["success_rate"], 0.5)
        self.assertEqual(aggregate["median_network_value"], 10)

    def test_aggregate_route_builder_attempts(self) -> None:
        aggregate = aggregate_route_builder_attempts(
            [
                {"build_success": True, "active_success": True, "operational_success": True},
                {"build_success": True, "active_success": True, "operational_success": False, "failure_reason": "no_delivery_after_wait"},
                {"build_success": False, "active_success": False, "operational_success": False, "error": "road_connection_failed"},
            ],
            target_success_rate=0.9,
        )

        self.assertEqual(aggregate["attempts"], 3)
        self.assertEqual(aggregate["build_success_rate"], 0.667)
        self.assertEqual(aggregate["active_success_rate"], 0.667)
        self.assertEqual(aggregate["operational_success_rate"], 0.333)
        self.assertFalse(aggregate["level1_pass"])
        self.assertEqual(aggregate["failure_counts"]["road_connection_failed"], 1)


if __name__ == "__main__":
    unittest.main()
