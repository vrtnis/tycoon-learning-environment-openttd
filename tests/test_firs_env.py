from __future__ import annotations

import unittest

from openttd_le.core.types import EnvError
from pathlib import Path
from types import SimpleNamespace

from openttd_le.envs.firs import _allocate_distinct_ports, _assert_fresh_observation, _firs_version_from_path, _normalized_cfg_sha256


class FIRSEnvTests(unittest.TestCase):
    def test_allocate_distinct_ports_returns_two_different_ports(self) -> None:
        game_port, admin_port = _allocate_distinct_ports()
        self.assertIsInstance(game_port, int)
        self.assertIsInstance(admin_port, int)
        self.assertNotEqual(game_port, admin_port)

    def test_fresh_observation_guard_rejects_stale_worlds(self) -> None:
        _assert_fresh_observation({"tick": 100, "routes": []})
        with self.assertRaises(EnvError):
            _assert_fresh_observation({"tick": 100, "routes": [{"route_id": "route_1"}]})
        with self.assertRaises(EnvError):
            _assert_fresh_observation({"tick": 500_001, "routes": []})

    def test_firs_version_parses_content_folder(self) -> None:
        path = Path(".openttd/newgrf/FIRS_Industries_5-5.2.0/firs.grf")
        self.assertEqual(_firs_version_from_path(path), "5-5.2.0")

    def test_normalized_cfg_hash_ignores_ephemeral_ports(self) -> None:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            first = Path(tmp) / "a.cfg"
            second = Path(tmp) / "b.cfg"
            first.write_text("server_port = 1111\nserver_admin_port = 2222\nseed = 1\n", encoding="ascii")
            second.write_text("server_port = 3333\nserver_admin_port = 4444\nseed = 1\n", encoding="ascii")
            self.assertEqual(_normalized_cfg_sha256(first), _normalized_cfg_sha256(second))

    def test_candidate_planning_stops_after_proven_feasible_route(self) -> None:
        from openttd_le.envs.firs import OpenTTDFIRSEnv

        class FakeSession:
            def __init__(self) -> None:
                self.planned: list[int] = []

            def candidate_routes(self) -> list[dict[str, int]]:
                return [
                    {"source_id": index, "destination_id": 100 + index, "cargo_id": 8, "production": 10, "distance": index}
                    for index in range(1, 6)
                ]

            def plan_cargo_route_action(self, action: dict[str, int], *, timeout: float) -> dict[str, int | bool | None]:
                self.planned.append(int(action["source_id"]))
                return {"feasible": True, "error": None, "path_tiles": 12}

        fake = FakeSession()
        env = OpenTTDFIRSEnv(workbook="scenario.xlsx", candidate_stop_after_feasible=1)
        env.session = fake  # type: ignore[assignment]
        env.observation = {"tick": 0, "routes": []}
        env.run_config = SimpleNamespace(vehicles_per_route=5)

        candidates = env.candidate_actions()

        self.assertEqual(fake.planned, [1])
        self.assertEqual(candidates[0]["feasibility"], "proven_feasible")
        self.assertEqual(candidates[0]["feasible"], True)
        self.assertTrue(all(candidate["feasible"] is False for candidate in candidates[1:]))
        self.assertTrue(all(str(candidate["feasibility"]).startswith("unknown_") for candidate in candidates[1:]))
        self.assertEqual(env.candidate_planning_summary["plan_attempts"], 1)
        self.assertEqual(env.candidate_planning_summary["proven_feasible"], 1)

    def test_candidate_planning_timeout_is_unknown_not_valid(self) -> None:
        from openttd_le.envs.firs import OpenTTDFIRSEnv

        class FakeSession:
            def __init__(self) -> None:
                self.planned: list[int] = []

            def candidate_routes(self) -> list[dict[str, int]]:
                return [
                    {"source_id": 1, "destination_id": 101, "cargo_id": 8, "production": 40, "distance": 10},
                    {"source_id": 2, "destination_id": 102, "cargo_id": 8, "production": 30, "distance": 12},
                    {"source_id": 3, "destination_id": 103, "cargo_id": 8, "production": 20, "distance": 14},
                ]

            def plan_cargo_route_action(self, action: dict[str, int], *, timeout: float) -> dict[str, int | bool | None]:
                source_id = int(action["source_id"])
                self.planned.append(source_id)
                if source_id == 1:
                    raise TimeoutError("planner safety timeout")
                return {"feasible": True, "error": None, "path_tiles": 9}

        fake = FakeSession()
        env = OpenTTDFIRSEnv(
            workbook="scenario.xlsx",
            candidate_plan_attempt_limit=2,
            candidate_stop_after_feasible=1,
            candidate_plan_timeout=1.0,
        )
        env.session = fake  # type: ignore[assignment]
        env.observation = {"tick": 0, "routes": []}
        env.run_config = SimpleNamespace(vehicles_per_route=5)

        candidates = env.candidate_actions()
        invalid = [candidate for candidate in candidates if not candidate["feasible"]]

        self.assertEqual(fake.planned, [1, 2])
        self.assertEqual(candidates[0]["feasibility"], "proven_feasible")
        self.assertEqual(candidates[0]["route"]["source_id"], 2)
        self.assertTrue(any(candidate["feasibility"] == "unknown_plan_timeout_or_error" for candidate in invalid))
        self.assertTrue(all(candidate["feasible"] is False for candidate in invalid))


if __name__ == "__main__":
    unittest.main()
