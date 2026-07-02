from __future__ import annotations

import unittest
import tempfile
from pathlib import Path

from openttd_le.backends.live import (
    _choose_action,
    _choose_coal_action,
    _find_distinct_free_ports,
    _live_config,
    _new_run_dir,
    _parse_json,
    _recording_window_title,
    _with_client_name,
)


class LiveBackendTests(unittest.TestCase):
    def test_live_config_supports_local_viewer_client(self) -> None:
        config = _live_config(seed=1, game_port=3979, admin_port=3977)
        self.assertIn("server_game_type = local", config)
        self.assertIn("client_name = TycoonLE OpenTTD Server", config)
        viewer_config = _with_client_name(config, "TycoonLE OpenTTD Viewer 3979")
        self.assertIn("client_name = TycoonLE OpenTTD Viewer 3979", viewer_config)
        self.assertIn("OpenTTDLEGameScript =", config)

    def test_parse_json_accepts_fenced_model_action(self) -> None:
        action = _parse_json('```json\n{"type":"road_burst","town_id":7,"label":"test"}\n```')
        self.assertEqual(action["type"], "road_burst")
        self.assertEqual(action["town_id"], 7)

    def test_parse_json_accepts_coal_objective_action(self) -> None:
        action = _parse_json('{"type":"build_coal_route","source_id":1,"destination_id":2,"cargo_id":0,"vehicles":4}')
        self.assertEqual(action["type"], "build_coal_route")
        self.assertEqual(action["vehicles"], 4)

    def test_heuristic_respects_focus_town(self) -> None:
        action = _choose_action(
            {"step": 0, "towns": [{"id": 7, "name": "Fondinghall"}, {"id": 6, "name": "Suston"}]},
            model="gpt-5.5",
            allow_heuristic=True,
            focus_town_id=6,
        )
        self.assertEqual(action["town_id"], 6)

    def test_coal_heuristic_builds_first_pair_then_waits(self) -> None:
        action = _choose_coal_action(
            {
                "coal_pairs": [
                    {"source_id": 4, "destination_id": 9, "cargo_id": 0, "distance": 20},
                ],
                "active_objective": None,
            },
            model="gpt-5.5",
            allow_heuristic=True,
        )
        self.assertEqual(action["type"], "build_coal_route")
        self.assertEqual(action["source_id"], 4)

        wait = _choose_coal_action(
            {"coal_pairs": [], "active_objective": {"delivered": 0}},
            model="gpt-5.5",
            allow_heuristic=True,
        )
        self.assertEqual(wait["type"], "wait")

    def test_recording_window_title_parses_capture_sources(self) -> None:
        self.assertEqual(_recording_window_title("window-region=OpenTTD 15.3"), "OpenTTD 15.3")
        self.assertEqual(_recording_window_title("title=OpenTTD Replay"), "OpenTTD Replay")
        self.assertIsNone(_recording_window_title("desktop"))

    def test_new_run_dir_avoids_same_second_collisions(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            first = _new_run_dir(Path(tmp), suffix="firs_env")
            second = _new_run_dir(Path(tmp), suffix="firs_env")
            self.assertNotEqual(first, second)
            self.assertTrue(first.exists())
            self.assertTrue(second.exists())

    def test_find_distinct_free_ports_does_not_return_same_port(self) -> None:
        game_port, admin_port = _find_distinct_free_ports()
        self.assertNotEqual(game_port, admin_port)


if __name__ == "__main__":
    unittest.main()
