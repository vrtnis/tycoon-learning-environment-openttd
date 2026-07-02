from __future__ import annotations

import unittest
from pathlib import Path

from openttd_le.envs.firs import _normalized_server_command
from openttd_le.research.reproducibility import (
    first_diff,
    normalize_determinism_trace,
    normalize_runtime_lock,
    stable_json_sha256,
)


class ReproducibilityTests(unittest.TestCase):
    def test_strict_trace_keeps_public_route_and_money_fields(self) -> None:
        trace = {
            "event": "step",
            "run_dir": "C:/tmp/run-a",
            "observation": {
                "tick": 2,
                "bank_balance": 499_987.25,
                "routes": [
                    {
                        "route_id": "route_1",
                        "source_station": 7,
                        "destination_station": 8,
                        "source_waiting": 42,
                        "profit": 123.45,
                    }
                ],
            },
        }

        normalized = normalize_determinism_trace(trace, mode="strict")

        self.assertNotIn("run_dir", normalized)
        self.assertEqual(normalized["observation"]["tick"], 2)
        self.assertEqual(normalized["observation"]["bank_balance"], 499_987.25)
        route = normalized["observation"]["routes"][0]
        self.assertEqual(route["source_station"], 7)
        self.assertEqual(route["destination_station"], 8)
        self.assertEqual(route["source_waiting"], 42)
        self.assertEqual(route["profit"], 123.45)

    def test_runtime_lock_normalization_compares_hashes_not_local_paths(self) -> None:
        left = {
            "openttd_executable": "C:/one/openttd.exe",
            "openttd_executable_sha256": "exe-hash",
            "opengfx_baseset": "C:/one/opengfx.tar",
            "opengfx_baseset_sha256": "gfx-hash",
            "firs_newgrf": "C:/one/firs.tar",
            "firs_newgrf_sha256": "firs-hash",
            "cfg": "C:/one/openttd.cfg",
            "cfg_sha256": "cfg-with-port-a",
            "cfg_effective_sha256": "cfg-effective",
            "server_command": ["C:/one/openttd.exe", "-D", "0.0.0.0:1111"],
            "server_command_effective": ["<openttd>", "-D", "0.0.0.0:<ephemeral>"],
            "game_port": 1111,
            "admin_port": 2222,
            "seed": 7,
        }
        right = {
            **left,
            "openttd_executable": "/two/openttd",
            "opengfx_baseset": "/two/opengfx.tar",
            "firs_newgrf": "/two/firs.tar",
            "cfg": "/two/openttd.cfg",
            "cfg_sha256": "cfg-with-port-b",
            "server_command": ["/two/openttd", "-D", "0.0.0.0:3333"],
            "game_port": 3333,
            "admin_port": 4444,
        }

        self.assertIsNone(first_diff(normalize_runtime_lock(left), normalize_runtime_lock(right)))
        self.assertEqual(stable_json_sha256(normalize_runtime_lock(left)), stable_json_sha256(normalize_runtime_lock(right)))

    def test_server_command_normalizer_removes_ephemeral_port_and_cfg_path(self) -> None:
        command = [str(Path("C:/OpenTTD/openttd.exe")), "-D", "0.0.0.0:3979", "-c", "C:/tmp/openttd.cfg", "-G", "1"]

        self.assertEqual(
            _normalized_server_command(command),
            ["<openttd>", "-D", "0.0.0.0:<ephemeral>", "-c", "<cfg>", "-G", "1"],
        )


if __name__ == "__main__":
    unittest.main()
