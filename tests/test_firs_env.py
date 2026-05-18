from __future__ import annotations

import unittest

from openttd_le.core.types import EnvError
from pathlib import Path

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


if __name__ == "__main__":
    unittest.main()
