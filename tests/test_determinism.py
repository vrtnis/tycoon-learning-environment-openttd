from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from openttd_le.research.determinism import DeterminismConfig, run_determinism_check


class FakeActionSpace:
    def __init__(self) -> None:
        self.seed_value: int | None = None

    def seed(self, seed: int) -> None:
        self.seed_value = seed


class FakeCoreEnv:
    def __init__(self, run_dir: Path, repeat: int) -> None:
        self.run_dir = run_dir
        self.executed_steps = 0
        self.runtime_lock = {
            "schema": "openttd-le-runtime-lock-v2",
            "openttd_executable": f"C:/tmp/{repeat}/openttd.exe",
            "openttd_executable_sha256": "exe-hash",
            "opengfx_baseset": f"C:/tmp/{repeat}/opengfx.tar",
            "opengfx_baseset_sha256": "gfx-hash",
            "firs_newgrf": f"C:/tmp/{repeat}/firs.tar",
            "firs_newgrf_sha256": "firs-hash",
            "cfg": str(run_dir / "openttd.cfg"),
            "cfg_sha256": f"port-specific-{repeat}",
            "cfg_effective_sha256": "cfg-effective",
            "server_command": [f"C:/tmp/{repeat}/openttd.exe", "-D", f"0.0.0.0:{30_000 + repeat}"],
            "server_command_effective": ["<openttd>", "-D", "0.0.0.0:<ephemeral>"],
            "game_port": 30_000 + repeat,
            "admin_port": 31_000 + repeat,
            "seed": 7,
            "gym_env_id": "OpenTTD-FIRS-Deterministic-v0",
        }

    def close(self) -> None:
        return None


class FakeGymEnv:
    created = 0

    def __init__(self, **kwargs) -> None:
        FakeGymEnv.created += 1
        output_root = Path(kwargs["output_root"])
        run_dir = output_root / f"fake_repeat_{FakeGymEnv.created}"
        run_dir.mkdir(parents=True, exist_ok=True)
        self.env = FakeCoreEnv(run_dir, FakeGymEnv.created)
        self.action_space = FakeActionSpace()

    def reset(self, *, seed: int | None = None):
        self.env.executed_steps = 0
        return self._observation(), self._info()

    def step(self, action_index: int):
        self.env.executed_steps += 1
        terminated = self.env.executed_steps >= 2
        return self._observation(), 3.0, terminated, False, self._info(action_index=action_index)

    def close(self) -> None:
        self.env.close()

    def _observation(self) -> dict:
        return {
            "tick": self.env.executed_steps,
            "bank_balance": 500_000.0 - self.env.executed_steps * 100.0,
            "route_count": 1.0 if self.env.executed_steps else 0.0,
            "cargo_delivered": float(self.env.executed_steps * 10),
            "action_mask": [1, 0, 0],
        }

    def _info(self, action_index: int | None = None) -> dict:
        info = {
            "action_mask": [1, 0, 0],
            "candidate_actions": [
                {
                    "id": "wait_1_month",
                    "feasible": True,
                    "action": {"type": "wait_months", "months": 1},
                }
            ],
            "native_observation": {
                "tick": self.env.executed_steps,
                "routes": [{"route_id": "route_1", "delivered": self.env.executed_steps * 10}],
            },
        }
        if action_index is not None:
            info["selected_action"] = {"index": action_index}
        return info


class DeterminismHarnessTests(unittest.TestCase):
    def test_determinism_check_writes_strict_trace_and_runtime_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            FakeGymEnv.created = 0
            with patch("openttd_le.research.determinism.OpenTTDFIRSGymEnv", FakeGymEnv):
                payload = run_determinism_check(
                    DeterminismConfig(
                        workbook=Path("scenario.xlsx"),
                        task_id="lab_raw_to_processor",
                        seed=7,
                        repeats=2,
                        output_root=Path(tmp),
                        max_candidates=3,
                        max_steps=3,
                        progress_path=Path(tmp) / "progress.jsonl",
                    )
                )

            self.assertTrue(payload["ok"])
            self.assertEqual(payload["schema"], "openttd-le-determinism-report-v2")
            self.assertEqual(payload["action_sequence"], [0, 0])
            self.assertEqual(len(payload["runs"]), 2)
            self.assertEqual(payload["runs"][0]["runtime_lock_sha256"], payload["runs"][1]["runtime_lock_sha256"])
            for run in payload["runs"]:
                self.assertTrue(Path(run["raw_trace"]).exists())
                self.assertTrue(Path(run["normalized_trace"]).exists())
                self.assertTrue(Path(run["runtime_lock"]).exists())
            report = json.loads((Path(tmp) / "determinism_report.json").read_text(encoding="utf-8"))
            self.assertTrue(report["contract"]["schema"].endswith("v1"))
            progress = (Path(tmp) / "progress.jsonl").read_text(encoding="utf-8").splitlines()
            self.assertEqual(len(progress), 3)
            self.assertIn('"event":"combination_complete"', progress[-1])


if __name__ == "__main__":
    unittest.main()
