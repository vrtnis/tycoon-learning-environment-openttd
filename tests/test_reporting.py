from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from openttd_le.research.reporting import write_benchmark_report


class ReportingTests(unittest.TestCase):
    def test_report_writer_emits_markdown_tables_and_curve(self) -> None:
        validity = {
            "ok": True,
            "suite": "suite",
            "tasks": ["task"],
            "seeds": [1],
            "sections": {
                "determinism": {"ok": True, "runs": 1, "passed": 1},
                "throughput": {"ok": True, "runs": 1, "median_step_seconds": 2.0},
            },
        }
        training = {
            "task_id": "task",
            "algorithms": ["scripted:first_valid"],
            "aggregate": {
                "runs": 1,
                "per_algorithm": {
                    "scripted:first_valid": {
                        "runs": 1,
                        "best_mean_reward": 1.0,
                        "final_mean_reward": 1.0,
                    }
                },
            },
            "runs": [
                {
                    "algorithm": "scripted:first_valid",
                    "seed": 1,
                    "curve_points": [{"timesteps": 0, "mean_reward": 1.0, "success_rate": 0.0}],
                }
            ],
        }
        route_builder = {
            "seed": 1,
            "economy": "basic_temperate",
            "aggregate": {
                "attempts": 1,
                "operational_success_rate": 1.0,
                "target_success_rate": 0.9,
                "level1_pass": True,
                "failure_counts": {},
            },
        }
        with tempfile.TemporaryDirectory() as tmp:
            payload = write_benchmark_report(
                validity_report=validity,
                training_report=training,
                route_builder_report=route_builder,
                output_dir=Path(tmp),
            )

            self.assertTrue(Path(payload["report"]).exists())
            self.assertTrue((Path(tmp) / "tables" / "validity_sections.csv").exists())
            self.assertTrue((Path(tmp) / "tables" / "learning_curve.csv").exists())
            self.assertTrue((Path(tmp) / "tables" / "route_builder_summary.csv").exists())
            self.assertTrue((Path(tmp) / "curves" / "learning_curves.svg").exists())


if __name__ == "__main__":
    unittest.main()
