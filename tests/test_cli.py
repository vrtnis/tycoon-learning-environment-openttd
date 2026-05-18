from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

from openttd_le.cli import main
from openttd_le.research.core_benchmark import CoreBenchmarkConfig, run_core_benchmark
from openttd_le.research.dataset import export_core_dataset


class CliTests(unittest.TestCase):
    def test_eval_writes_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "eval",
                        "--agent",
                        "greedy",
                        "--scenario",
                        "coal_easy_001",
                        "--out",
                        tmp,
                    ]
                )
            self.assertEqual(code, 0)
            summary = json.loads(stdout.getvalue().strip().splitlines()[0])
            run_dir = Path(summary["run_dir"])
            self.assertTrue((run_dir / "summary.json").exists())
            self.assertTrue((run_dir / "actions.jsonl").exists())
            self.assertTrue((run_dir / "episode.jsonl").exists())
            self.assertTrue((run_dir / "candidate_actions.jsonl").exists())
            self.assertTrue((run_dir / "rewards.jsonl").exists())
            self.assertTrue((run_dir / "diagnostics.jsonl").exists())
            self.assertTrue((run_dir / "replay.json").exists())
            self.assertTrue((run_dir / "metrics.csv").exists())
            self.assertTrue((run_dir / "final_state.json").exists())
            self.assertTrue((run_dir / "screenshots" / "final_map.svg").exists())

            dataset = export_core_dataset(run_dir, Path(tmp) / "dataset.jsonl")
            self.assertTrue(dataset.exists())
            self.assertIn("candidate_actions", dataset.read_text(encoding="utf-8"))

    def test_export_dataset_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                self.assertEqual(
                    main(["eval", "--agent", "greedy", "--scenario", "coal_easy_001", "--out", tmp, "--max-steps", "1"]),
                    0,
                )
            dataset_path = Path(tmp) / "dataset.jsonl"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["export-dataset", "--run", tmp, "--out", str(dataset_path)])

            self.assertEqual(code, 0)
            self.assertTrue(dataset_path.exists())

    def test_core_benchmark_writes_aggregate_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = run_core_benchmark(
                CoreBenchmarkConfig(
                    agents=("candidate_rank",),
                    seeds=(1,),
                    tasks=("coal_easy_001",),
                    output_root=Path(tmp),
                    max_steps=2,
                )
            )

            self.assertTrue((Path(tmp) / "benchmark_summary.json").exists())
            self.assertIn("candidate_rank", payload["aggregate"])

    def test_benchmark_core_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "benchmark-core",
                        "--agents",
                        "candidate_rank",
                        "--seeds",
                        "1",
                        "--tasks",
                        "coal_easy_001",
                        "--max-steps",
                        "1",
                        "--out",
                        tmp,
                    ]
                )

            self.assertEqual(code, 0)
            self.assertTrue((Path(tmp) / "benchmark_summary.json").exists())

    def test_benchmark_core_procedural_cli(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(
                    [
                        "benchmark-core",
                        "--suite",
                        "procedural",
                        "--split",
                        "dev",
                        "--agents",
                        "candidate_rank",
                        "--seeds",
                        "1",
                        "--max-steps",
                        "1",
                        "--procedural-count-per-family",
                        "1",
                        "--out",
                        tmp,
                    ]
                )

            self.assertEqual(code, 0)
            summary = json.loads((Path(tmp) / "benchmark_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(summary["suite"], "procedural")
            self.assertEqual(summary["split"], "dev")
            self.assertTrue(all(task.startswith("proc_dev_") for task in summary["tasks"]))

    def test_list_procedural_scenarios_cli(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(["list-procedural-scenarios", "--split", "test", "--count-per-family", "1"])

        self.assertEqual(code, 0)
        self.assertIn("proc_test_", stdout.getvalue())


if __name__ == "__main__":
    unittest.main()
