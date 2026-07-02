from __future__ import annotations

import json
import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

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
                        "--backend",
                        "toy",
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
                    main(
                        [
                            "eval",
                            "--agent",
                            "greedy",
                            "--backend",
                            "toy",
                            "--scenario",
                            "coal_easy_001",
                            "--out",
                            tmp,
                            "--max-steps",
                            "1",
                        ]
                    ),
                    0,
                )
            dataset_path = Path(tmp) / "dataset.jsonl"
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["export-dataset", "--run", tmp, "--out", str(dataset_path)])

            self.assertEqual(code, 0)
            self.assertTrue(dataset_path.exists())

    def test_eval_defaults_to_real_openttd_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp) / "run"
            run_dir.mkdir()

            class FakeAgent:
                def act(self, observation: dict) -> dict:
                    return {"type": "wait_months", "months": 1}

                def close(self) -> None:
                    return None

            class FakeEnv:
                max_steps = 1

                def __init__(self, **kwargs: object) -> None:
                    self.kwargs = kwargs

                def reset(self) -> tuple[dict, dict]:
                    return {"routes": []}, {"run_dir": str(run_dir)}

                def step(self, action: dict) -> tuple[dict, float, bool, bool, dict]:
                    return (
                        {"routes": []},
                        1.0,
                        True,
                        False,
                        {
                            "actions": [{"action": action, "result": {"type": "result"}, "observation": {"routes": []}}],
                            "reward_details": {"reward": 1.0},
                            "snapshot": {},
                        },
                    )

                def summary(self, *, agent: str, model: str | None) -> dict:
                    return {
                        "objective": "openttd_firs_env",
                        "completed": True,
                        "total_reward": 1.0,
                        "backend": "openttd",
                        "run_dir": str(run_dir),
                    }

                def launch_info(self, *, summary_path: Path | None = None) -> dict:
                    return {"run_dir": str(run_dir), "summary": str(summary_path)}

                def close(self) -> None:
                    return None

            with (
                patch("openttd_le.cli.make_firs_agent", return_value=FakeAgent()),
                patch("openttd_le.cli.OpenTTDFIRSEnv", side_effect=FakeEnv) as env_cls,
                patch("openttd_le.cli.export_run_to_xlsx", return_value=run_dir / "report.xlsx"),
                patch("openttd_le.cli.export_replay", return_value=run_dir / "replay.json"),
            ):
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(["eval", "--scenario", "lab_raw_to_processor", "--out", tmp])

            self.assertEqual(code, 0)
            env_cls.assert_called_once()
            kwargs = env_cls.call_args.kwargs
            self.assertEqual(kwargs["task_id"], "lab_raw_to_processor")
            self.assertEqual(kwargs["seed"], 1)
            summary = json.loads(stdout.getvalue().strip())
            self.assertEqual(summary["backend"], "openttd")

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

    def test_list_openttd_scenarios_cli(self) -> None:
        stdout = StringIO()
        with redirect_stdout(stdout):
            code = main(["list-openttd-scenarios"])

        self.assertEqual(code, 0)
        self.assertIn("lab_raw_to_processor", stdout.getvalue())

    def test_smoke_openttd_reports_firs_readiness(self) -> None:
        with patch("openttd_le.cli.OpenTTDBackend") as backend_cls, patch(
            "openttd_le.cli._firs_readiness", return_value={"ready": False, "error": "missing"}
        ):
            backend_cls.return_value.smoke.return_value = {"executable": "openttd.exe", "exists": True}
            stdout = StringIO()
            with redirect_stdout(stdout):
                code = main(["smoke-openttd", "--firs"])

        self.assertEqual(code, 0)
        backend_cls.return_value.smoke.assert_called_once()
        backend_cls.return_value.smoke_launch.assert_not_called()
        payload = json.loads(stdout.getvalue())
        self.assertFalse(payload["firs"]["ready"])

    def test_determinism_check_cli_invokes_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "ok": True,
                "report": str(Path(tmp) / "determinism_report.json"),
                "comparisons": [],
            }
            with patch("openttd_le.cli.run_determinism_check", return_value=report) as runner:
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "determinism-check",
                            "--workbook",
                            "scenario.xlsx",
                            "--scenario",
                            "lab_raw_to_processor",
                            "--openttd-user-dir",
                            ".openttd",
                            "--out",
                            tmp,
                            "--agent",
                            "first_valid",
                            "--seed",
                            "7",
                            "--repeats",
                            "2",
                        ]
                    )

            self.assertEqual(code, 0)
            config = runner.call_args.args[0]
            self.assertEqual(config.task_id, "lab_raw_to_processor")
            self.assertEqual(config.seed, 7)
            self.assertEqual(config.repeats, 2)
            self.assertEqual(config.openttd_user_dir, Path(".openttd"))
            self.assertTrue(json.loads(stdout.getvalue())["ok"])

    def test_benchmark_validity_pack_cli_invokes_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "ok": True,
                "report": str(Path(tmp) / "validity_report.json"),
                "sections": {"determinism": {"ok": True}},
            }
            with patch("openttd_le.cli.run_validity_pack", return_value=report) as runner:
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "benchmark-validity-pack",
                            "--workbook",
                            "scenario.xlsx",
                            "--tasks",
                            "lab_raw_to_processor",
                            "--agents",
                            "first_valid",
                            "--seeds",
                            "1,2",
                            "--openttd-user-dir",
                            ".openttd",
                            "--out",
                            tmp,
                            "--skip-route-builder",
                        ]
                    )

            self.assertEqual(code, 0)
            config = runner.call_args.args[0]
            self.assertEqual(config.tasks, ("lab_raw_to_processor",))
            self.assertEqual(config.agents, ("first_valid",))
            self.assertEqual(config.seeds, (1, 2))
            self.assertTrue(config.skip_route_builder)
            self.assertTrue(json.loads(stdout.getvalue())["ok"])

    def test_train_rl_baselines_cli_invokes_runner(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            report = {
                "report": str(Path(tmp) / "rl_training_report.json"),
                "artifacts": {"benchmark_report": str(Path(tmp) / "benchmark_report.md")},
                "aggregate": {"runs": 1},
            }
            with patch("openttd_le.cli.run_rl_training", return_value=report) as runner:
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "train-rl-baselines",
                            "--workbook",
                            "scenario.xlsx",
                            "--scenario",
                            "lab_raw_to_processor",
                            "--algorithms",
                            "scripted:first_valid",
                            "--seeds",
                            "1,2",
                            "--openttd-user-dir",
                            ".openttd",
                            "--out",
                            tmp,
                        ]
                    )

            self.assertEqual(code, 0)
            config = runner.call_args.args[0]
            self.assertEqual(config.algorithms, ("scripted:first_valid",))
            self.assertEqual(config.seeds, (1, 2))
            self.assertEqual(config.openttd_user_dir, Path(".openttd"))
            self.assertEqual(json.loads(stdout.getvalue())["aggregate"]["runs"], 1)

    def test_build_benchmark_report_cli_invokes_writer(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            payload = {"report": str(Path(tmp) / "benchmark_report.md"), "tables": {}, "curves": {}}
            with patch("openttd_le.cli.write_benchmark_report", return_value=payload) as writer:
                stdout = StringIO()
                with redirect_stdout(stdout):
                    code = main(
                        [
                            "build-benchmark-report",
                            "--validity-report",
                            "validity_report.json",
                            "--training-report",
                            "rl_training_report.json",
                            "--route-builder-report",
                            "route_builder_summary.json",
                            "--out",
                            tmp,
                        ]
                    )

            self.assertEqual(code, 0)
            kwargs = writer.call_args.kwargs
            self.assertEqual(kwargs["validity_report"], Path("validity_report.json"))
            self.assertEqual(kwargs["training_report"], Path("rl_training_report.json"))
            self.assertEqual(kwargs["route_builder_report"], Path("route_builder_summary.json"))
            self.assertEqual(json.loads(stdout.getvalue())["report"], payload["report"])


if __name__ == "__main__":
    unittest.main()
