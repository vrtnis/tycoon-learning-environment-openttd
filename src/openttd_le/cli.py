from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path
from typing import Any

from openttd_le import __version__
from openttd_le.agents import make_agent, make_firs_agent
from openttd_le.backends.firs import load_firs_config, verify_firs_installed
from openttd_le.backends.live import (
    launch_coal_objective,
    launch_firs_benchmark,
    launch_firs_live,
    launch_firs_replay,
    launch_firs_research,
    launch_gpt_live,
    launch_route_builder_benchmark,
)
from openttd_le.backends.openttd import OpenTTDBackend
from openttd_le.backends.visual import ensure_opengfx, install_bridge, install_live_bridge, launch_watch_game
from openttd_le.core.artifacts import RunArtifacts
from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.procedural import DEFAULT_PROCEDURAL_COUNT_PER_FAMILY, PROCEDURAL_FAMILIES, generate_procedural_scenarios
from openttd_le.core.scenarios import load_registry
from openttd_le.envs import OpenTTDFIRSEnv
from openttd_le.research.benchmarks import load_benchmark_tasks
from openttd_le.research.core_benchmark import CORE_SUITE_TASKS, CoreBenchmarkConfig, run_core_benchmark
from openttd_le.research.dataset import export_core_dataset
from openttd_le.research.determinism import DeterminismConfig, run_determinism_check
from openttd_le.research.gym_baselines import GYM_BASELINE_AGENTS, GymBaselineConfig, run_gym_baselines
from openttd_le.research.reporting import write_benchmark_report
from openttd_le.research.rl_training import RLModelEvalConfig, RLTrainingConfig, run_rl_model_eval, run_rl_training
from openttd_le.research.validity import ValidityConfig, run_validity_pack
from openttd_le.replay import export_replay
from openttd_le.replay_render import render_core_replay
from openttd_le.workbooks.export import export_run_to_xlsx
from openttd_le.workbooks.template import create_firs_ops_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="tycoonle-openttd")
    parser.add_argument("--version", action="version", version=f"tycoonle-openttd {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-scenarios", help="List bundled scenarios.")
    list_parser.add_argument("--scenario-file", default=None)

    list_openttd_parser = subparsers.add_parser("list-openttd-scenarios", help="List real OpenTTD/FIRS benchmark tasks.")
    list_openttd_parser.add_argument("--scenario-file", default=None)

    list_procedural_parser = subparsers.add_parser("list-procedural-scenarios", help="List generated procedural scenarios.")
    list_procedural_parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    list_procedural_parser.add_argument("--count-per-family", type=int, default=DEFAULT_PROCEDURAL_COUNT_PER_FAMILY)

    eval_parser = subparsers.add_parser("eval", help="Run an agent on a real OpenTTD/FIRS scenario.")
    eval_parser.add_argument("--scenario", required=True)
    eval_parser.add_argument(
        "--agent",
        choices=["openai", "heuristic", "random", "greedy", "candidate_rank", "preview_rerank", "openrouter"],
        default="openai",
    )
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument(
        "--backend",
        choices=["openttd", "toy"],
        default="openttd",
        help="Use real OpenTTD/FIRS by default. 'toy' is a mock backend for CI and fast unit tests.",
    )
    eval_parser.add_argument("--scenario-file", default=None)
    eval_parser.add_argument("--workbook", default="templates/firs_ops_plan.xlsx")
    eval_parser.add_argument("--executable", default=None)
    eval_parser.add_argument("--openttd-user-dir", default=None)
    eval_parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Allow the built-in deterministic OpenTTD bridge policy instead of an API-backed model.",
    )
    eval_parser.add_argument("--runs", type=int, default=1)
    eval_parser.add_argument("--seed", type=int, default=1)
    eval_parser.add_argument("--out", default="runs")
    eval_parser.add_argument("--max-steps", type=int, default=None)
    eval_parser.add_argument("--step-delay", type=float, default=0.0)

    summary_parser = subparsers.add_parser("summarize", help="Summarize run artifacts.")
    summary_parser.add_argument("runs_dir", nargs="?", default="runs")
    summary_parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")

    benchmark_core_parser = subparsers.add_parser(
        "benchmark-core",
        help="Run the local researcher MVP benchmark suite across agents and seeds.",
    )
    benchmark_core_parser.add_argument("--suite", choices=["core", "procedural"], default="core")
    benchmark_core_parser.add_argument("--split", choices=["train", "dev", "test"], default="dev")
    benchmark_core_parser.add_argument(
        "--agents",
        default="random,greedy,candidate_rank,preview_rerank",
        help="Comma-separated agent names.",
    )
    benchmark_core_parser.add_argument("--seeds", default="1,2,3", help="Comma-separated integer seeds.")
    benchmark_core_parser.add_argument(
        "--tasks",
        default="",
        help="Comma-separated scenario ids. Defaults to the selected suite.",
    )
    benchmark_core_parser.add_argument("--backend", choices=["toy"], default="toy")
    benchmark_core_parser.add_argument("--out", default="runs_core")
    benchmark_core_parser.add_argument("--max-steps", type=int, default=None)
    benchmark_core_parser.add_argument("--procedural-count-per-family", type=int, default=DEFAULT_PROCEDURAL_COUNT_PER_FAMILY)

    check_gym_parser = subparsers.add_parser(
        "check-gym-env",
        help="Validate the Gymnasium adapter contract with gymnasium.utils.env_checker.",
    )
    check_gym_parser.add_argument("--env-id", default=None, help="Registered Gymnasium id, e.g. OpenTTD-FIRS-Lab-v0.")
    check_gym_parser.add_argument("--backend", choices=["openttd", "toy"], default="toy")
    check_gym_parser.add_argument("--scenario", default=None)
    check_gym_parser.add_argument("--workbook", default="scenario.xlsx")
    check_gym_parser.add_argument("--executable", default=None)
    check_gym_parser.add_argument("--openttd-user-dir", default=None)
    check_gym_parser.add_argument("--max-candidates", type=int, default=24)
    check_gym_parser.add_argument("--max-steps", type=int, default=2)
    check_gym_parser.add_argument("--deterministic", action="store_true", help="Check the strict deterministic real OpenTTD/FIRS adapter.")
    check_gym_parser.add_argument("--skip-render-check", action=argparse.BooleanOptionalAction, default=True)

    determinism_parser = subparsers.add_parser(
        "determinism-check",
        help="Repeat a fixed Gym/OpenTTD action script and compare normalized state/reward traces.",
    )
    determinism_parser.add_argument("--workbook", default="scenario.xlsx")
    determinism_parser.add_argument("--scenario", default="lab_raw_to_processor")
    determinism_parser.add_argument("--executable", default=None)
    determinism_parser.add_argument("--openttd-user-dir", default=None)
    determinism_parser.add_argument("--out", default="runs_determinism")
    determinism_parser.add_argument(
        "--agent",
        default="first_valid",
        choices=GYM_BASELINE_AGENTS,
        help="Deterministic baseline used to generate the action sequence.",
    )
    determinism_parser.add_argument("--seed", type=int, default=1)
    determinism_parser.add_argument("--repeats", type=int, default=3)
    determinism_parser.add_argument("--max-candidates", type=int, default=24)
    determinism_parser.add_argument("--max-steps", type=int, default=3)
    determinism_parser.add_argument(
        "--trace-mode",
        choices=("strict", "semantic"),
        default="strict",
        help="Strict compares the public trace except runtime artifacts; semantic uses the older relaxed normalizer.",
    )
    determinism_parser.add_argument(
        "--no-fixed-action-script",
        action="store_true",
        help="Recompute baseline actions on every repeat instead of replaying the first repeat's action indices.",
    )
    determinism_parser.add_argument(
        "--no-runtime-lock-compare",
        action="store_true",
        help="Do not fail when comparable runtime/input fingerprints differ.",
    )
    determinism_parser.add_argument(
        "--progress-jsonl",
        default=None,
        help="Append repeat and combination progress events to this JSONL file.",
    )

    benchmark_gym_parser = subparsers.add_parser(
        "benchmark-gym",
        help="Run non-GPT Gymnasium baselines against real OpenTTD/FIRS.",
    )
    benchmark_gym_parser.add_argument("--workbook", default="scenario.xlsx")
    benchmark_gym_parser.add_argument("--scenario", default="lab_raw_to_processor")
    benchmark_gym_parser.add_argument("--executable", default=None)
    benchmark_gym_parser.add_argument("--openttd-user-dir", default=None)
    benchmark_gym_parser.add_argument("--out", default="runs_gym_baselines")
    benchmark_gym_parser.add_argument(
        "--agents",
        default="masked_random,first_valid,highest_production,shortest_route",
        help=f"Comma-separated baseline names. Choices: {','.join(GYM_BASELINE_AGENTS)}",
    )
    benchmark_gym_parser.add_argument("--seeds", default="1")
    benchmark_gym_parser.add_argument("--max-candidates", type=int, default=24)
    benchmark_gym_parser.add_argument("--max-steps", type=int, default=8)
    benchmark_gym_parser.add_argument("--deterministic", action="store_true")

    train_rl_parser = subparsers.add_parser(
        "train-rl-baselines",
        help="Run RL training/eval harnesses and emit learning curves for real OpenTTD/FIRS Gym tasks.",
    )
    train_rl_parser.add_argument("--workbook", default="scenario.xlsx")
    train_rl_parser.add_argument("--scenario", default="lab_raw_to_processor")
    train_rl_parser.add_argument("--executable", default=None)
    train_rl_parser.add_argument("--openttd-user-dir", default=None)
    train_rl_parser.add_argument("--out", default="runs_rl")
    train_rl_parser.add_argument(
        "--algorithms",
        default="scripted:masked_random,scripted:first_valid",
        help="Comma-separated algorithms: scripted:<baseline>, dqn, maskable_ppo.",
    )
    train_rl_parser.add_argument("--seeds", default="1")
    train_rl_parser.add_argument("--max-candidates", type=int, default=24)
    train_rl_parser.add_argument("--max-steps", type=int, default=8)
    train_rl_parser.add_argument("--total-timesteps", type=int, default=64)
    train_rl_parser.add_argument("--eval-interval", type=int, default=32)
    train_rl_parser.add_argument("--eval-episodes", type=int, default=1)

    eval_rl_parser = subparsers.add_parser(
        "eval-rl-model",
        help="Run a saved RL model against a real OpenTTD/FIRS Gym task and write eval artifacts.",
    )
    eval_rl_parser.add_argument("--model", required=True, help="Path to a saved DQN or MaskablePPO .zip model.")
    eval_rl_parser.add_argument("--algorithm", choices=["auto", "dqn", "maskable_ppo"], default="auto")
    eval_rl_parser.add_argument("--workbook", default="scenario.xlsx")
    eval_rl_parser.add_argument("--scenario", default="lab_raw_to_processor")
    eval_rl_parser.add_argument("--executable", default=None)
    eval_rl_parser.add_argument("--openttd-user-dir", default=None)
    eval_rl_parser.add_argument("--out", default="runs_rl_eval")
    eval_rl_parser.add_argument("--seeds", default="1")
    eval_rl_parser.add_argument("--max-candidates", type=int, default=24)
    eval_rl_parser.add_argument("--max-steps", type=int, default=8)
    eval_rl_parser.add_argument("--eval-episodes", type=int, default=1)
    eval_rl_parser.add_argument("--stochastic", action="store_true", help="Sample from the policy instead of using deterministic actions.")

    validity_parser = subparsers.add_parser(
        "benchmark-validity-pack",
        help="Run the real OpenTTD/FIRS benchmark validity pack: determinism, baselines, throughput, and construction reliability.",
    )
    validity_parser.add_argument("--workbook", default="scenario.xlsx")
    validity_parser.add_argument("--suite-file", default=None)
    validity_parser.add_argument("--benchmark-file", default=None)
    validity_parser.add_argument("--tasks", default="", help="Comma-separated task ids. Defaults to the suite manifest.")
    validity_parser.add_argument(
        "--agents",
        default="",
        help=f"Comma-separated baseline names. Defaults to the suite manifest. Choices: {','.join(GYM_BASELINE_AGENTS)}",
    )
    validity_parser.add_argument("--seeds", default="", help="Comma-separated integer seeds. Defaults to the suite manifest.")
    validity_parser.add_argument("--executable", default=None)
    validity_parser.add_argument("--openttd-user-dir", default=None)
    validity_parser.add_argument("--out", default="runs_validity")
    validity_parser.add_argument("--max-candidates", type=int, default=24)
    validity_parser.add_argument("--determinism-repeats", type=int, default=None)
    validity_parser.add_argument("--determinism-max-steps", type=int, default=None)
    validity_parser.add_argument("--baseline-max-steps", type=int, default=None)
    validity_parser.add_argument("--throughput-steps", type=int, default=None)
    validity_parser.add_argument("--route-builder-attempts", type=int, default=None)
    validity_parser.add_argument("--route-builder-wait-months", type=int, default=6)
    validity_parser.add_argument("--route-builder-target-success-rate", type=float, default=None)
    validity_parser.add_argument("--skip-determinism", action="store_true")
    validity_parser.add_argument("--skip-baselines", action="store_true")
    validity_parser.add_argument("--skip-throughput", action="store_true")
    validity_parser.add_argument("--skip-route-builder", action="store_true")

    report_parser = subparsers.add_parser(
        "build-benchmark-report",
        help="Build Markdown, CSV tables, and SVG curves from validity/training JSON reports.",
    )
    report_parser.add_argument("--validity-report", default=None)
    report_parser.add_argument("--training-report", default=None)
    report_parser.add_argument("--route-builder-report", default=None)
    report_parser.add_argument("--out", default="runs_report")
    report_parser.add_argument("--title", default="TycoonLE OpenTTD FIRS Benchmark Report")

    smoke_parser = subparsers.add_parser("smoke-openttd", help="Check real OpenTTD executable integration.")
    smoke_parser.add_argument("--executable", default=None)
    smoke_parser.add_argument("--launch", action="store_true", help="Start a short dedicated-server smoke run.")
    smoke_parser.add_argument("--scenario", default="coal_easy_001")
    smoke_parser.add_argument("--firs", action="store_true", help="Also verify OpenGFX, bundled bridge scripts, and FIRS NewGRF.")
    smoke_parser.add_argument("--openttd-user-dir", default=None)

    bridge_parser = subparsers.add_parser("install-bridge", help="Install the OpenTTD NoAI bridge.")
    bridge_parser.add_argument("--json", action="store_true", help="Emit JSON instead of text.")

    watch_parser = subparsers.add_parser("watch-gpt", help="Launch a visible OpenTTD bridge demo.")
    watch_parser.add_argument("--executable", default=None)
    watch_parser.add_argument("--out", default="runs_watch")
    watch_parser.add_argument("--seed", type=int, default=1)
    watch_parser.add_argument("--resolution", default="1280x800")
    watch_parser.add_argument("--model", default="gpt-5.5")
    watch_parser.add_argument("--no-launch", action="store_true", help="Prepare artifacts without opening OpenTTD.")
    watch_parser.add_argument(
        "--no-plan",
        action="store_true",
        help="Skip the optional OpenAI planning call even if OPENAI_API_KEY is set.",
    )

    live_parser = subparsers.add_parser("play-gpt-live", help="Launch OpenTTD and let GPT drive visible actions.")
    live_parser.add_argument("--executable", default=None)
    live_parser.add_argument("--out", default="runs_live")
    live_parser.add_argument("--seed", type=int, default=1)
    live_parser.add_argument("--steps", type=int, default=4)
    live_parser.add_argument("--resolution", default="1280x800")
    live_parser.add_argument("--model", default="gpt-5.5")
    live_parser.add_argument("--focus-town-id", type=int, default=None)
    live_parser.add_argument("--start-delay", type=float, default=8.0)
    live_parser.add_argument("--step-delay", type=float, default=4.0)
    live_parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Use a deterministic local policy when OPENAI_API_KEY is unavailable.",
    )

    coal_parser = subparsers.add_parser("play-coal-live", help="Launch OpenTTD and let GPT pursue a coal delivery.")
    coal_parser.add_argument("--executable", default=None)
    coal_parser.add_argument("--out", default="runs_coal")
    coal_parser.add_argument("--seed", type=int, default=1)
    coal_parser.add_argument("--steps", type=int, default=6)
    coal_parser.add_argument("--resolution", default="1280x800")
    coal_parser.add_argument("--model", default="gpt-5.5")
    coal_parser.add_argument("--start-delay", type=float, default=10.0)
    coal_parser.add_argument("--step-delay", type=float, default=3.0)
    coal_parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Use a deterministic local policy when OPENAI_API_KEY is unavailable.",
    )

    firs_init_parser = subparsers.add_parser("firs-init-workbook", help="Create a FIRS operations workbook.")
    firs_init_parser.add_argument("--out", default="scenario.xlsx")
    firs_init_parser.add_argument("--config", default=None)

    firs_parser = subparsers.add_parser("play-firs-live", help="Run the workbook-driven FIRS operations loop.")
    firs_parser.add_argument("--workbook", required=True)
    firs_parser.add_argument("--executable", default=None)
    firs_parser.add_argument(
        "--openttd-user-dir",
        default=None,
        help="OpenTTD user data directory for FIRS assets, scripts, and base graphics.",
    )
    firs_parser.add_argument("--out", default="runs_firs")
    firs_parser.add_argument("--steps", type=int, default=10)
    firs_parser.add_argument("--resolution", default="1280x800")
    firs_parser.add_argument("--model", default="gpt-5.5")
    firs_parser.add_argument("--record", action="store_true")
    firs_parser.add_argument(
        "--record-source",
        default=None,
        help='ffmpeg gdigrab/x11grab input. Defaults to title="OpenTTD 15.3" on Windows.',
    )
    firs_parser.add_argument(
        "--repl",
        action="store_true",
        help="Use a persistent safe Python REPL for GPT actions instead of one-shot JSON actions.",
    )
    firs_parser.add_argument("--start-delay", type=float, default=10.0)
    firs_parser.add_argument("--step-delay", type=float, default=3.0)
    firs_parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Use a deterministic local policy when OPENAI_API_KEY is unavailable.",
    )

    firs_research_parser = subparsers.add_parser(
        "firs-research-run",
        help="Run a headless FLE-style FIRS benchmark with persistent Python REPL artifacts.",
    )
    firs_research_parser.add_argument("--workbook", required=True)
    firs_research_parser.add_argument("--executable", default=None)
    firs_research_parser.add_argument(
        "--openttd-user-dir",
        default=None,
        help="OpenTTD user data directory for FIRS assets, scripts, and base graphics.",
    )
    firs_research_parser.add_argument("--out", default="runs_firs_research")
    firs_research_parser.add_argument("--steps", type=int, default=32)
    firs_research_parser.add_argument("--model", default="gpt-5.5")
    firs_research_parser.add_argument("--task", default=None, help="Benchmark task id from scenarios/firs_benchmarks.json.")
    firs_research_parser.add_argument("--benchmark-file", default=None)
    firs_research_parser.add_argument("--step-delay", type=float, default=0.0)
    firs_research_parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Use a deterministic local policy when OPENAI_API_KEY is unavailable.",
    )

    firs_benchmark_parser = subparsers.add_parser(
        "firs-benchmark",
        help="Run FIRS research tasks across models/repeats and aggregate results.",
    )
    firs_benchmark_parser.add_argument("--workbook", required=True)
    firs_benchmark_parser.add_argument("--executable", default=None)
    firs_benchmark_parser.add_argument("--openttd-user-dir", default=None)
    firs_benchmark_parser.add_argument("--out", default="runs_firs_benchmark")
    firs_benchmark_parser.add_argument("--models", default="gpt-5.5", help="Comma-separated model names.")
    firs_benchmark_parser.add_argument("--tasks", default=None, help="Comma-separated task ids. Defaults to all tasks.")
    firs_benchmark_parser.add_argument("--repeats", type=int, default=1)
    firs_benchmark_parser.add_argument("--benchmark-file", default=None)
    firs_benchmark_parser.add_argument("--allow-heuristic", action="store_true")

    route_builder_parser = subparsers.add_parser(
        "benchmark-route-builder",
        help="Measure physical FIRS route-construction reliability without GPT planning.",
    )
    route_builder_parser.add_argument("--workbook", default="templates/firs_ops_plan.xlsx")
    route_builder_parser.add_argument("--executable", default=None)
    route_builder_parser.add_argument("--openttd-user-dir", default=None)
    route_builder_parser.add_argument("--out", default="runs_route_builder")
    route_builder_parser.add_argument("--seed", type=int, default=None)
    route_builder_parser.add_argument("--economy", default=None)
    route_builder_parser.add_argument("--attempts", type=int, default=20)
    route_builder_parser.add_argument("--vehicles", type=int, default=None)
    route_builder_parser.add_argument("--wait-months", type=int, default=6)
    route_builder_parser.add_argument("--max-path-tiles", type=int, default=256)
    route_builder_parser.add_argument("--target-success-rate", type=float, default=0.9)

    export_parser = subparsers.add_parser("export-xlsx", help="Export a run directory to a FIRS Excel report.")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--workbook", default=None)

    dataset_parser = subparsers.add_parser("export-dataset", help="Export core research traces to JSONL or Parquet.")
    dataset_parser.add_argument("--run", required=True, help="Run directory or parent directory containing runs.")
    dataset_parser.add_argument("--out", required=True)
    dataset_parser.add_argument("--format", choices=["jsonl", "parquet"], default=None)

    replay_parser = subparsers.add_parser("export-replay", help="Export a run directory to a replay manifest JSON.")
    replay_parser.add_argument("--run", required=True)
    replay_parser.add_argument("--out", default=None)

    render_core_parser = subparsers.add_parser("render-core-replay", help="Render core/toy episode traces to SVG frames or MP4.")
    render_core_source = render_core_parser.add_mutually_exclusive_group(required=True)
    render_core_source.add_argument("--episode", default=None, help="Path to episode.jsonl.")
    render_core_source.add_argument("--replay", default=None, help="Path to replay.json; sibling episode.jsonl will be used.")
    render_core_parser.add_argument("--out", required=True, help="Output frame directory or .mp4 path.")
    render_core_parser.add_argument("--fps", type=int, default=1)

    play_replay_parser = subparsers.add_parser("play-replay", help="Replay macro-actions from replay.json in OpenTTD/FIRS.")
    play_replay_parser.add_argument("--replay", required=True)
    play_replay_parser.add_argument("--workbook", default=None)
    play_replay_parser.add_argument("--executable", default=None)
    play_replay_parser.add_argument("--openttd-user-dir", default=None)
    play_replay_parser.add_argument("--out", default="runs_replay")
    play_replay_parser.add_argument("--resolution", default="1280x720")
    play_replay_parser.add_argument("--record", dest="record", action="store_true", default=True)
    play_replay_parser.add_argument("--no-record", dest="record", action="store_false")
    play_replay_parser.add_argument("--record-source", default=None)
    play_replay_parser.add_argument("--sync-video", action="store_true", help="Block until 8x timelapse encoding finishes.")
    play_replay_parser.add_argument("--start-delay", type=float, default=10.0)
    play_replay_parser.add_argument("--action-delay", type=float, default=2.0)

    args = parser.parse_args(argv)
    if args.command == "list-scenarios":
        return _list_scenarios(args.scenario_file)
    if args.command == "list-openttd-scenarios":
        return _list_openttd_scenarios(args.scenario_file)
    if args.command == "list-procedural-scenarios":
        return _list_procedural_scenarios(args.split, args.count_per_family)
    if args.command == "eval":
        return _eval(args)
    if args.command == "summarize":
        return _summarize(args.runs_dir, as_json=args.json)
    if args.command == "benchmark-core":
        payload = run_core_benchmark(
            CoreBenchmarkConfig(
                suite=args.suite,
                split=args.split,
                agents=tuple(_split_csv(args.agents)),
                seeds=tuple(int(item) for item in _split_csv(args.seeds)),
                tasks=tuple(_split_csv(args.tasks)) if args.tasks else (),
                procedural_count_per_family=args.procedural_count_per_family,
                backend=args.backend,
                output_root=Path(args.out),
                max_steps=args.max_steps,
            )
        )
        print(json.dumps({"summary": str(Path(args.out) / "benchmark_summary.json"), "aggregate": payload["aggregate"]}, indent=2))
        return 0
    if args.command == "check-gym-env":
        return _check_gym_env(args)
    if args.command == "determinism-check":
        payload = run_determinism_check(
            DeterminismConfig(
                workbook=Path(args.workbook),
                task_id=args.scenario,
                executable=args.executable,
                openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
                output_root=Path(args.out),
                agent=args.agent,
                seed=args.seed,
                repeats=args.repeats,
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                trace_mode=args.trace_mode,
                fixed_action_script=not args.no_fixed_action_script,
                compare_runtime_lock=not args.no_runtime_lock_compare,
                progress_path=Path(args.progress_jsonl) if args.progress_jsonl else None,
            )
        )
        print(json.dumps(payload, indent=2))
        return 0 if payload["ok"] else 1
    if args.command == "benchmark-gym":
        payload = run_gym_baselines(
            GymBaselineConfig(
                workbook=Path(args.workbook),
                task_id=args.scenario,
                executable=args.executable,
                openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
                output_root=Path(args.out),
                agents=tuple(_split_csv(args.agents)),
                seeds=tuple(int(item) for item in _split_csv(args.seeds)),
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                deterministic=args.deterministic,
            )
        )
        print(json.dumps({"summary": str(Path(args.out) / "gym_benchmark_summary.json"), "aggregate": payload["aggregate"]}, indent=2))
        return 0
    if args.command == "train-rl-baselines":
        payload = run_rl_training(
            RLTrainingConfig(
                workbook=Path(args.workbook),
                task_id=args.scenario,
                algorithms=tuple(_split_csv(args.algorithms)),
                seeds=tuple(int(item) for item in _split_csv(args.seeds)),
                output_root=Path(args.out),
                executable=args.executable,
                openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                total_timesteps=args.total_timesteps,
                eval_interval=args.eval_interval,
                eval_episodes=args.eval_episodes,
            )
        )
        print(
            json.dumps(
                {
                    "report": payload["report"],
                    "artifacts": payload.get("artifacts", {}),
                    "aggregate": payload["aggregate"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "eval-rl-model":
        payload = run_rl_model_eval(
            RLModelEvalConfig(
                model=Path(args.model),
                workbook=Path(args.workbook),
                task_id=args.scenario,
                algorithm=args.algorithm,
                seeds=tuple(int(item) for item in _split_csv(args.seeds)),
                output_root=Path(args.out),
                executable=args.executable,
                openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
                max_candidates=args.max_candidates,
                max_steps=args.max_steps,
                eval_episodes=args.eval_episodes,
                deterministic_policy=not args.stochastic,
            )
        )
        print(
            json.dumps(
                {
                    "report": payload["artifacts"]["report"],
                    "artifacts": payload.get("artifacts", {}),
                    "aggregate": payload["aggregate"],
                },
                indent=2,
            )
        )
        return 0
    if args.command == "benchmark-validity-pack":
        payload = run_validity_pack(
            ValidityConfig(
                workbook=Path(args.workbook),
                suite_file=Path(args.suite_file) if args.suite_file else None,
                benchmark_file=Path(args.benchmark_file) if args.benchmark_file else None,
                tasks=tuple(_split_csv(args.tasks)) if args.tasks else (),
                agents=tuple(_split_csv(args.agents)) if args.agents else (),
                seeds=tuple(int(item) for item in _split_csv(args.seeds)) if args.seeds else (),
                executable=args.executable,
                openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
                output_root=Path(args.out),
                max_candidates=args.max_candidates,
                determinism_repeats=args.determinism_repeats,
                determinism_max_steps=args.determinism_max_steps,
                baseline_max_steps=args.baseline_max_steps,
                throughput_steps=args.throughput_steps,
                route_builder_attempts=args.route_builder_attempts,
                route_builder_wait_months=args.route_builder_wait_months,
                route_builder_target_success_rate=args.route_builder_target_success_rate,
                skip_determinism=args.skip_determinism,
                skip_baselines=args.skip_baselines,
                skip_throughput=args.skip_throughput,
                skip_route_builder=args.skip_route_builder,
            )
        )
        print(json.dumps({"ok": payload["ok"], "report": payload["report"], "sections": payload["sections"]}, indent=2))
        return 0 if payload["ok"] else 1
    if args.command == "build-benchmark-report":
        payload = write_benchmark_report(
            validity_report=Path(args.validity_report) if args.validity_report else None,
            training_report=Path(args.training_report) if args.training_report else None,
            route_builder_report=Path(args.route_builder_report) if args.route_builder_report else None,
            output_dir=Path(args.out),
            title=args.title,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "smoke-openttd":
        backend = OpenTTDBackend(executable=args.executable)
        if args.launch:
            scenario = load_registry().get(args.scenario)
            payload = backend.smoke_launch(scenario)
        else:
            payload = backend.smoke()
        if args.firs:
            payload["firs"] = _firs_readiness(args.openttd_user_dir)
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "install-bridge":
        target = install_bridge()
        payload = {"bridge_dir": str(target)}
        print(json.dumps(payload, indent=2) if args.json else f"Installed bridge: {target}")
        return 0
    if args.command == "watch-gpt":
        payload = launch_watch_game(
            executable=args.executable,
            output_root=Path(args.out),
            seed=args.seed,
            resolution=args.resolution,
            model=args.model,
            write_plan=not args.no_plan,
            launch=not args.no_launch,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "play-gpt-live":
        payload = launch_gpt_live(
            executable=args.executable,
            output_root=Path(args.out),
            seed=args.seed,
            steps=args.steps,
            resolution=args.resolution,
            model=args.model,
            allow_heuristic=args.allow_heuristic,
            focus_town_id=args.focus_town_id,
            start_delay=args.start_delay,
            step_delay=args.step_delay,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "play-coal-live":
        payload = launch_coal_objective(
            executable=args.executable,
            output_root=Path(args.out),
            seed=args.seed,
            steps=args.steps,
            resolution=args.resolution,
            model=args.model,
            allow_heuristic=args.allow_heuristic,
            start_delay=args.start_delay,
            step_delay=args.step_delay,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "firs-init-workbook":
        config = load_firs_config(args.config) if args.config else None
        path = create_firs_ops_workbook(Path(args.out), config=config)
        print(json.dumps({"workbook": str(path)}, indent=2))
        return 0
    if args.command == "play-firs-live":
        payload = launch_firs_live(
            workbook=Path(args.workbook),
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            output_root=Path(args.out),
            steps=args.steps,
            resolution=args.resolution,
            model=args.model,
            record=args.record,
            record_source=args.record_source,
            repl=args.repl,
            allow_heuristic=args.allow_heuristic,
            start_delay=args.start_delay,
            step_delay=args.step_delay,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "firs-research-run":
        payload = launch_firs_research(
            workbook=Path(args.workbook),
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            output_root=Path(args.out),
            steps=args.steps,
            model=args.model,
            benchmark_task=args.task,
            benchmark_file=Path(args.benchmark_file) if args.benchmark_file else None,
            allow_heuristic=args.allow_heuristic,
            step_delay=args.step_delay,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "firs-benchmark":
        payload = launch_firs_benchmark(
            workbook=Path(args.workbook),
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            output_root=Path(args.out),
            models=[item.strip() for item in args.models.split(",") if item.strip()],
            tasks=[item.strip() for item in args.tasks.split(",") if item.strip()] if args.tasks else None,
            repeats=args.repeats,
            benchmark_file=Path(args.benchmark_file) if args.benchmark_file else None,
            allow_heuristic=args.allow_heuristic,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "benchmark-route-builder":
        payload = launch_route_builder_benchmark(
            workbook=Path(args.workbook),
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            output_root=Path(args.out),
            seed=args.seed,
            economy=args.economy,
            attempts=args.attempts,
            vehicles=args.vehicles,
            wait_months=args.wait_months,
            max_path_tiles=args.max_path_tiles,
            target_success_rate=args.target_success_rate,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "export-xlsx":
        path = export_run_to_xlsx(Path(args.run), Path(args.out), source_workbook=args.workbook)
        print(json.dumps({"report": str(path)}, indent=2))
        return 0
    if args.command == "export-dataset":
        path = export_core_dataset(Path(args.run), Path(args.out), output_format=args.format)
        print(json.dumps({"dataset": str(path)}, indent=2))
        return 0
    if args.command == "export-replay":
        path = export_replay(Path(args.run), Path(args.out) if args.out else None)
        print(json.dumps({"replay": str(path)}, indent=2))
        return 0
    if args.command == "render-core-replay":
        payload = render_core_replay(
            episode=Path(args.episode) if args.episode else None,
            replay=Path(args.replay) if args.replay else None,
            out=Path(args.out),
            fps=args.fps,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "play-replay":
        payload = launch_firs_replay(
            replay=Path(args.replay),
            workbook=Path(args.workbook) if args.workbook else None,
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            output_root=Path(args.out),
            resolution=args.resolution,
            record=args.record,
            record_source=args.record_source,
            async_video=not args.sync_video,
            start_delay=args.start_delay,
            action_delay=args.action_delay,
        )
        print(json.dumps(payload, indent=2))
        return 0
    raise AssertionError(args.command)


def _list_scenarios(scenario_file: str | None) -> int:
    registry = load_registry(scenario_file)
    for scenario in registry.list():
        print(f"{scenario.id}\t{scenario.name}\t{scenario.task}")
    return 0


def _list_openttd_scenarios(scenario_file: str | None) -> int:
    for task in load_benchmark_tasks(scenario_file):
        success = ",".join(sorted(task.success)) if task.success else "-"
        print(
            f"{task.id}\t{task.split}\t{task.difficulty}\t{task.mode}\t"
            f"seed={task.seed}\teconomy={task.economy}\tsteps={task.steps}\t"
            f"success={success}\t{task.description}"
        )
    return 0


def _list_procedural_scenarios(split: str, count_per_family: int) -> int:
    scenarios = generate_procedural_scenarios(split=split, families=PROCEDURAL_FAMILIES, count_per_family=count_per_family)
    for scenario in scenarios:
        print(f"{scenario.id}\t{scenario.name}\t{scenario.task}")
    return 0


def _check_gym_env(args: argparse.Namespace) -> int:
    try:
        import gymnasium as gym
        from gymnasium.utils.env_checker import check_env
    except ImportError as exc:
        raise SystemExit("Gymnasium is not installed. Install with: python -m pip install -e .[gymnasium]") from exc

    from openttd_le.adapters.gymnasium import (
        FIRS_DETERMINISTIC_GYM_ID,
        FIRS_GYM_ID,
        TOY_GYM_ID,
        OpenTTDFIRSGymEnv,
        OpenTTDLEGymEnv,
        register_envs,
    )

    register_envs()
    scenario = args.scenario or ("lab_raw_to_processor" if args.backend == "openttd" else "coal_easy_001")
    max_steps = max(2, int(args.max_steps or 2))
    if args.env_id:
        env_kwargs: dict[str, Any] = {"max_candidates": args.max_candidates}
        if args.env_id == FIRS_GYM_ID or args.env_id == FIRS_DETERMINISTIC_GYM_ID or "FIRS" in args.env_id:
            env_kwargs.update(
                {
                    "workbook": args.workbook,
                    "task_id": scenario,
                    "executable": args.executable,
                    "openttd_user_dir": Path(args.openttd_user_dir) if args.openttd_user_dir else None,
                    "max_steps": max_steps,
                    "deterministic": args.deterministic or args.env_id == FIRS_DETERMINISTIC_GYM_ID,
                }
            )
        elif args.env_id == TOY_GYM_ID:
            env_kwargs.update({"task_id": scenario})
        env = gym.make(args.env_id, **env_kwargs)
        env_name = args.env_id
    elif args.backend == "openttd":
        env = OpenTTDFIRSGymEnv(
            workbook=args.workbook,
            task_id=scenario,
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            max_candidates=args.max_candidates,
            max_steps=max_steps,
            deterministic=args.deterministic,
        )
        env_name = FIRS_DETERMINISTIC_GYM_ID if args.deterministic else FIRS_GYM_ID
    else:
        env = OpenTTDLEGymEnv(task_id=scenario, max_candidates=args.max_candidates)
        env_name = TOY_GYM_ID

    try:
        target_env = env.unwrapped if hasattr(env, "unwrapped") else env
        check_env(target_env, skip_render_check=args.skip_render_check)
        obs, info = env.reset(seed=1)
        masks = env.action_masks() if hasattr(env, "action_masks") else info.get("action_mask")
        payload = {
            "ok": True,
            "env_id": env_name,
            "backend": args.backend,
            "observation_keys": sorted(obs.keys()) if isinstance(obs, dict) else [],
            "action_space": str(env.action_space),
            "observation_space": str(env.observation_space),
            "action_mask_length": len(masks) if masks is not None else None,
            "candidate_actions": len(info.get("candidate_actions", []) or []),
        }
        print(json.dumps(payload, indent=2))
        return 0
    finally:
        env.close()


def _split_csv(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def _summarize(runs_dir: str, as_json: bool = False) -> int:
    root = Path(runs_dir)
    summaries = []
    for path in root.glob("*/summary.json"):
        with path.open("r", encoding="utf-8") as handle:
            summaries.append(json.load(handle))
    summaries.sort(key=lambda item: (item["scenario_id"], item["agent"], item["seed"]))
    if as_json:
        print(json.dumps({"runs": summaries}, indent=2))
        return 0
    if not summaries:
        print(f"No summary.json files found under {root}.")
        return 0
    print("scenario\tagent\tseed\tscore\tcargo\tprofit\tinvalid")
    for item in summaries:
        print(
            f"{item['scenario_id']}\t{item['agent']}\t{item['seed']}\t"
            f"{item['score']}\t{item['cargo_delivered']}\t"
            f"{round(item['operating_profit'], 2)}\t{item['invalid_actions']}"
        )
    avg_score = sum(item["score"] for item in summaries) / len(summaries)
    print(f"avg_score\t{round(avg_score, 3)}")
    return 0


def _firs_readiness(openttd_user_dir: str | None) -> dict[str, Any]:
    user_dir = Path(openttd_user_dir).expanduser().resolve() if openttd_user_dir else None
    if user_dir is not None:
        user_dir.mkdir(parents=True, exist_ok=True)
        os.environ["OPENTTD_USER_DIR"] = str(user_dir)
    payload: dict[str, Any] = {"ready": False}
    try:
        payload["opengfx"] = str(ensure_opengfx())
        payload["bridge"] = install_live_bridge()
        install = verify_firs_installed(user_dir)
        payload["firs_newgrf"] = str(install.newgrf_path)
        payload["openttd_user_dir"] = str(install.user_dir)
        payload["ready"] = True
    except Exception as exc:
        payload["error"] = str(exc)
    return payload


def _eval(args: argparse.Namespace) -> int:
    if args.backend == "openttd":
        return _eval_openttd(args)
    return _eval_toy(args)


def _eval_openttd(args: argparse.Namespace) -> int:
    if args.agent not in {"openai", "heuristic"}:
        raise SystemExit(
            "Real OpenTTD eval currently supports --agent openai or --agent heuristic. "
            "Use --backend toy only for local mock-backend debugging."
        )
    summaries = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        agent = make_firs_agent(args.agent, model=args.model)
        env = OpenTTDFIRSEnv(
            workbook=Path(args.workbook),
            executable=args.executable,
            openttd_user_dir=Path(args.openttd_user_dir) if args.openttd_user_dir else None,
            output_root=Path(args.out),
            task_id=args.scenario,
            benchmark_file=Path(args.scenario_file) if args.scenario_file else None,
            seed=seed,
            max_steps=args.max_steps,
        )
        try:
            observation, info = env.reset()
            run_dir = Path(info["run_dir"])
            trace_path = run_dir / "firs_trace.jsonl"
            observations_path = run_dir / "observations.jsonl"
            rewards_path = run_dir / "rewards.jsonl"
            actions_path = run_dir / "actions.jsonl"
            summary_path = run_dir / "summary.json"
            report_path = run_dir / "report.xlsx"
            replay_path = run_dir / "replay.json"

            with (
                trace_path.open("a", encoding="utf-8") as trace,
                observations_path.open("a", encoding="utf-8") as observations_file,
                rewards_path.open("a", encoding="utf-8") as rewards_file,
                actions_path.open("a", encoding="utf-8") as actions_file,
            ):
                _write_jsonl_event(trace, "initial_observation", 0, observation)
                _write_jsonl(observations_file, {"step": 0, "observation": observation})
                for step in range(1, env.max_steps + 1):
                    action = agent.act(observation)
                    observation, _reward, terminated, truncated, step_info = env.step(action)
                    for executed in step_info.get("actions", []):
                        _write_jsonl(actions_file, {"step": step, **executed})
                        _write_jsonl_event(trace, "action", step, executed["action"])
                        _write_jsonl_event(trace, "result", step, executed["result"])
                        _write_jsonl_event(trace, "observation", step, executed["observation"])
                    reward_details = step_info.get("reward_details") or {}
                    _write_jsonl(rewards_file, {"step": step, **reward_details, "snapshot": step_info.get("snapshot")})
                    _write_jsonl(observations_file, {"step": step, "observation": observation})
                    if terminated or truncated:
                        break
                    if args.step_delay > 0:
                        time.sleep(args.step_delay)

            summary = env.summary(agent=args.agent, model=args.model or ("gpt-5.5" if args.agent == "openai" else None))
            summary.update(
                {
                    "run_index": run_index + 1,
                    "trace": str(trace_path),
                    "observations": str(observations_path),
                    "rewards": str(rewards_path),
                    "actions": str(actions_path),
                    "summary": str(summary_path),
                    "report": str(report_path),
                    "replay": str(replay_path),
                }
            )
            summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
            launch_info = env.launch_info(summary_path=summary_path)
            launch_info.update(
                {
                    "agent": args.agent,
                    "model": summary.get("model"),
                    "trace": str(trace_path),
                    "observations": str(observations_path),
                    "rewards": str(rewards_path),
                    "actions": str(actions_path),
                    "report": str(report_path),
                    "replay": str(replay_path),
                }
            )
            (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
            export_run_to_xlsx(run_dir, report_path, source_workbook=args.workbook)
            export_replay(run_dir, replay_path)
            summaries.append(summary)
            print(json.dumps(summary, separators=(",", ":")))
        finally:
            agent.close()
            env.close()

    if len(summaries) > 1:
        completed = sum(1 for item in summaries if item.get("completed"))
        avg_reward = sum(float(item.get("total_reward", 0) or 0) for item in summaries) / len(summaries)
        print(
            json.dumps(
                {
                    "runs": len(summaries),
                    "completed": completed,
                    "completion_rate": round(completed / len(summaries), 3),
                    "avg_total_reward": round(avg_reward, 3),
                },
                separators=(",", ":"),
            )
        )
    return 0


def _write_jsonl_event(handle: Any, event: str, step: int, data: dict[str, Any]) -> None:
    _write_jsonl(handle, {"event": event, "step": step, "data": data})


def _write_jsonl(handle: Any, data: dict[str, Any]) -> None:
    handle.write(json.dumps(data, separators=(",", ":")) + "\n")
    handle.flush()


def _eval_toy(args: argparse.Namespace) -> int:
    registry = load_registry(args.scenario_file)
    output_root = Path(args.out)
    summaries = []
    for run_index in range(args.runs):
        seed = args.seed + run_index
        agent = make_agent(args.agent, model=args.model, seed=seed)
        env = OpenTTDLEnv(backend=args.backend, registry=registry)
        artifacts = RunArtifacts(output_root, args.scenario, args.agent, seed)
        observation, _ = env.reset(args.scenario, seed=seed)
        max_steps = args.max_steps or observation["time"]["max_steps"]
        action_count = 0
        try:
            while action_count < max_steps:
                previous_observation = observation
                candidate_actions = env.candidate_actions()
                action = agent.act(observation)
                preview = env.preview(action)
                result = env.step(action)
                action_count += 1
                observation = result.observation
                artifacts.log_step(
                    action_count,
                    observation,
                    action,
                    result.reward,
                    result.info,
                    previous_observation=previous_observation,
                    candidate_actions=candidate_actions,
                    preview=preview,
                )
                if result.terminated or result.truncated:
                    break
            summary = _summary(args, seed, action_count, observation, artifacts.run_dir)
            artifacts.write_final(summary, env.artifact_state(), observation)
            summaries.append(summary)
            print(json.dumps(summary, separators=(",", ":")))
        finally:
            agent.close()
            env.close()

    if len(summaries) > 1:
        avg_score = sum(item["score"] for item in summaries) / len(summaries)
        print(json.dumps({"runs": len(summaries), "avg_score": round(avg_score, 3)}, separators=(",", ":")))
    return 0


def _summary(
    args: argparse.Namespace,
    seed: int,
    action_count: int,
    observation: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    metrics = observation["metrics"]
    return {
        "run_id": run_dir.name,
        "scenario_id": args.scenario,
        "agent": args.agent,
        "model": args.model,
        "backend": args.backend,
        "seed": seed,
        "steps": action_count,
        "month": observation["time"]["month"],
        "score": metrics["score"],
        "cargo_delivered": metrics["cargo_delivered"],
        "operating_profit": metrics["operating_profit"],
        "cash": observation["company"]["cash"],
        "loan": observation["company"]["loan"],
        "invalid_actions": metrics["invalid_actions"],
        "run_dir": str(run_dir),
    }


if __name__ == "__main__":
    raise SystemExit(main())
