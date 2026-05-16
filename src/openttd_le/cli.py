from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from openttd_le import __version__
from openttd_le.agents import make_agent
from openttd_le.backends.firs import load_firs_config
from openttd_le.backends.live import launch_coal_objective, launch_firs_live, launch_firs_research, launch_gpt_live
from openttd_le.backends.openttd import OpenTTDBackend
from openttd_le.backends.visual import install_bridge, launch_watch_game
from openttd_le.core.artifacts import RunArtifacts
from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.scenarios import load_registry
from openttd_le.workbooks.export import export_run_to_xlsx
from openttd_le.workbooks.template import create_firs_ops_workbook


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openttd-le")
    parser.add_argument("--version", action="version", version=f"openttd-le {__version__}")
    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-scenarios", help="List bundled scenarios.")
    list_parser.add_argument("--scenario-file", default=None)

    eval_parser = subparsers.add_parser("eval", help="Run an agent on a scenario.")
    eval_parser.add_argument("--scenario", required=True)
    eval_parser.add_argument("--agent", choices=["random", "greedy", "openai", "openrouter"], default="greedy")
    eval_parser.add_argument("--model", default=None)
    eval_parser.add_argument("--backend", choices=["toy", "openttd"], default="toy")
    eval_parser.add_argument("--scenario-file", default=None)
    eval_parser.add_argument("--runs", type=int, default=1)
    eval_parser.add_argument("--seed", type=int, default=1)
    eval_parser.add_argument("--out", default="runs")
    eval_parser.add_argument("--max-steps", type=int, default=None)

    summary_parser = subparsers.add_parser("summarize", help="Summarize run artifacts.")
    summary_parser.add_argument("runs_dir", nargs="?", default="runs")
    summary_parser.add_argument("--json", action="store_true", help="Emit JSON instead of a text table.")

    smoke_parser = subparsers.add_parser("smoke-openttd", help="Check real OpenTTD executable integration.")
    smoke_parser.add_argument("--executable", default=None)
    smoke_parser.add_argument("--launch", action="store_true", help="Start a short dedicated-server smoke run.")
    smoke_parser.add_argument("--scenario", default="coal_easy_001")

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
    firs_research_parser.add_argument("--step-delay", type=float, default=0.0)
    firs_research_parser.add_argument(
        "--allow-heuristic",
        action="store_true",
        help="Use a deterministic local policy when OPENAI_API_KEY is unavailable.",
    )

    export_parser = subparsers.add_parser("export-xlsx", help="Export a run directory to a FIRS Excel report.")
    export_parser.add_argument("--run", required=True)
    export_parser.add_argument("--out", required=True)
    export_parser.add_argument("--workbook", default=None)

    args = parser.parse_args(argv)
    if args.command == "list-scenarios":
        return _list_scenarios(args.scenario_file)
    if args.command == "eval":
        return _eval(args)
    if args.command == "summarize":
        return _summarize(args.runs_dir, as_json=args.json)
    if args.command == "smoke-openttd":
        backend = OpenTTDBackend(executable=args.executable)
        if args.launch:
            scenario = load_registry().get(args.scenario)
            print(json.dumps(backend.smoke_launch(scenario), indent=2))
        else:
            print(json.dumps(backend.smoke(), indent=2))
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
            allow_heuristic=args.allow_heuristic,
            step_delay=args.step_delay,
        )
        print(json.dumps(payload, indent=2))
        return 0
    if args.command == "export-xlsx":
        path = export_run_to_xlsx(Path(args.run), Path(args.out), source_workbook=args.workbook)
        print(json.dumps({"report": str(path)}, indent=2))
        return 0
    raise AssertionError(args.command)


def _list_scenarios(scenario_file: str | None) -> int:
    registry = load_registry(scenario_file)
    for scenario in registry.list():
        print(f"{scenario.id}\t{scenario.name}\t{scenario.task}")
    return 0


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


def _eval(args: argparse.Namespace) -> int:
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
                action = agent.act(observation)
                result = env.step(action)
                action_count += 1
                observation = result.observation
                artifacts.log_step(action_count, observation, action, result.reward, result.info)
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
