from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv
from openttd_le.backends.live import launch_route_builder_benchmark
from openttd_le.research.benchmarks import BenchmarkTask, load_benchmark_tasks
from openttd_le.research.determinism import DeterminismConfig, run_determinism_check
from openttd_le.research.gym_baselines import GYM_BASELINE_AGENTS, GymBaselineConfig, run_gym_baselines, select_baseline_action
from openttd_le.research.reporting import write_benchmark_report


VALIDITY_REPORT_SCHEMA = "openttd-le-firs-validity-report-v1"
VALIDITY_SUITE_SCHEMA = "openttd-le-firs-validity-suite-v1"


@dataclass(frozen=True)
class ValiditySuite:
    name: str
    benchmark_file: Path
    tasks: tuple[str, ...]
    agents: tuple[str, ...]
    seeds: tuple[int, ...]
    gates: dict[str, Any]
    description: str = ""


@dataclass(frozen=True)
class ValidityConfig:
    workbook: Path
    suite_file: Path | None = None
    benchmark_file: Path | None = None
    tasks: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    seeds: tuple[int, ...] = ()
    output_root: Path = Path("runs_validity")
    executable: str | None = None
    openttd_user_dir: Path | None = None
    max_candidates: int = 24
    determinism_repeats: int | None = None
    determinism_max_steps: int | None = None
    baseline_max_steps: int | None = None
    throughput_steps: int | None = None
    route_builder_attempts: int | None = None
    route_builder_wait_months: int = 6
    route_builder_target_success_rate: float | None = None
    skip_determinism: bool = False
    skip_baselines: bool = False
    skip_throughput: bool = False
    skip_route_builder: bool = False


def default_validity_suite_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scenarios" / "firs_validity_suite.json"


def load_validity_suite(path: Path | str | None = None) -> ValiditySuite:
    suite_path = Path(path) if path else default_validity_suite_path()
    payload = json.loads(suite_path.read_text(encoding="utf-8"))
    if payload.get("schema") != VALIDITY_SUITE_SCHEMA:
        raise ValueError(f"Unsupported validity suite schema in {suite_path}: {payload.get('schema')}")
    benchmark_file = Path(str(payload.get("benchmark_file") or "scenarios/firs_benchmarks.json"))
    if not benchmark_file.is_absolute():
        benchmark_file = suite_path.parents[1] / benchmark_file
    return ValiditySuite(
        name=str(payload.get("name") or suite_path.stem),
        description=str(payload.get("description") or ""),
        benchmark_file=benchmark_file,
        tasks=tuple(str(item) for item in payload.get("tasks", [])),
        agents=tuple(str(item) for item in payload.get("default_agents", [])),
        seeds=tuple(int(item) for item in payload.get("default_seeds", [1])),
        gates=dict(payload.get("gates", {})),
    )


def run_validity_pack(config: ValidityConfig) -> dict[str, Any]:
    suite = load_validity_suite(config.suite_file)
    benchmark_file = config.benchmark_file or suite.benchmark_file
    tasks = _resolve_tasks(config.tasks or suite.tasks, benchmark_file)
    agents = _validate_agents(config.agents or suite.agents or ("masked_random", "first_valid"))
    seeds = config.seeds or suite.seeds or (1,)
    gates = dict(suite.gates)

    determinism_repeats = int(config.determinism_repeats or gates.get("determinism_repeats") or 3)
    determinism_max_steps = int(config.determinism_max_steps or gates.get("determinism_max_steps") or 3)
    baseline_max_steps = int(config.baseline_max_steps or gates.get("baseline_max_steps") or 8)
    throughput_steps = int(config.throughput_steps or gates.get("throughput_steps") or 3)
    route_builder_attempts = int(config.route_builder_attempts if config.route_builder_attempts is not None else gates.get("route_builder_attempts", 20))
    target_success_rate = float(
        config.route_builder_target_success_rate
        if config.route_builder_target_success_rate is not None
        else gates.get("route_builder_target_success_rate", 0.9)
    )

    config.output_root.mkdir(parents=True, exist_ok=True)
    manifest = {
        "schema": VALIDITY_REPORT_SCHEMA,
        "suite": suite.name,
        "description": suite.description,
        "benchmark_file": str(benchmark_file),
        "workbook": str(config.workbook),
        "tasks": [task.id for task in tasks],
        "task_metadata": [
            {
                "id": task.id,
                "split": task.split,
                "difficulty": task.difficulty,
                "mode": task.mode,
                "tags": list(task.tags),
                "steps": task.steps,
            }
            for task in tasks
        ],
        "agents": list(agents),
        "seeds": list(seeds),
        "gates": {
            "determinism_repeats": determinism_repeats,
            "determinism_max_steps": determinism_max_steps,
            "baseline_max_steps": baseline_max_steps,
            "throughput_steps": throughput_steps,
            "route_builder_attempts": route_builder_attempts,
            "route_builder_wait_months": config.route_builder_wait_months,
            "route_builder_target_success_rate": target_success_rate,
        },
    }
    (config.output_root / "suite_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    determinism_reports = []
    if not config.skip_determinism:
        for task in tasks:
            for seed in seeds:
                out = config.output_root / "determinism" / task.id / f"seed_{seed}"
                determinism_reports.append(
                    run_determinism_check(
                        DeterminismConfig(
                            workbook=config.workbook,
                            task_id=task.id,
                            agent="first_valid",
                            seed=seed,
                            repeats=determinism_repeats,
                            output_root=out,
                            executable=config.executable,
                            openttd_user_dir=config.openttd_user_dir,
                            max_candidates=config.max_candidates,
                            max_steps=determinism_max_steps,
                        )
                    )
                )

    baseline_reports = []
    if not config.skip_baselines:
        for task in tasks:
            out = config.output_root / "baselines" / task.id
            baseline_reports.append(
                run_gym_baselines(
                    GymBaselineConfig(
                        workbook=config.workbook,
                        task_id=task.id,
                        agents=agents,
                        seeds=seeds,
                        output_root=out,
                        executable=config.executable,
                        openttd_user_dir=config.openttd_user_dir,
                        max_candidates=config.max_candidates,
                        max_steps=baseline_max_steps,
                        deterministic=True,
                    )
                )
            )

    throughput_reports = []
    if not config.skip_throughput:
        for task in tasks:
            for seed in seeds:
                throughput_reports.append(
                    measure_gym_throughput(
                        workbook=config.workbook,
                        task_id=task.id,
                        seed=seed,
                        output_root=config.output_root / "throughput" / task.id / f"seed_{seed}",
                        executable=config.executable,
                        openttd_user_dir=config.openttd_user_dir,
                        max_candidates=config.max_candidates,
                        max_steps=throughput_steps,
                    )
                )

    route_builder_reports = []
    if not config.skip_route_builder and route_builder_attempts > 0:
        seen_seed_economies = sorted({(seed, task.economy) for task in tasks for seed in seeds})
        for seed, economy in seen_seed_economies:
            route_builder_reports.append(
                launch_route_builder_benchmark(
                    workbook=config.workbook,
                    executable=config.executable,
                    openttd_user_dir=config.openttd_user_dir,
                    output_root=config.output_root / "route_builder",
                    seed=seed,
                    economy=economy,
                    attempts=route_builder_attempts,
                    wait_months=config.route_builder_wait_months,
                    target_success_rate=target_success_rate,
                )
            )

    report = {
        **manifest,
        "sections": {
            "determinism": _summarize_determinism(determinism_reports, skipped=config.skip_determinism),
            "baselines": _summarize_baselines(baseline_reports, skipped=config.skip_baselines),
            "throughput": _summarize_throughput(throughput_reports, skipped=config.skip_throughput),
            "route_builder": _summarize_route_builder(route_builder_reports, skipped=config.skip_route_builder or route_builder_attempts <= 0),
        },
        "artifacts": {
            "suite_manifest": str(config.output_root / "suite_manifest.json"),
            "determinism_dir": str(config.output_root / "determinism"),
            "baselines_dir": str(config.output_root / "baselines"),
            "throughput_dir": str(config.output_root / "throughput"),
            "route_builder_dir": str(config.output_root / "route_builder"),
        },
    }
    report["ok"] = all(section.get("ok", True) for section in report["sections"].values())
    report_path = config.output_root / "validity_report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["report"] = str(report_path)
    report_artifacts = write_benchmark_report(validity_report=report, output_dir=config.output_root)
    report["artifacts"]["benchmark_report"] = report_artifacts["report"]
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def measure_gym_throughput(
    *,
    workbook: Path,
    task_id: str,
    seed: int,
    output_root: Path,
    executable: str | None,
    openttd_user_dir: Path | None,
    max_candidates: int,
    max_steps: int,
) -> dict[str, Any]:
    output_root.mkdir(parents=True, exist_ok=True)
    env = OpenTTDFIRSGymEnv(
        workbook=str(workbook),
        task_id=task_id,
        executable=executable,
        openttd_user_dir=openttd_user_dir,
        output_root=output_root,
        max_candidates=max_candidates,
        max_steps=max_steps,
        deterministic=True,
    )
    step_seconds: list[float] = []
    total_reward = 0.0
    try:
        reset_start = time.perf_counter()
        observation, info = env.reset(seed=seed)
        reset_seconds = time.perf_counter() - reset_start
        for step in range(max_steps):
            action_index = select_baseline_action("first_valid", observation, info, random.Random(seed + step))
            step_start = time.perf_counter()
            observation, reward, terminated, truncated, info = env.step(action_index)
            step_seconds.append(time.perf_counter() - step_start)
            total_reward += float(reward)
            if terminated or truncated:
                break
        payload = {
            "schema": "openttd-le-throughput-report-v1",
            "task_id": task_id,
            "seed": seed,
            "run_dir": str(env.env.run_dir) if getattr(env.env, "run_dir", None) else None,
            "reset_seconds": round(reset_seconds, 3),
            "steps": len(step_seconds),
            "total_reward": round(total_reward, 3),
            "step_seconds": [round(item, 3) for item in step_seconds],
            "median_step_seconds": round(median(step_seconds), 3) if step_seconds else 0.0,
            "transitions_per_hour": round(3600.0 / median(step_seconds), 3) if step_seconds and median(step_seconds) > 0 else 0.0,
        }
        (output_root / "throughput.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload
    finally:
        env.close()


def _resolve_tasks(task_ids: tuple[str, ...], benchmark_file: Path | str | None) -> list[BenchmarkTask]:
    available = {task.id: task for task in load_benchmark_tasks(benchmark_file)}
    missing = [task_id for task_id in task_ids if task_id not in available]
    if missing:
        raise ValueError(f"Unknown validity task(s): {', '.join(missing)}")
    return [available[task_id] for task_id in task_ids]


def _validate_agents(agents: tuple[str, ...]) -> tuple[str, ...]:
    invalid = [agent for agent in agents if agent not in GYM_BASELINE_AGENTS]
    if invalid:
        raise ValueError(f"Unknown baseline agent(s): {', '.join(invalid)}. Choices: {', '.join(GYM_BASELINE_AGENTS)}")
    return agents


def _summarize_determinism(reports: list[dict[str, Any]], *, skipped: bool) -> dict[str, Any]:
    if skipped:
        return {"skipped": True, "ok": True}
    passed = sum(1 for report in reports if report.get("ok"))
    return {"ok": passed == len(reports), "runs": len(reports), "passed": passed}


def _summarize_baselines(reports: list[dict[str, Any]], *, skipped: bool) -> dict[str, Any]:
    if skipped:
        return {"skipped": True, "ok": True}
    runs = sum(int(report.get("aggregate", {}).get("runs", 0) or 0) for report in reports)
    per_task = {str(report.get("task_id")): report.get("aggregate", {}) for report in reports}
    return {"ok": runs > 0, "runs": runs, "per_task": per_task}


def _summarize_throughput(reports: list[dict[str, Any]], *, skipped: bool) -> dict[str, Any]:
    if skipped:
        return {"skipped": True, "ok": True}
    medians = [float(report.get("median_step_seconds", 0) or 0) for report in reports if report.get("steps")]
    return {
        "ok": bool(reports),
        "runs": len(reports),
        "median_step_seconds": round(median(medians), 3) if medians else 0.0,
        "median_transitions_per_hour": round(3600.0 / median(medians), 3) if medians and median(medians) > 0 else 0.0,
    }


def _summarize_route_builder(reports: list[dict[str, Any]], *, skipped: bool) -> dict[str, Any]:
    if skipped:
        return {"skipped": True, "ok": True}
    rates = [float(report.get("aggregate", {}).get("operational_success_rate", 0) or 0) for report in reports]
    feasible_rates = [
        float(report.get("aggregate", {}).get("feasible_operational_success_rate", 0) or 0)
        for report in reports
    ]
    passes = sum(1 for report in reports if report.get("aggregate", {}).get("level1_pass"))
    feasible_passes = sum(1 for report in reports if report.get("aggregate", {}).get("feasible_level1_pass"))
    return {
        "ok": bool(reports) and passes == len(reports),
        "runs": len(reports),
        "passed": passes,
        "feasible_passed": feasible_passes,
        "median_operational_success_rate": round(median(rates), 3) if rates else 0.0,
        "median_feasible_operational_success_rate": round(median(feasible_rates), 3) if feasible_rates else 0.0,
    }
