from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean, median, pstdev
from typing import Any

from openttd_le.agents import make_agent
from openttd_le.core.artifacts import RunArtifacts
from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.procedural import (
    DEFAULT_PROCEDURAL_COUNT_PER_FAMILY,
    PROCEDURAL_FAMILIES,
    generate_procedural_scenarios,
)
from openttd_le.core.scenarios import ScenarioRegistry, load_registry
from openttd_le.core.schemas import schema_manifest


CORE_SUITE_TASKS = [
    "coal_easy_001",
    "passenger_pair_001",
    "wood_mill_001",
    "low_cash_recovery_001",
    "multi_cargo_001",
]


@dataclass(frozen=True)
class CoreBenchmarkConfig:
    suite: str = "core"
    split: str = "dev"
    agents: tuple[str, ...] = ("random", "greedy", "candidate_rank", "preview_rerank")
    seeds: tuple[int, ...] = (1, 2, 3)
    tasks: tuple[str, ...] = ()
    procedural_count_per_family: int = DEFAULT_PROCEDURAL_COUNT_PER_FAMILY
    backend: str = "toy"
    output_root: Path = Path("runs_core")
    max_steps: int | None = None


def run_core_benchmark(config: CoreBenchmarkConfig) -> dict[str, Any]:
    registry = benchmark_registry(config)
    task_ids = benchmark_task_ids(config, registry)
    summaries: list[dict[str, Any]] = []
    for task_id in task_ids:
        scenario = registry.get(task_id)
        for agent_name in config.agents:
            for seed in config.seeds:
                agent = make_agent(agent_name, seed=seed)
                env = OpenTTDLEnv(backend=config.backend, registry=registry)
                artifacts = RunArtifacts(config.output_root, task_id, agent_name, seed)
                observation, _ = env.reset(task_id, seed=seed)
                max_steps = config.max_steps or scenario.budget.max_steps
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
                    summary = _summary(task_id, agent_name, seed, config.backend, action_count, observation, artifacts.run_dir)
                    artifacts.write_final(summary, env.artifact_state(), observation)
                    summaries.append(summary)
                finally:
                    agent.close()
                    env.close()

    aggregate = aggregate_core_benchmark(summaries)
    payload = {
        "schema": "openttd-le-core-benchmark-v1",
        "schemas": schema_manifest(),
        "suite": config.suite,
        "split": config.split,
        "backend": config.backend,
        "tasks": list(task_ids),
        "agents": list(config.agents),
        "seeds": list(config.seeds),
        "procedural": {
            "families": list(PROCEDURAL_FAMILIES) if config.suite == "procedural" else [],
            "count_per_family": config.procedural_count_per_family if config.suite == "procedural" else 0,
        },
        "runs": summaries,
        "aggregate": aggregate,
    }
    config.output_root.mkdir(parents=True, exist_ok=True)
    (config.output_root / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def benchmark_registry(config: CoreBenchmarkConfig) -> ScenarioRegistry:
    registry = load_registry()
    if config.suite == "procedural":
        return registry.extend(
            generate_procedural_scenarios(
                split=config.split,
                count_per_family=config.procedural_count_per_family,
            )
        )
    if config.suite == "core":
        return registry
    raise ValueError(f"Unknown benchmark suite: {config.suite}")


def benchmark_task_ids(config: CoreBenchmarkConfig, registry: ScenarioRegistry | None = None) -> tuple[str, ...]:
    if config.tasks:
        return config.tasks
    if config.suite == "core":
        return tuple(CORE_SUITE_TASKS)
    if config.suite == "procedural":
        scenarios = generate_procedural_scenarios(
            split=config.split,
            count_per_family=config.procedural_count_per_family,
        )
        return tuple(scenario.id for scenario in scenarios)
    if registry is not None:
        return tuple(scenario.id for scenario in registry.list())
    raise ValueError(f"Unknown benchmark suite: {config.suite}")


def aggregate_core_benchmark(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        groups.setdefault(str(summary["agent"]), []).append(summary)
    return {agent: _aggregate_group(rows) for agent, rows in sorted(groups.items())}


def _aggregate_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    scores = [float(row.get("score", 0) or 0) for row in rows]
    cargo = [float(row.get("cargo_delivered", 0) or 0) for row in rows]
    invalid = [float(row.get("invalid_actions", 0) or 0) for row in rows]
    completed = sum(1 for row in rows if _completed(row))
    return {
        "runs": len(rows),
        "success_rate": round(completed / len(rows), 3) if rows else 0.0,
        "mean_score": round(mean(scores), 3) if scores else 0.0,
        "median_score": round(median(scores), 3) if scores else 0.0,
        "std_score": round(pstdev(scores), 3) if len(scores) > 1 else 0.0,
        "mean_cargo_delivered": round(mean(cargo), 3) if cargo else 0.0,
        "mean_invalid_actions": round(mean(invalid), 3) if invalid else 0.0,
    }


def _summary(
    task_id: str,
    agent_name: str,
    seed: int,
    backend: str,
    action_count: int,
    observation: dict[str, Any],
    run_dir: Path,
) -> dict[str, Any]:
    metrics = observation["metrics"]
    return {
        "run_id": run_dir.name,
        "schema": "openttd-le-run-summary-v1",
        "schemas": schema_manifest(),
        "scenario_id": task_id,
        "agent": agent_name,
        "model": None,
        "backend": backend,
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


def _completed(summary: dict[str, Any]) -> bool:
    return float(summary.get("score", 0) or 0) >= 50.0 and int(summary.get("invalid_actions", 0) or 0) == 0
