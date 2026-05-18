from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


ROUTE_BUILDER_INFEASIBLE_REASONS = {
    "no_source_station_site",
    "no_destination_station_site",
    "no_path_between_station_candidates",
    "path_too_long_for_single_macro",
    "no_road_path",
    "no_matching_cargo",
    "no_road_engine",
    "no_road_vehicle_for_cargo",
    "source_not_road_compatible",
    "destination_not_road_compatible",
}


@dataclass(frozen=True)
class BenchmarkTask:
    id: str
    mode: str
    description: str
    seed: int
    economy: str
    steps: int
    success: dict[str, Any]
    objectives: list[dict[str, Any]]
    prompt: str = ""
    split: str = "dev"
    difficulty: str = "medium"
    tags: tuple[str, ...] = ()


def default_benchmark_path() -> Path:
    return Path(__file__).resolve().parents[3] / "scenarios" / "firs_benchmarks.json"


def load_benchmark_tasks(path: str | Path | None = None) -> list[BenchmarkTask]:
    source = Path(path) if path else default_benchmark_path()
    data = json.loads(source.read_text(encoding="utf-8"))
    return [
        BenchmarkTask(
            id=str(item["id"]),
            mode=str(item.get("mode", "lab")),
            description=str(item.get("description", "")),
            seed=int(item.get("seed", 1)),
            economy=str(item.get("economy", "basic_temperate")),
            steps=int(item.get("steps", 32)),
            success=dict(item.get("success", {})),
            objectives=list(item.get("objectives", [])),
            prompt=str(item.get("prompt", "")),
            split=str(item.get("split", "dev")),
            difficulty=str(item.get("difficulty", "medium")),
            tags=tuple(str(tag) for tag in item.get("tags", [])),
        )
        for item in data.get("tasks", [])
    ]


def select_task(task_id: str | None, path: str | Path | None = None) -> BenchmarkTask | None:
    if not task_id:
        return None
    for task in load_benchmark_tasks(path):
        if task.id == task_id:
            return task
    raise ValueError(f"Unknown benchmark task: {task_id}")


def task_to_workbook_meta(task: BenchmarkTask, workbook_meta: dict[str, Any]) -> dict[str, Any]:
    fields = dict(workbook_meta.get("fields", {}))
    fields.update({"seed": task.seed, "economy": task.economy, "benchmark_task": task.id})
    return {
        **workbook_meta,
        "fields": fields,
        "objectives": list(task.objectives),
        "benchmark_task": {
            "id": task.id,
            "mode": task.mode,
            "description": task.description,
            "split": task.split,
            "difficulty": task.difficulty,
            "tags": list(task.tags),
            "success": task.success,
            "prompt": task.prompt,
        },
    }


def aggregate_runs(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"runs": 0}
    rewards = [float(item.get("total_reward", 0) or 0) for item in summaries]
    network_values = [
        float(item.get("final_score", {}).get("network_value", 0) or item.get("total_reward", 0) or 0)
        for item in summaries
    ]
    completed = sum(1 for item in summaries if item.get("completed"))
    return {
        "runs": len(summaries),
        "completed": completed,
        "success_rate": round(completed / len(summaries), 3),
        "median_reward": round(median(rewards), 3),
        "median_network_value": round(median(network_values), 3),
        "models": sorted({str(item.get("model", "")) for item in summaries}),
        "tasks": sorted({str(item.get("benchmark_task", "")) for item in summaries if item.get("benchmark_task")}),
    }


def aggregate_route_builder_attempts(
    attempts: list[dict[str, Any]],
    *,
    target_success_rate: float = 0.9,
) -> dict[str, Any]:
    if not attempts:
        return {
            "attempts": 0,
            "build_successes": 0,
            "active_successes": 0,
            "operational_successes": 0,
            "feasible_attempts": 0,
            "feasible_operational_successes": 0,
            "build_success_rate": 0.0,
            "active_success_rate": 0.0,
            "operational_success_rate": 0.0,
            "feasible_operational_success_rate": 0.0,
            "target_success_rate": target_success_rate,
            "level1_pass": False,
            "feasible_level1_pass": False,
            "failure_counts": {},
            "infeasible_failure_counts": {},
        }
    build_successes = sum(1 for item in attempts if item.get("build_success"))
    active_successes = sum(1 for item in attempts if item.get("active_success"))
    operational_successes = sum(1 for item in attempts if item.get("operational_success"))
    failure_counts: dict[str, int] = {}
    infeasible_failure_counts: dict[str, int] = {}
    feasible_attempts = 0
    feasible_operational_successes = 0
    for item in attempts:
        reason = str(item.get("failure_reason") or item.get("error") or "")
        infeasible = bool(reason and reason in ROUTE_BUILDER_INFEASIBLE_REASONS)
        if infeasible:
            infeasible_failure_counts[reason] = infeasible_failure_counts.get(reason, 0) + 1
        else:
            feasible_attempts += 1
            if item.get("operational_success"):
                feasible_operational_successes += 1
        if item.get("operational_success"):
            continue
        reason = reason or "not_operational"
        failure_counts[reason] = failure_counts.get(reason, 0) + 1
    count = len(attempts)
    operational_rate = operational_successes / count
    feasible_operational_rate = feasible_operational_successes / feasible_attempts if feasible_attempts else 0.0
    return {
        "attempts": count,
        "build_successes": build_successes,
        "active_successes": active_successes,
        "operational_successes": operational_successes,
        "feasible_attempts": feasible_attempts,
        "feasible_operational_successes": feasible_operational_successes,
        "infeasible_attempts": count - feasible_attempts,
        "build_success_rate": round(build_successes / count, 3),
        "active_success_rate": round(active_successes / count, 3),
        "operational_success_rate": round(operational_rate, 3),
        "feasible_operational_success_rate": round(feasible_operational_rate, 3),
        "target_success_rate": target_success_rate,
        "level1_pass": operational_rate >= target_success_rate,
        "feasible_level1_pass": feasible_attempts > 0 and feasible_operational_rate >= target_success_rate,
        "failure_counts": dict(sorted(failure_counts.items())),
        "infeasible_failure_counts": dict(sorted(infeasible_failure_counts.items())),
    }
