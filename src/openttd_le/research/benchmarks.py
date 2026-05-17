from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import median
from typing import Any


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
