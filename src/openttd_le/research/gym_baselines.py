from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv
from openttd_le.replay import export_replay
from openttd_le.workbooks.export import export_run_to_xlsx


GYM_BASELINE_AGENTS = ("random", "masked_random", "first_valid", "highest_production", "shortest_route")


@dataclass(frozen=True)
class GymBaselineConfig:
    workbook: Path
    task_id: str = "lab_raw_to_processor"
    agents: tuple[str, ...] = ("masked_random", "first_valid", "highest_production")
    seeds: tuple[int, ...] = (1,)
    output_root: Path = Path("runs_gym_baselines")
    executable: str | None = None
    openttd_user_dir: Path | None = None
    max_candidates: int = 24
    max_steps: int = 8
    deterministic: bool = False


def run_gym_baselines(config: GymBaselineConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    summaries: list[dict[str, Any]] = []
    for agent_name in config.agents:
        if agent_name not in GYM_BASELINE_AGENTS:
            raise ValueError(f"Unknown Gym baseline '{agent_name}'. Choices: {', '.join(GYM_BASELINE_AGENTS)}")
        for seed in config.seeds:
            summaries.append(_run_one(config, agent_name=agent_name, seed=seed))
    payload = {
        "schema": "openttd-le-gym-baseline-benchmark-v1",
        "task_id": config.task_id,
        "agents": list(config.agents),
        "seeds": list(config.seeds),
        "runs": summaries,
        "aggregate": _aggregate(summaries),
    }
    (config.output_root / "gym_benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_one(config: GymBaselineConfig, *, agent_name: str, seed: int) -> dict[str, Any]:
    rng = random.Random(seed)
    env = OpenTTDFIRSGymEnv(
        workbook=str(config.workbook),
        task_id=config.task_id,
        executable=config.executable,
        openttd_user_dir=config.openttd_user_dir,
        output_root=config.output_root,
        max_candidates=config.max_candidates,
        max_steps=config.max_steps,
        deterministic=config.deterministic,
    )
    try:
        observation, info = env.reset(seed=seed)
        raw_run_dir = info.get("run_dir") or getattr(env.env, "run_dir", None)
        if raw_run_dir is None:
            raise RuntimeError("Gym baseline run did not expose a run directory.")
        run_dir = Path(raw_run_dir)
        trace_path = run_dir / "firs_trace.jsonl"
        observations_path = run_dir / "observations.jsonl"
        rewards_path = run_dir / "rewards.jsonl"
        actions_path = run_dir / "actions.jsonl"
        summary_path = run_dir / "summary.json"
        report_path = run_dir / "report.xlsx"
        replay_path = run_dir / "replay.json"
        total_reward = 0.0

        with (
            trace_path.open("a", encoding="utf-8") as trace,
            observations_path.open("a", encoding="utf-8") as observations_file,
            rewards_path.open("a", encoding="utf-8") as rewards_file,
            actions_path.open("a", encoding="utf-8") as actions_file,
        ):
            _write_jsonl_event(trace, "initial_observation", 0, info.get("native_observation", {}))
            _write_jsonl(observations_file, {"step": 0, "observation": info.get("native_observation", {})})
            for step in range(1, config.max_steps + 1):
                action_index = select_baseline_action(agent_name, observation, info, rng)
                observation, reward, terminated, truncated, info = env.step(action_index)
                total_reward += float(reward)
                for executed in info.get("actions", []):
                    _write_jsonl(actions_file, {"step": step, "action_index": action_index, **executed})
                    _write_jsonl_event(trace, "action", step, executed.get("action", {}))
                    _write_jsonl_event(trace, "result", step, executed.get("result", {}))
                    _write_jsonl_event(trace, "observation", step, executed.get("observation", info.get("native_observation", {})))
                reward_details = info.get("reward_details") or {}
                _write_jsonl(rewards_file, {"step": step, "reward": reward, **reward_details, "snapshot": info.get("snapshot")})
                _write_jsonl(observations_file, {"step": step, "observation": info.get("native_observation", {})})
                if terminated or truncated:
                    break

        summary = env.env.summary(agent=agent_name, model=None)
        summary.update(
            {
                "gym_env_id": "OpenTTD-FIRS-Deterministic-v0" if config.deterministic else "OpenTTD-FIRS-Lab-v0",
                "gym_agent": agent_name,
                "seed": seed,
                "total_reward": round(total_reward, 3),
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
        launch_info = env.env.launch_info(summary_path=summary_path)
        launch_info.update(
            {
                "agent": agent_name,
                "gym_env_id": "OpenTTD-FIRS-Deterministic-v0" if config.deterministic else "OpenTTD-FIRS-Lab-v0",
                "trace": str(trace_path),
                "observations": str(observations_path),
                "rewards": str(rewards_path),
                "actions": str(actions_path),
                "report": str(report_path),
                "replay": str(replay_path),
            }
        )
        (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
        export_run_to_xlsx(run_dir, report_path, source_workbook=config.workbook)
        export_replay(run_dir, replay_path)
        return summary
    finally:
        env.close()


def select_baseline_action(
    agent_name: str,
    observation: dict[str, Any],
    info: dict[str, Any],
    rng: random.Random,
) -> int:
    raw_mask = info.get("action_mask")
    if raw_mask is None:
        raw_mask = observation.get("action_mask")
    mask = list(raw_mask) if raw_mask is not None else []
    valid = [index for index, value in enumerate(mask) if int(value)]
    if agent_name == "random":
        return rng.randrange(len(mask)) if mask else 0
    if not valid:
        return 0
    if agent_name == "masked_random":
        return rng.choice(valid)
    if agent_name == "first_valid":
        return valid[0]

    candidates = info.get("candidate_actions") or []
    if agent_name == "highest_production":
        return max(valid, key=lambda index: _candidate_number(candidates, index, "production", default=0.0))
    if agent_name == "shortest_route":
        return min(valid, key=lambda index: _candidate_number(candidates, index, "distance", default=1_000_000.0))
    raise ValueError(f"Unknown Gym baseline: {agent_name}")


def _candidate_number(candidates: list[dict[str, Any]], index: int, field: str, *, default: float) -> float:
    if index >= len(candidates):
        return default
    route = candidates[index].get("route", {}) or {}
    try:
        return float(route.get(field, default) or default)
    except (TypeError, ValueError):
        return default


def _aggregate(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for summary in summaries:
        grouped.setdefault(str(summary.get("gym_agent") or summary.get("agent")), []).append(summary)
    per_agent = {}
    for agent, rows in grouped.items():
        completed = sum(1 for row in rows if row.get("completed"))
        rewards = [float(row.get("total_reward", 0) or 0) for row in rows]
        per_agent[agent] = {
            "runs": len(rows),
            "completed": completed,
            "success_rate": round(completed / len(rows), 3) if rows else 0.0,
            "avg_total_reward": round(sum(rewards) / len(rewards), 3) if rewards else 0.0,
        }
    return {"runs": len(summaries), "per_agent": per_agent}


def _write_jsonl_event(handle: Any, event: str, step: int, data: dict[str, Any]) -> None:
    _write_jsonl(handle, {"event": event, "step": step, "data": data})


def _write_jsonl(handle: Any, data: dict[str, Any]) -> None:
    handle.write(json.dumps(data, separators=(",", ":")) + "\n")
    handle.flush()
