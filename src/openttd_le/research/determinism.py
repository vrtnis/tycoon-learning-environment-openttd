from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv
from openttd_le.research.gym_baselines import GYM_BASELINE_AGENTS, select_baseline_action
from openttd_le.research.reproducibility import first_diff, normalize_gym_info, normalize_value


DETERMINISM_REPORT_SCHEMA = "openttd-le-determinism-report-v1"


@dataclass(frozen=True)
class DeterminismConfig:
    workbook: Path
    task_id: str = "lab_raw_to_processor"
    agent: str = "first_valid"
    seed: int = 1
    repeats: int = 3
    output_root: Path = Path("runs_determinism")
    executable: str | None = None
    openttd_user_dir: Path | None = None
    max_candidates: int = 24
    max_steps: int = 3


def run_determinism_check(config: DeterminismConfig) -> dict[str, Any]:
    if config.agent not in GYM_BASELINE_AGENTS:
        raise ValueError(f"Unknown baseline '{config.agent}'. Choices: {', '.join(GYM_BASELINE_AGENTS)}")
    if config.repeats < 2:
        raise ValueError("--repeats must be at least 2")

    config.output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for repeat in range(config.repeats):
        runs.append(_run_once(config, repeat=repeat + 1))

    baseline = runs[0]["normalized_trace"]
    comparisons = []
    ok = True
    for run in runs[1:]:
        diff = first_diff(baseline, run["normalized_trace"])
        same = diff is None
        ok = ok and same
        comparisons.append({"repeat": run["repeat"], "same": same, "diff": diff})

    payload = {
        "schema": DETERMINISM_REPORT_SCHEMA,
        "ok": ok,
        "task_id": config.task_id,
        "agent": config.agent,
        "seed": config.seed,
        "repeats": config.repeats,
        "max_steps": config.max_steps,
        "runs": [
            {
                "repeat": run["repeat"],
                "run_dir": run["run_dir"],
                "steps": run["steps"],
                "total_reward": run["total_reward"],
                "runtime": run.get("runtime", {}),
            }
            for run in runs
        ],
        "comparisons": comparisons,
    }
    report_path = config.output_root / "determinism_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report"] = str(report_path)
    return payload


def _run_once(config: DeterminismConfig, *, repeat: int) -> dict[str, Any]:
    rng = random.Random(config.seed)
    env = OpenTTDFIRSGymEnv(
        workbook=str(config.workbook),
        task_id=config.task_id,
        executable=config.executable,
        openttd_user_dir=config.openttd_user_dir,
        output_root=config.output_root,
        max_candidates=config.max_candidates,
        max_steps=config.max_steps,
        deterministic=True,
    )
    trace: list[dict[str, Any]] = []
    total_reward = 0.0
    try:
        observation, info = env.reset(seed=config.seed)
        trace.append(
            {
                "event": "reset",
                "observation": normalize_value(observation),
                "info": normalize_gym_info(info),
            }
        )
        steps = 0
        for step in range(1, config.max_steps + 1):
            action_index = select_baseline_action(config.agent, observation, info, rng)
            observation, reward, terminated, truncated, info = env.step(action_index)
            total_reward += float(reward)
            steps = step
            trace.append(
                {
                    "event": "step",
                    "step": step,
                    "action_index": int(action_index),
                    "observation": normalize_value(observation),
                    "reward": round(float(reward), 6),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": normalize_gym_info(info),
                }
            )
            if terminated or truncated:
                break
        run_dir = str(env.env.run_dir) if getattr(env.env, "run_dir", None) else None
        runtime = getattr(env.env, "runtime_lock", None) or {}
        return {
            "repeat": repeat,
            "run_dir": run_dir,
            "steps": steps,
            "total_reward": round(total_reward, 6),
            "runtime": runtime,
            "normalized_trace": trace,
        }
    finally:
        env.close()
