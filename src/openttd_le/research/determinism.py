from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv
from openttd_le.research.gym_baselines import GYM_BASELINE_AGENTS, select_baseline_action
from openttd_le.research.reproducibility import (
    determinism_contract,
    first_diff,
    normalize_determinism_trace,
    normalize_runtime_lock,
    stable_json_sha256,
)


DETERMINISM_REPORT_SCHEMA = "openttd-le-determinism-report-v2"


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
    trace_mode: str = "strict"
    fixed_action_script: bool = True
    compare_runtime_lock: bool = True
    progress_path: Path | None = None


def run_determinism_check(config: DeterminismConfig) -> dict[str, Any]:
    if config.agent not in GYM_BASELINE_AGENTS:
        raise ValueError(f"Unknown baseline '{config.agent}'. Choices: {', '.join(GYM_BASELINE_AGENTS)}")
    if config.repeats < 2:
        raise ValueError("--repeats must be at least 2")
    if config.trace_mode not in {"strict", "semantic"}:
        raise ValueError("trace_mode must be 'strict' or 'semantic'")

    config.output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    action_script: list[int] | None = None
    for repeat in range(config.repeats):
        run = _run_once(config, repeat=repeat + 1, action_script=action_script)
        runs.append(run)
        if config.fixed_action_script and action_script is None:
            action_script = list(run["action_sequence"])
        _append_progress(
            config.progress_path,
            {
                "event": "repeat_complete",
                "task_id": config.task_id,
                "agent": config.agent,
                "seed": config.seed,
                "repeat": repeat + 1,
                "repeats": config.repeats,
                "steps": run["steps"],
                "total_reward": run["total_reward"],
                "trace_sha256": run["trace_sha256"],
                "runtime_lock_sha256": run["runtime_lock_sha256"],
                "run_dir": run["run_dir"],
            },
        )

    baseline = runs[0]["normalized_trace"]
    baseline_lock = runs[0]["runtime_comparable"]
    comparisons = []
    ok = True
    for run in runs[1:]:
        trace_diff = first_diff(baseline, run["normalized_trace"])
        runtime_diff = (
            first_diff(baseline_lock, run["runtime_comparable"])
            if config.compare_runtime_lock
            else None
        )
        same = trace_diff is None and runtime_diff is None
        ok = ok and same
        comparisons.append(
            {
                "repeat": run["repeat"],
                "same": same,
                "trace_diff": trace_diff,
                "runtime_lock_diff": runtime_diff,
            }
        )

    payload = {
        "schema": DETERMINISM_REPORT_SCHEMA,
        "contract": determinism_contract(),
        "ok": ok,
        "task_id": config.task_id,
        "agent": config.agent,
        "seed": config.seed,
        "repeats": config.repeats,
        "max_steps": config.max_steps,
        "max_candidates": config.max_candidates,
        "trace_mode": config.trace_mode,
        "fixed_action_script": config.fixed_action_script,
        "compare_runtime_lock": config.compare_runtime_lock,
        "action_sequence": action_script if action_script is not None else runs[0]["action_sequence"],
        "runs": [
            {
                "repeat": run["repeat"],
                "run_dir": run["run_dir"],
                "steps": run["steps"],
                "total_reward": run["total_reward"],
                "action_sequence": run["action_sequence"],
                "trace_sha256": run["trace_sha256"],
                "runtime_lock_sha256": run["runtime_lock_sha256"],
                "raw_trace": run["raw_trace_path"],
                "normalized_trace": run["normalized_trace_path"],
                "runtime_lock": run["runtime_lock_path"],
                "runtime": run.get("runtime", {}),
                "runtime_comparable": run["runtime_comparable"],
            }
            for run in runs
        ],
        "comparisons": comparisons,
    }
    report_path = config.output_root / "determinism_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report"] = str(report_path)
    _append_progress(
        config.progress_path,
        {
            "event": "combination_complete",
            "task_id": config.task_id,
            "agent": config.agent,
            "seed": config.seed,
            "ok": ok,
            "report": str(report_path),
            "comparisons": comparisons,
        },
    )
    return payload


def _run_once(
    config: DeterminismConfig,
    *,
    repeat: int,
    action_script: list[int] | None,
) -> dict[str, Any]:
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
    action_sequence: list[int] = []
    total_reward = 0.0
    try:
        if hasattr(env.action_space, "seed"):
            env.action_space.seed(config.seed)
        observation, info = env.reset(seed=config.seed)
        trace.append(
            {
                "event": "reset",
                "seed": config.seed,
                "observation": observation,
                "info": info,
            }
        )
        steps = 0
        for step in range(1, config.max_steps + 1):
            if action_script is not None:
                if step > len(action_script):
                    break
                action_index = int(action_script[step - 1])
            else:
                action_index = select_baseline_action(config.agent, observation, info, rng)
            action_sequence.append(int(action_index))
            observation, reward, terminated, truncated, info = env.step(action_index)
            total_reward += float(reward)
            steps = step
            trace.append(
                {
                    "event": "step",
                    "step": step,
                    "action_index": int(action_index),
                    "observation": observation,
                    "reward": round(float(reward), 6),
                    "terminated": bool(terminated),
                    "truncated": bool(truncated),
                    "info": info,
                }
            )
            if terminated or truncated:
                break
        run_dir = str(env.env.run_dir) if getattr(env.env, "run_dir", None) else None
        runtime = getattr(env.env, "runtime_lock", None) or {}
        artifact_dir = Path(run_dir) if run_dir else config.output_root / f"repeat_{repeat:02d}"
        artifact_dir.mkdir(parents=True, exist_ok=True)
        normalized_trace = normalize_determinism_trace(trace, mode=config.trace_mode)
        runtime_comparable = normalize_runtime_lock(runtime)
        raw_trace_path = artifact_dir / "determinism_raw_trace.json"
        normalized_trace_path = artifact_dir / "determinism_normalized_trace.json"
        runtime_lock_path = artifact_dir / "determinism_runtime_lock.json"
        _write_json(raw_trace_path, trace)
        _write_json(normalized_trace_path, normalized_trace)
        _write_json(runtime_lock_path, runtime)
        return {
            "repeat": repeat,
            "run_dir": run_dir,
            "steps": steps,
            "total_reward": round(total_reward, 6),
            "action_sequence": action_sequence,
            "runtime": runtime,
            "runtime_comparable": runtime_comparable,
            "trace_sha256": stable_json_sha256(normalized_trace),
            "runtime_lock_sha256": stable_json_sha256(runtime_comparable),
            "raw_trace_path": str(raw_trace_path),
            "normalized_trace_path": str(normalized_trace_path),
            "runtime_lock_path": str(runtime_lock_path),
            "normalized_trace": normalized_trace,
        }
    finally:
        env.close()


def _write_json(path: Path, payload: Any) -> None:
    path.write_text(json.dumps(payload, indent=2, default=_json_default), encoding="utf-8")


def _json_default(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)


def _append_progress(path: Path | None, payload: dict[str, Any]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":"), default=_json_default) + "\n")
