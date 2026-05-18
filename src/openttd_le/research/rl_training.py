from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from openttd_le.adapters.gymnasium import OpenTTDFIRSGymEnv
from openttd_le.research.gym_baselines import GYM_BASELINE_AGENTS, select_baseline_action
from openttd_le.research.reporting import write_benchmark_report


RL_TRAINING_SCHEMA = "openttd-le-rl-training-report-v1"
SCRIPTED_PREFIX = "scripted:"


@dataclass(frozen=True)
class RLTrainingConfig:
    workbook: Path
    task_id: str = "lab_raw_to_processor"
    algorithms: tuple[str, ...] = ("scripted:masked_random", "scripted:first_valid")
    seeds: tuple[int, ...] = (1,)
    output_root: Path = Path("runs_rl")
    executable: str | None = None
    openttd_user_dir: Path | None = None
    max_candidates: int = 24
    max_steps: int = 8
    total_timesteps: int = 64
    eval_interval: int = 32
    eval_episodes: int = 1


def run_rl_training(config: RLTrainingConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    runs: list[dict[str, Any]] = []
    for algorithm in config.algorithms:
        for seed in config.seeds:
            runs.append(_run_algorithm(config, algorithm=algorithm, seed=seed))

    payload = {
        "schema": RL_TRAINING_SCHEMA,
        "task_id": config.task_id,
        "algorithms": list(config.algorithms),
        "seeds": list(config.seeds),
        "max_steps": config.max_steps,
        "total_timesteps": config.total_timesteps,
        "eval_interval": config.eval_interval,
        "eval_episodes": config.eval_episodes,
        "runs": runs,
        "aggregate": _aggregate_training_runs(runs),
    }
    report_path = config.output_root / "rl_training_report.json"
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    payload["report"] = str(report_path)
    artifacts = write_benchmark_report(training_report=payload, output_dir=config.output_root)
    payload["artifacts"] = {"benchmark_report": artifacts["report"], **artifacts}
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _run_algorithm(config: RLTrainingConfig, *, algorithm: str, seed: int) -> dict[str, Any]:
    algorithm_slug = algorithm.replace(":", "_")
    run_dir = config.output_root / algorithm_slug / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    if algorithm.startswith(SCRIPTED_PREFIX) or algorithm in GYM_BASELINE_AGENTS:
        agent = algorithm.removeprefix(SCRIPTED_PREFIX)
        return _run_scripted(config, agent_name=agent, algorithm=algorithm, seed=seed, run_dir=run_dir)
    if algorithm == "dqn":
        return _run_dqn(config, seed=seed, run_dir=run_dir)
    if algorithm in {"maskable_ppo", "ppo_masked"}:
        return _run_maskable_ppo(config, seed=seed, run_dir=run_dir)
    raise ValueError(
        "Unknown RL algorithm "
        f"'{algorithm}'. Use scripted:<{','.join(GYM_BASELINE_AGENTS)}>, dqn, or maskable_ppo."
    )


def _run_scripted(
    config: RLTrainingConfig,
    *,
    agent_name: str,
    algorithm: str,
    seed: int,
    run_dir: Path,
) -> dict[str, Any]:
    if agent_name not in GYM_BASELINE_AGENTS:
        raise ValueError(f"Unknown scripted baseline '{agent_name}'. Choices: {', '.join(GYM_BASELINE_AGENTS)}")
    rewards = []
    successes = []
    for episode in range(config.eval_episodes):
        reward, success = _evaluate_scripted_episode(config, agent_name=agent_name, seed=seed + episode)
        rewards.append(reward)
        successes.append(success)
    curve_points = [
        {
            "timesteps": 0,
            "mean_reward": round(mean(rewards), 3) if rewards else 0.0,
            "success_rate": round(sum(1 for item in successes if item) / len(successes), 3) if successes else 0.0,
        }
    ]
    _write_curve_csv(run_dir / "learning_curve.csv", curve_points)
    return {
        "algorithm": algorithm,
        "seed": seed,
        "timesteps": 0,
        "curve_points": curve_points,
        "best_mean_reward": curve_points[-1]["mean_reward"],
        "final_mean_reward": curve_points[-1]["mean_reward"],
        "learning_curve": str(run_dir / "learning_curve.csv"),
        "run_dir": str(run_dir),
        "note": "Scripted baseline evaluation; no learning update is performed.",
    }


def _run_dqn(config: RLTrainingConfig, *, seed: int, run_dir: Path) -> dict[str, Any]:
    try:
        from stable_baselines3 import DQN
    except ImportError as exc:
        raise RuntimeError("DQN training requires: python -m pip install stable-baselines3") from exc

    env = _make_env(config, seed=seed, output_root=run_dir / "train")
    model_path = run_dir / "dqn_model.zip"
    curve_points: list[dict[str, Any]] = []
    try:
        model = DQN(
            "MultiInputPolicy",
            env,
            seed=seed,
            verbose=0,
            learning_starts=0,
            buffer_size=max(128, config.total_timesteps),
            train_freq=1,
            gradient_steps=1,
        )
        trained = 0
        for checkpoint in _checkpoints(config.total_timesteps, config.eval_interval):
            chunk = checkpoint - trained
            if chunk > 0:
                model.learn(total_timesteps=chunk, reset_num_timesteps=(trained == 0))
                trained = checkpoint
            curve_points.append(_evaluate_model(config, model, seed=seed + checkpoint, timesteps=checkpoint))
        model.save(str(model_path))
    finally:
        env.close()
    _write_curve_csv(run_dir / "learning_curve.csv", curve_points)
    return _training_run_summary("dqn", seed, config.total_timesteps, run_dir, model_path, curve_points)


def _run_maskable_ppo(config: RLTrainingConfig, *, seed: int, run_dir: Path) -> dict[str, Any]:
    try:
        from sb3_contrib import MaskablePPO
    except ImportError as exc:
        raise RuntimeError("Maskable PPO training requires: python -m pip install sb3-contrib stable-baselines3") from exc

    env = _make_env(config, seed=seed, output_root=run_dir / "train")
    model_path = run_dir / "maskable_ppo_model.zip"
    curve_points: list[dict[str, Any]] = []
    try:
        model = MaskablePPO(
            "MultiInputPolicy",
            env,
            seed=seed,
            verbose=0,
            n_steps=max(2, config.max_steps),
            batch_size=max(2, config.max_steps),
        )
        trained = 0
        for checkpoint in _checkpoints(config.total_timesteps, config.eval_interval):
            chunk = checkpoint - trained
            if chunk > 0:
                model.learn(total_timesteps=chunk, reset_num_timesteps=(trained == 0))
                trained = checkpoint
            curve_points.append(_evaluate_model(config, model, seed=seed + checkpoint, timesteps=checkpoint, masked=True))
        model.save(str(model_path))
    finally:
        env.close()
    _write_curve_csv(run_dir / "learning_curve.csv", curve_points)
    return _training_run_summary("maskable_ppo", seed, config.total_timesteps, run_dir, model_path, curve_points)


def _evaluate_model(
    config: RLTrainingConfig,
    model: Any,
    *,
    seed: int,
    timesteps: int,
    masked: bool = False,
) -> dict[str, Any]:
    rewards = []
    successes = []
    for episode in range(config.eval_episodes):
        env = _make_env(config, seed=seed + episode, output_root=config.output_root / "eval")
        total_reward = 0.0
        try:
            observation, info = env.reset(seed=seed + episode)
            for _ in range(config.max_steps):
                if masked:
                    action, _ = model.predict(observation, deterministic=True, action_masks=env.action_masks())
                else:
                    action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, info = env.step(int(action))
                total_reward += float(reward)
                if terminated or truncated:
                    break
            rewards.append(total_reward)
            successes.append(bool(terminated))
        finally:
            env.close()
    return {
        "timesteps": timesteps,
        "mean_reward": round(mean(rewards), 3) if rewards else 0.0,
        "success_rate": round(sum(1 for item in successes if item) / len(successes), 3) if successes else 0.0,
    }


def _evaluate_scripted_episode(
    config: RLTrainingConfig,
    *,
    agent_name: str,
    seed: int,
) -> tuple[float, bool]:
    rng = random.Random(seed)
    env = _make_env(config, seed=seed, output_root=config.output_root / "scripted_eval")
    total_reward = 0.0
    terminated = False
    try:
        observation, info = env.reset(seed=seed)
        for _ in range(config.max_steps):
            action = select_baseline_action(agent_name, observation, info, rng)
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += float(reward)
            if terminated or truncated:
                break
        return round(total_reward, 3), bool(terminated)
    finally:
        env.close()


def _make_env(config: RLTrainingConfig, *, seed: int, output_root: Path) -> OpenTTDFIRSGymEnv:
    output_root.mkdir(parents=True, exist_ok=True)
    return OpenTTDFIRSGymEnv(
        workbook=str(config.workbook),
        task_id=config.task_id,
        executable=config.executable,
        openttd_user_dir=config.openttd_user_dir,
        output_root=output_root,
        max_candidates=config.max_candidates,
        max_steps=config.max_steps,
        deterministic=True,
    )


def _checkpoints(total_timesteps: int, eval_interval: int) -> list[int]:
    total = max(0, int(total_timesteps))
    interval = max(1, int(eval_interval))
    if total == 0:
        return [0]
    points = list(range(interval, total + 1, interval))
    if points[-1] != total:
        points.append(total)
    return points


def _training_run_summary(
    algorithm: str,
    seed: int,
    timesteps: int,
    run_dir: Path,
    model_path: Path,
    curve_points: list[dict[str, Any]],
) -> dict[str, Any]:
    rewards = [float(point.get("mean_reward", 0) or 0) for point in curve_points]
    return {
        "algorithm": algorithm,
        "seed": seed,
        "timesteps": timesteps,
        "curve_points": curve_points,
        "best_mean_reward": round(max(rewards), 3) if rewards else 0.0,
        "final_mean_reward": round(rewards[-1], 3) if rewards else 0.0,
        "learning_curve": str(run_dir / "learning_curve.csv"),
        "model": str(model_path),
        "run_dir": str(run_dir),
    }


def _aggregate_training_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    per_algorithm: dict[str, list[dict[str, Any]]] = {}
    for run in runs:
        per_algorithm.setdefault(str(run.get("algorithm")), []).append(run)
    aggregate = {}
    for algorithm, items in per_algorithm.items():
        best = [float(item.get("best_mean_reward", 0) or 0) for item in items]
        final = [float(item.get("final_mean_reward", 0) or 0) for item in items]
        aggregate[algorithm] = {
            "runs": len(items),
            "best_mean_reward": round(mean(best), 3) if best else 0.0,
            "final_mean_reward": round(mean(final), 3) if final else 0.0,
        }
    return {"runs": len(runs), "per_algorithm": aggregate}


def _write_curve_csv(path: Path, curve_points: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timesteps", "mean_reward", "success_rate"])
        writer.writeheader()
        writer.writerows(curve_points)
