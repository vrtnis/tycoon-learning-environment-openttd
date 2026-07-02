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
from openttd_le.replay import export_replay


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


@dataclass(frozen=True)
class RLModelEvalConfig:
    model: Path
    workbook: Path
    task_id: str = "lab_raw_to_processor"
    algorithm: str = "auto"
    seeds: tuple[int, ...] = (1,)
    output_root: Path = Path("runs_rl_eval")
    executable: str | None = None
    openttd_user_dir: Path | None = None
    max_candidates: int = 24
    max_steps: int = 8
    eval_episodes: int = 1
    deterministic_policy: bool = True


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


def run_rl_model_eval(config: RLModelEvalConfig) -> dict[str, Any]:
    config.output_root.mkdir(parents=True, exist_ok=True)
    algorithm = _resolve_eval_algorithm(config.algorithm, config.model)
    model = _load_rl_model(algorithm, config.model)

    episodes: list[dict[str, Any]] = []
    steps: list[dict[str, Any]] = []
    for seed in config.seeds:
        for episode in range(config.eval_episodes):
            episode_seed = seed + episode
            episode_summary, episode_steps = _run_model_eval_episode(
                config,
                model,
                algorithm=algorithm,
                seed=episode_seed,
                episode=episode,
            )
            episodes.append(episode_summary)
            steps.extend(episode_steps)

    payload = {
        "schema": "openttd-le-rl-model-eval-report-v1",
        "model": str(config.model),
        "algorithm": algorithm,
        "task_id": config.task_id,
        "seeds": list(config.seeds),
        "max_steps": config.max_steps,
        "eval_episodes": config.eval_episodes,
        "deterministic_policy": config.deterministic_policy,
        "episodes": episodes,
        "aggregate": _aggregate_eval_episodes(episodes),
    }
    report_path = config.output_root / "rl_model_eval_report.json"
    trace_path = config.output_root / "step_trace.csv"
    markdown_path = config.output_root / "model_eval_report.md"
    _write_eval_steps_csv(trace_path, steps)
    payload["artifacts"] = {
        "report": str(report_path),
        "step_trace": str(trace_path),
        "markdown_report": str(markdown_path),
    }
    report_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    _write_model_eval_markdown(markdown_path, payload)
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


def _resolve_eval_algorithm(algorithm: str, model_path: Path) -> str:
    if algorithm != "auto":
        if algorithm in {"maskable_ppo", "ppo_masked", "dqn"}:
            return "maskable_ppo" if algorithm == "ppo_masked" else algorithm
        raise ValueError("Unknown RL eval algorithm. Use auto, dqn, or maskable_ppo.")
    lower_name = model_path.name.lower()
    if "maskable" in lower_name or "ppo" in lower_name:
        return "maskable_ppo"
    if "dqn" in lower_name:
        return "dqn"
    raise ValueError("Could not infer RL algorithm from model path. Pass --algorithm dqn or --algorithm maskable_ppo.")


def _load_rl_model(algorithm: str, model_path: Path) -> Any:
    if algorithm == "dqn":
        try:
            from stable_baselines3 import DQN
        except ImportError as exc:
            raise RuntimeError("DQN evaluation requires: python -m pip install stable-baselines3") from exc
        return DQN.load(str(model_path))
    if algorithm == "maskable_ppo":
        try:
            from sb3_contrib import MaskablePPO
        except ImportError as exc:
            raise RuntimeError("Maskable PPO evaluation requires: python -m pip install sb3-contrib stable-baselines3") from exc
        return MaskablePPO.load(str(model_path))
    raise ValueError(f"Unknown RL eval algorithm: {algorithm}")


def _run_model_eval_episode(
    config: RLModelEvalConfig,
    model: Any,
    *,
    algorithm: str,
    seed: int,
    episode: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    env = OpenTTDFIRSGymEnv(
        workbook=str(config.workbook),
        task_id=config.task_id,
        executable=config.executable,
        openttd_user_dir=config.openttd_user_dir,
        output_root=config.output_root / "episodes" / f"seed_{seed}_episode_{episode}",
        max_candidates=config.max_candidates,
        max_steps=config.max_steps,
        deterministic=True,
    )
    episode_steps: list[dict[str, Any]] = []
    total_reward = 0.0
    terminated = False
    truncated = False
    try:
        observation, info = env.reset(seed=seed)
        raw_run_dir = info.get("run_dir") or getattr(env.env, "run_dir", "")
        run_dir = Path(raw_run_dir) if raw_run_dir else None
        if run_dir is None:
            raise RuntimeError("RL model eval did not expose a run directory.")
        trace_path = run_dir / "firs_trace.jsonl"
        observations_path = run_dir / "observations.jsonl"
        rewards_path = run_dir / "rewards.jsonl"
        actions_path = run_dir / "actions.jsonl"
        summary_path = run_dir / "summary.json"
        launch_path = run_dir / "launch.json"
        replay_path = run_dir / "replay.json"

        with (
            trace_path.open("a", encoding="utf-8") as trace,
            observations_path.open("a", encoding="utf-8") as observations_file,
            rewards_path.open("a", encoding="utf-8") as rewards_file,
            actions_path.open("a", encoding="utf-8") as actions_file,
        ):
            _write_jsonl_event(trace, "initial_observation", 0, info.get("native_observation", {}))
            _write_jsonl(observations_file, {"step": 0, "observation": info.get("native_observation", {})})
            for step in range(config.max_steps):
                action_mask = env.action_masks()
                if algorithm == "maskable_ppo":
                    action, _ = model.predict(
                        observation,
                        deterministic=config.deterministic_policy,
                        action_masks=action_mask,
                    )
                else:
                    action, _ = model.predict(observation, deterministic=config.deterministic_policy)
                observation, reward, terminated, truncated, info = env.step(int(action))
                total_reward += float(reward)
                selected = info.get("selected_action") or {}
                result = info.get("result") or {}
                focus = _focus_from_action_result(selected, result)
                native_observation = info.get("native_observation") or {}
                _write_jsonl(
                    actions_file,
                    {
                        "step": step + 1,
                        "action_index": int(action),
                        "action": selected,
                        "result": result,
                        "observation": native_observation,
                    },
                )
                _write_jsonl_event(trace, "action", step + 1, selected)
                _write_jsonl_event(trace, "result", step + 1, result)
                _write_jsonl_event(trace, "observation", step + 1, native_observation)
                if focus:
                    _write_jsonl_event(trace, "focus", step + 1, focus)
                reward_details = info.get("reward_details") or {}
                _write_jsonl(
                    rewards_file,
                    {
                        "step": step + 1,
                        "reward": reward,
                        **reward_details,
                        "snapshot": info.get("snapshot"),
                    },
                )
                _write_jsonl(observations_file, {"step": step + 1, "observation": native_observation})
                routes = native_observation.get("routes", []) or []
                delivered_routes = sum(1 for route in routes if float(route.get("delivered", 0) or 0) > 0)
                cargo_delivered = sum(float(route.get("delivered", 0) or 0) for route in routes)
                episode_steps.append(
                    {
                        "seed": seed,
                        "episode": episode,
                        "step": step + 1,
                        "action": int(action),
                        "action_type": selected.get("type"),
                        "valid_actions": int(sum(int(value) for value in action_mask)),
                        "reward": round(float(reward), 3),
                        "total_reward": round(total_reward, 3),
                        "terminated": bool(terminated),
                        "truncated": bool(truncated),
                        "invalid_action": bool(info.get("invalid_action", False)),
                        "route_count": len(routes),
                        "delivered_routes": delivered_routes,
                        "cargo_delivered": round(cargo_delivered, 3),
                        "run_dir": str(run_dir),
                        "replay": str(replay_path),
                    }
                )
                if terminated or truncated:
                    break

        summary = env.env.summary(agent=f"rl:{algorithm}", model=str(config.model))
        summary.update(
            {
                "gym_env_id": "OpenTTD-FIRS-Deterministic-v0",
                "gym_agent": f"rl:{algorithm}",
                "seed": seed,
                "episode": episode,
                "total_reward": round(total_reward, 3),
                "trace": str(trace_path),
                "observations": str(observations_path),
                "rewards": str(rewards_path),
                "actions": str(actions_path),
                "summary": str(summary_path),
                "replay": str(replay_path),
            }
        )
        summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
        launch_info = env.env.launch_info(summary_path=summary_path)
        launch_info.update(
            {
                "agent": f"rl:{algorithm}",
                "model": str(config.model),
                "gym_env_id": "OpenTTD-FIRS-Deterministic-v0",
                "trace": str(trace_path),
                "observations": str(observations_path),
                "rewards": str(rewards_path),
                "actions": str(actions_path),
                "replay": str(replay_path),
            }
        )
        launch_path.write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
        export_replay(run_dir, replay_path)

        summary = {
            "seed": seed,
            "episode": episode,
            "steps": len(episode_steps),
            "total_reward": round(total_reward, 3),
            "terminated": bool(terminated),
            "truncated": bool(truncated),
            "success": bool(terminated),
            "run_dir": str(run_dir),
            "replay": str(replay_path),
        }
        if episode_steps:
            summary.update(
                {
                    "final_route_count": episode_steps[-1]["route_count"],
                    "final_delivered_routes": episode_steps[-1]["delivered_routes"],
                    "final_cargo_delivered": episode_steps[-1]["cargo_delivered"],
                }
            )
        return summary, episode_steps
    finally:
        env.close()


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


def _aggregate_eval_episodes(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    rewards = [float(episode.get("total_reward", 0) or 0) for episode in episodes]
    successes = [bool(episode.get("success")) for episode in episodes]
    return {
        "episodes": len(episodes),
        "success_rate": round(sum(1 for item in successes if item) / len(successes), 3) if successes else 0.0,
        "mean_total_reward": round(mean(rewards), 3) if rewards else 0.0,
        "best_total_reward": round(max(rewards), 3) if rewards else 0.0,
        "mean_steps": round(mean(float(episode.get("steps", 0) or 0) for episode in episodes), 3) if episodes else 0.0,
    }


def _write_curve_csv(path: Path, curve_points: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["timesteps", "mean_reward", "success_rate"])
        writer.writeheader()
        writer.writerows(curve_points)


def _write_eval_steps_csv(path: Path, steps: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "seed",
        "episode",
        "step",
        "action",
        "action_type",
        "valid_actions",
        "reward",
        "total_reward",
        "terminated",
        "truncated",
        "invalid_action",
        "route_count",
        "delivered_routes",
        "cargo_delivered",
        "run_dir",
        "replay",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(steps)


def _write_model_eval_markdown(path: Path, report: dict[str, Any]) -> None:
    aggregate = report.get("aggregate", {})
    lines = [
        "# TycoonLE OpenTTD RL Model Eval",
        "",
        f"- Model: `{report.get('model')}`",
        f"- Algorithm: `{report.get('algorithm')}`",
        f"- Task: `{report.get('task_id')}`",
        f"- Seeds: {', '.join(str(seed) for seed in report.get('seeds', []))}",
        f"- Max steps: {report.get('max_steps')}",
        f"- Episodes: {aggregate.get('episodes', 0)}",
        f"- Mean reward: {aggregate.get('mean_total_reward', 0)}",
        f"- Success rate: {aggregate.get('success_rate', 0)}",
        "",
        "| Seed | Episode | Steps | Reward | Success | Routes | Delivered Routes | Cargo Delivered | Run Dir |",
        "| ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for episode in report.get("episodes", []):
        lines.append(
            "| "
            f"{episode.get('seed')} | "
            f"{episode.get('episode')} | "
            f"{episode.get('steps')} | "
            f"{episode.get('total_reward')} | "
            f"{episode.get('success')} | "
            f"{episode.get('final_route_count', 0)} | "
            f"{episode.get('final_delivered_routes', 0)} | "
            f"{episode.get('final_cargo_delivered', 0)} | "
            f"`{episode.get('run_dir', '')}` |"
        )
    lines.extend(
        [
            "",
            f"Step trace: `{report.get('artifacts', {}).get('step_trace')}`",
            f"JSON report: `{report.get('artifacts', {}).get('report')}`",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _focus_from_action_result(action: dict[str, Any], result: dict[str, Any]) -> dict[str, Any]:
    focus: dict[str, Any] = {}
    if "type" in action:
        focus["action_type"] = action.get("type")
    if "type" in result:
        focus["result_type"] = result.get("type")
    for key in (
        "source_id",
        "destination_id",
        "cargo_id",
        "route_id",
        "source_station",
        "destination_station",
        "vehicle_id",
    ):
        if key in action:
            focus[key] = action.get(key)
        if key in result:
            focus[key] = result.get(key)
    return focus


def _write_jsonl_event(handle: Any, event: str, step: int, data: dict[str, Any]) -> None:
    _write_jsonl(handle, {"event": event, "step": step, "data": data})


def _write_jsonl(handle: Any, data: dict[str, Any]) -> None:
    handle.write(json.dumps(data, separators=(",", ":")) + "\n")
    handle.flush()
