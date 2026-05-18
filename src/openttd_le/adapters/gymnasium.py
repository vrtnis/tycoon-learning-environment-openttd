from __future__ import annotations

from typing import Any

from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.scenarios import load_registry

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
except ImportError:
    gym = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]


class OpenTTDLEGymEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gymnasium adapter for choosing an index into the current action frontier.

    The native OpenTTD-LE API remains the source of truth. This adapter keeps
    standard RL loops simple by exposing `Discrete(max_candidates)` actions and
    placing the rich candidate frontier in `info["candidate_actions"]`.
    """

    metadata = {"render_modes": ["ansi"], "render_fps": 1}

    def __init__(
        self,
        task_id: str = "coal_easy_001",
        *,
        backend: str = "toy",
        max_candidates: int = 24,
        render_mode: str | None = None,
    ) -> None:
        if gym is None or spaces is None or np is None:
            raise RuntimeError("OpenTTDLEGymEnv requires the optional 'gymnasium' extra: pip install -e .[gymnasium]")

        self.task_id = task_id
        self.backend = backend
        self.max_candidates = max_candidates
        self.render_mode = render_mode
        self.env = OpenTTDLEnv(backend=backend, registry=load_registry())
        self.action_space = spaces.Discrete(max_candidates)
        self.observation_space = spaces.Dict(
            {
                "score": spaces.Box(low=0.0, high=100.0, shape=(), dtype=np.float32),
                "cash": spaces.Box(low=-1_000_000_000.0, high=1_000_000_000.0, shape=(), dtype=np.float32),
                "loan": spaces.Box(low=0.0, high=1_000_000_000.0, shape=(), dtype=np.float32),
                "month": spaces.Box(low=0.0, high=10_000.0, shape=(), dtype=np.float32),
                "route_count": spaces.Box(low=0.0, high=10_000.0, shape=(), dtype=np.float32),
                "vehicles": spaces.Box(low=0.0, high=100_000.0, shape=(), dtype=np.float32),
                "cargo_delivered": spaces.Box(low=0.0, high=1_000_000_000.0, shape=(), dtype=np.float32),
                "candidate_rank_scores": spaces.Box(
                    low=-1_000_000.0,
                    high=1_000_000.0,
                    shape=(max_candidates,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(max_candidates),
            }
        )
        self._last_observation: dict[str, Any] | None = None

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        task_id = str((options or {}).get("task_id") or self.task_id)
        observation, info = self.env.reset(task_id, seed=seed)
        self._last_observation = observation
        candidates = self.env.candidate_actions(limit=self.max_candidates)
        return self._encode_observation(observation, candidates), {**info, "candidate_actions": candidates, "native_observation": observation}

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise RuntimeError("Environment must be reset before step().")
        candidates = self.env.candidate_actions(limit=self.max_candidates)
        action_index = int(action)
        if action_index < 0 or action_index >= len(candidates):
            selected = {"type": "wait", "months": 1}
            preview = {"feasible": False, "diagnostics": ["candidate_index_out_of_range"]}
        else:
            selected = dict(candidates[action_index]["action"])
            preview = self.env.preview(selected)
        result = self.env.step(selected)
        self._last_observation = result.observation
        next_candidates = self.env.candidate_actions(limit=self.max_candidates)
        info = {
            **result.info,
            "selected_action": selected,
            "selected_preview": preview,
            "candidate_actions": next_candidates,
            "native_observation": result.observation,
        }
        return (
            self._encode_observation(result.observation, next_candidates),
            float(result.reward),
            bool(result.terminated),
            bool(result.truncated),
            info,
        )

    def render(self) -> str | None:
        if self._last_observation is None:
            return None
        metrics = self._last_observation["metrics"]
        line = (
            f"score={metrics['score']} cargo={metrics['cargo_delivered']} "
            f"profit={metrics['operating_profit']} event={self._last_observation['last_event']}"
        )
        if self.render_mode == "ansi":
            return line
        print(line)
        return None

    def close(self) -> None:
        self.env.close()

    def _encode_observation(self, observation: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if np is None:
            raise RuntimeError("OpenTTDLEGymEnv requires numpy through the optional gymnasium extra.")

        scores = np.zeros((self.max_candidates,), dtype=np.float32)
        mask = np.zeros((self.max_candidates,), dtype="int8")
        for index, candidate in enumerate(candidates[: self.max_candidates]):
            scores[index] = float(candidate.get("rank_score", 0) or 0)
            mask[index] = 1 if candidate.get("feasible") else 0
        metrics = observation["metrics"]
        company = observation["company"]
        return {
            "score": np.array(metrics["score"], dtype=np.float32),
            "cash": np.array(company["cash"], dtype=np.float32),
            "loan": np.array(company["loan"], dtype=np.float32),
            "month": np.array(observation["time"]["month"], dtype=np.float32),
            "route_count": np.array(metrics["route_count"], dtype=np.float32),
            "vehicles": np.array(metrics["vehicles"], dtype=np.float32),
            "cargo_delivered": np.array(metrics["cargo_delivered"], dtype=np.float32),
            "candidate_rank_scores": scores,
            "action_mask": mask,
        }


def make(task_id: str = "coal_easy_001", **kwargs: Any) -> OpenTTDLEGymEnv:
    return OpenTTDLEGymEnv(task_id=task_id, **kwargs)
