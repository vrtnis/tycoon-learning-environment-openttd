from __future__ import annotations

from collections.abc import Callable
from typing import Any

from openttd_le.core.env import OpenTTDLEnv
from openttd_le.core.scenarios import load_registry
from openttd_le.envs import OpenTTDFIRSEnv
from openttd_le.research.reproducibility import (
    normalize_candidate_actions,
    normalize_gym_info,
    normalize_observation,
    normalize_result,
    normalize_value,
)

try:
    import gymnasium as gym
    import numpy as np
    from gymnasium import spaces
    from gymnasium.envs.registration import EnvSpec
except ImportError:
    gym = None  # type: ignore[assignment]
    np = None  # type: ignore[assignment]
    spaces = None  # type: ignore[assignment]
    EnvSpec = None  # type: ignore[assignment]


FIRS_GYM_ID = "OpenTTD-FIRS-Lab-v0"
FIRS_DETERMINISTIC_GYM_ID = "OpenTTD-FIRS-Deterministic-v0"
TOY_GYM_ID = "OpenTTDLE-Toy-v0"

FIRS_OBSERVATION_SPEC = {
    "tick": "float32[1], current OpenTTD game tick",
    "bank_balance": "float32[1], current company bank balance",
    "route_count": "float32[1], registered physical cargo routes",
    "delivered_routes": "float32[1], routes with at least one delivery",
    "cargo_delivered": "float32[1], total delivered cargo units across routes",
    "route_profit": "float32[1], sum of route vehicle profit",
    "candidate_production": "float32[max_candidates], production estimate for each candidate route action",
    "action_mask": "int8[max_candidates], 1 when the candidate index is valid for this state",
}


class OpenTTDLEGymEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gymnasium adapter for choosing an index into the current action frontier.

    The native TycoonLE OpenTTD API remains the source of truth. This adapter keeps
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
        if EnvSpec is not None:
            self.spec = EnvSpec(
                id=TOY_GYM_ID,
                entry_point="openttd_le.adapters.gymnasium:OpenTTDLEGymEnv",
                nondeterministic=False,
                kwargs={"task_id": task_id, "backend": backend, "max_candidates": max_candidates},
            )
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
        self._last_candidates: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        task_id = str((options or {}).get("task_id") or self.task_id)
        observation, info = self.env.reset(task_id, seed=seed)
        self._last_observation = observation
        candidates = self.env.candidate_actions(limit=self.max_candidates)
        self._last_candidates = candidates
        return self._encode_observation(observation, candidates), {
            **info,
            "candidate_actions": candidates,
            "action_mask": self.action_masks(),
            "native_observation": observation,
        }

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
        self._last_candidates = next_candidates
        info = {
            **result.info,
            "selected_action": selected,
            "selected_preview": preview,
            "candidate_actions": next_candidates,
            "action_mask": self.action_masks(),
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

    def action_masks(self) -> Any:
        if np is None:
            raise RuntimeError("OpenTTDLEGymEnv requires numpy through the optional gymnasium extra.")
        mask = np.zeros((self.max_candidates,), dtype="int8")
        for index, candidate in enumerate(self._last_candidates[: self.max_candidates]):
            mask[index] = 1 if candidate.get("feasible") else 0
        return mask

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


class OpenTTDFIRSGymEnv(gym.Env if gym is not None else object):  # type: ignore[misc]
    """Gymnasium adapter for the real OpenTTD/FIRS environment.

    The native :class:`OpenTTDFIRSEnv` remains the canonical API. This adapter
    exposes a fixed discrete action space over the current macro-action
    candidates and returns the full OpenTTD observation in `info`.
    """

    metadata = {"render_modes": ["external"], "render_fps": 15}

    def __init__(
        self,
        *,
        workbook: str = "scenario.xlsx",
        task_id: str = "lab_raw_to_processor",
        max_candidates: int = 24,
        invalid_action_penalty: float = -1.0,
        render_mode: str | None = None,
        env: Any | None = None,
        deterministic: bool = False,
        **env_kwargs: Any,
    ) -> None:
        if gym is None or spaces is None or np is None:
            raise RuntimeError("OpenTTDFIRSGymEnv requires the optional 'gymnasium' extra: pip install -e .[gymnasium]")

        self.task_id = task_id
        self.max_candidates = max_candidates
        self.invalid_action_penalty = float(invalid_action_penalty)
        self.render_mode = render_mode
        self.deterministic = bool(deterministic)
        self.env = env or OpenTTDFIRSEnv(workbook=workbook, task_id=task_id, deterministic=self.deterministic, **env_kwargs)
        if EnvSpec is not None:
            spec_kwargs = {
                "workbook": workbook,
                "task_id": task_id,
                "max_candidates": max_candidates,
                "invalid_action_penalty": invalid_action_penalty,
                "deterministic": self.deterministic,
                **env_kwargs,
            }
            self.spec = EnvSpec(
                id=FIRS_DETERMINISTIC_GYM_ID if self.deterministic else FIRS_GYM_ID,
                entry_point="openttd_le.adapters.gymnasium:OpenTTDFIRSGymEnv",
                nondeterministic=not self.deterministic,
                kwargs=spec_kwargs,
            )
        self.action_space = spaces.Discrete(max_candidates)
        self.observation_space = spaces.Dict(
            {
                "tick": spaces.Box(low=0.0, high=1_000_000_000.0, shape=(1,), dtype=np.float32),
                "bank_balance": spaces.Box(low=-1_000_000_000.0, high=1_000_000_000.0, shape=(1,), dtype=np.float32),
                "route_count": spaces.Box(low=0.0, high=10_000.0, shape=(1,), dtype=np.float32),
                "delivered_routes": spaces.Box(low=0.0, high=10_000.0, shape=(1,), dtype=np.float32),
                "cargo_delivered": spaces.Box(low=0.0, high=1_000_000_000.0, shape=(1,), dtype=np.float32),
                "route_profit": spaces.Box(low=-1_000_000_000.0, high=1_000_000_000.0, shape=(1,), dtype=np.float32),
                "candidate_production": spaces.Box(
                    low=0.0,
                    high=1_000_000_000.0,
                    shape=(max_candidates,),
                    dtype=np.float32,
                ),
                "action_mask": spaces.MultiBinary(max_candidates),
            }
        )
        self._last_observation: dict[str, Any] | None = None
        self._last_candidates: list[dict[str, Any]] = []

    def reset(self, *, seed: int | None = None, options: dict[str, Any] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        super().reset(seed=seed)
        observation, info = self.env.reset(seed=seed)
        self._last_observation = observation
        candidates = _firs_candidates(observation, info)
        self._last_candidates = candidates
        encoded = self._encode_observation(observation, candidates)
        return encoded, self._info_payload(
            {
                **info,
                "candidate_actions": candidates,
                "action_mask": self.action_masks(),
                "native_observation": observation,
            }
        )

    def step(self, action: int) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self._last_observation is None:
            raise RuntimeError("Environment must be reset before step().")

        candidates = self._last_candidates
        action_index = int(action)
        invalid_action = action_index < 0 or action_index >= min(len(candidates), self.max_candidates)
        if not invalid_action and not bool(candidates[action_index].get("feasible", True)):
            invalid_action = True
        if invalid_action:
            selected = {"type": "wait_months", "months": 1, "label": "invalid candidate index no-op"}
        else:
            selected = dict(candidates[action_index]["action"])

        observation, reward, terminated, truncated, info = self.env.step(selected)
        if invalid_action:
            reward += self.invalid_action_penalty
        self._last_observation = observation
        next_candidates = _firs_candidates(observation, info)
        self._last_candidates = next_candidates
        encoded = self._encode_observation(observation, next_candidates)
        return encoded, float(reward), terminated, truncated, self._info_payload(
            {
                **info,
                "selected_action": selected,
                "invalid_action": invalid_action,
                "candidate_actions": next_candidates,
                "action_mask": self.action_masks(),
                "native_observation": observation,
            }
        )

    def render(self) -> str | None:
        if self._last_observation is None:
            return None
        routes = self._last_observation.get("routes", []) or []
        cargo = sum(float(route.get("delivered", 0) or 0) for route in routes)
        line = (
            f"tick={self._last_observation.get('tick')} routes={len(routes)} "
            f"cargo_delivered={round(cargo, 3)}"
        )
        if self.render_mode == "ansi":
            return line
        print(line)
        return None

    def close(self) -> None:
        self.env.close()

    def action_masks(self) -> Any:
        return _mask(self._last_candidates, self.max_candidates)

    def _encode_observation(self, observation: dict[str, Any], candidates: list[dict[str, Any]]) -> dict[str, Any]:
        if np is None:
            raise RuntimeError("OpenTTDFIRSGymEnv requires numpy through the optional gymnasium extra.")

        routes = observation.get("routes", []) or []
        candidate_production = np.zeros((self.max_candidates,), dtype=np.float32)
        for index, candidate in enumerate(candidates[: self.max_candidates]):
            route = candidate.get("route", {}) or {}
            candidate_production[index] = float(route.get("production", 0) or 0)
        cargo_delivered = sum(float(route.get("delivered", 0) or 0) for route in routes)
        route_profit = sum(float(route.get("profit", route.get("vehicle_profit", 0)) or 0) for route in routes)
        delivered_routes = sum(1 for route in routes if float(route.get("delivered", 0) or 0) > 0)
        tick = self.env.executed_steps if self.deterministic and hasattr(self.env, "executed_steps") else observation.get("tick", 0)
        bank_balance = float(observation.get("bank_balance", 0) or 0)
        route_profit_value = float(route_profit)
        if self.deterministic:
            bank_balance = _quantize_money(bank_balance)
            route_profit_value = _quantize_money(route_profit_value)
        return {
            "tick": np.array([float(tick or 0)], dtype=np.float32),
            "bank_balance": np.array([bank_balance], dtype=np.float32),
            "route_count": np.array([float(len(routes))], dtype=np.float32),
            "delivered_routes": np.array([float(delivered_routes)], dtype=np.float32),
            "cargo_delivered": np.array([float(cargo_delivered)], dtype=np.float32),
            "route_profit": np.array([route_profit_value], dtype=np.float32),
            "candidate_production": candidate_production,
            "action_mask": _mask(candidates, self.max_candidates),
        }

    def _info_payload(self, info: dict[str, Any]) -> dict[str, Any]:
        if not self.deterministic:
            return info
        normalized = normalize_gym_info(info)
        normalized["deterministic"] = True
        if "candidate_actions" in normalized:
            normalized["candidate_actions"] = normalize_candidate_actions(info.get("candidate_actions") or [])
        if "native_observation" in info and isinstance(info["native_observation"], dict):
            normalized["native_observation"] = normalize_observation(
                info["native_observation"],
                decision_step=int(getattr(self.env, "executed_steps", 0)),
            )
        if "result" in info:
            normalized["result"] = normalize_result(info.get("result"))
        if "selected_action" in info:
            normalized["selected_action"] = normalize_value(info.get("selected_action"))
        if "action_mask" in info:
            normalized["action_mask"] = normalize_value(info.get("action_mask"))
        return normalized


def make_firs(**kwargs: Any) -> OpenTTDFIRSGymEnv:
    return OpenTTDFIRSGymEnv(**kwargs)


def make_firs_vector(
    num_envs: int,
    *,
    asynchronous: bool = False,
    **env_kwargs: Any,
) -> Any:
    if gym is None:
        raise RuntimeError("make_firs_vector requires the optional 'gymnasium' extra: pip install -e .[gymnasium]")
    constructors: list[Callable[[], OpenTTDFIRSGymEnv]] = []
    for _ in range(num_envs):
        kwargs = dict(env_kwargs)

        def _factory(kwargs: dict[str, Any] = kwargs) -> OpenTTDFIRSGymEnv:
            return OpenTTDFIRSGymEnv(**kwargs)

        constructors.append(_factory)
    vector_cls = gym.vector.AsyncVectorEnv if asynchronous else gym.vector.SyncVectorEnv
    return vector_cls(constructors)


def register_envs() -> None:
    if gym is None:
        return
    _register_once(
        TOY_GYM_ID,
        "openttd_le.adapters.gymnasium:OpenTTDLEGymEnv",
        kwargs={"task_id": "coal_easy_001"},
    )
    _register_once(
        FIRS_GYM_ID,
        "openttd_le.adapters.gymnasium:OpenTTDFIRSGymEnv",
        kwargs={"task_id": "lab_raw_to_processor"},
    )
    _register_once(
        FIRS_DETERMINISTIC_GYM_ID,
        "openttd_le.adapters.gymnasium:OpenTTDFIRSGymEnv",
        kwargs={"task_id": "lab_raw_to_processor", "deterministic": True},
    )


def _register_once(env_id: str, entry_point: str, *, kwargs: dict[str, Any] | None = None) -> None:
    if gym is None:
        return
    try:
        gym.spec(env_id)
        return
    except Exception:
        pass
    gym.register(id=env_id, entry_point=entry_point, kwargs=kwargs or {})


def _firs_candidates(observation: dict[str, Any], info: dict[str, Any]) -> list[dict[str, Any]]:
    candidates = info.get("candidate_actions") or observation.get("candidate_actions") or []
    return list(candidates)


def _mask(candidates: list[dict[str, Any]], max_candidates: int) -> Any:
    if np is None:
        raise RuntimeError("Gymnasium adapter requires numpy.")
    mask = np.zeros((max_candidates,), dtype="int8")
    for index, candidate in enumerate(candidates[:max_candidates]):
        if candidate.get("action") and bool(candidate.get("feasible", True)):
            mask[index] = 1
    return mask


def _quantize_money(value: float) -> float:
    return float(round(value / 100.0) * 100.0)


register_envs()
