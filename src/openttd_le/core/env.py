from __future__ import annotations

from typing import Any

from openttd_le.backends import Backend, make_backend
from openttd_le.core.actions import normalize_action
from openttd_le.core.observation import build_observation
from openttd_le.core.research import candidate_actions_from_observation, decompose_step_reward, preview_action
from openttd_le.core.scenarios import ScenarioRegistry, load_registry
from openttd_le.core.types import EnvError, GameState, Scenario, StepResult


class OpenTTDLEnv:
    def __init__(
        self,
        backend: str | Backend = "toy",
        registry: ScenarioRegistry | None = None,
    ) -> None:
        self.backend = make_backend(backend) if isinstance(backend, str) else backend
        self.registry = registry or load_registry()
        self.scenario: Scenario | None = None
        self.state: GameState | None = None

    def reset(self, scenario_id: str, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.scenario = self.registry.get(scenario_id)
        self.state = self.backend.reset(self.scenario, seed=seed)
        return self.observe(), {"scenario_id": scenario_id, "seed": seed}

    def step(self, action: dict[str, Any]) -> StepResult:
        self._require_reset()
        previous_observation = self.observe()
        previous_score = self.state.metrics.score  # type: ignore[union-attr]
        applied_action = action
        try:
            normalized = normalize_action(action)
            applied_action = normalized
            self.state = self.backend.apply(normalized)
        except EnvError as exc:
            self.state.metrics.invalid_actions += 1  # type: ignore[union-attr]
            self.state.last_event = f"Invalid action: {exc}"  # type: ignore[union-attr]
        reward = self.state.metrics.score - previous_score  # type: ignore[union-attr]
        obs = self.observe()
        reward_details = decompose_step_reward(
            previous_observation,
            obs,
            action=applied_action,
            score_delta=reward,
        )
        return StepResult(
            observation=obs,
            reward=reward,
            terminated=bool(self.state.done),
            truncated=False,
            info={
                "score": self.state.metrics.score,
                "metrics": obs["metrics"],
                "last_event": self.state.last_event,
                "reward_details": reward_details,
            },
        )

    def observe(self) -> dict[str, Any]:
        scenario, state = self._require_reset()
        return build_observation(scenario, state)

    def candidate_actions(self, limit: int = 24) -> list[dict[str, Any]]:
        return candidate_actions_from_observation(self.observe(), limit=limit)

    def preview(self, action: dict[str, Any]) -> dict[str, Any]:
        return preview_action(self.observe(), action)

    def close(self) -> None:
        self.backend.close()

    def artifact_state(self) -> dict[str, Any]:
        return self.backend.artifact_state()

    def _require_reset(self) -> tuple[Scenario, GameState]:
        if self.scenario is None or self.state is None:
            raise RuntimeError("Environment must be reset before use.")
        return self.scenario, self.state
