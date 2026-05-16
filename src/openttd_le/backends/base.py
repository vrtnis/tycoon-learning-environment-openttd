from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any

from openttd_le.core.types import GameState, Scenario


class Backend(ABC):
    """Simulator backend contract for OpenTTD-LE."""

    @abstractmethod
    def reset(self, scenario: Scenario, seed: int | None = None) -> GameState:
        raise NotImplementedError

    @abstractmethod
    def apply(self, action: dict[str, Any]) -> GameState:
        raise NotImplementedError

    @abstractmethod
    def close(self) -> None:
        raise NotImplementedError

    def artifact_state(self) -> dict[str, Any]:
        return {}
