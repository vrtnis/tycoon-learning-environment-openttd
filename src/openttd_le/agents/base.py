from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class Agent(ABC):
    name = "agent"

    @abstractmethod
    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        raise NotImplementedError

    def close(self) -> None:
        return None
