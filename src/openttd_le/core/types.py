from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


TransportMode = Literal["road", "rail"]
NodeKind = Literal["town", "industry"]
ActionType = Literal["build_route", "add_vehicle", "wait", "take_loan", "repay_loan"]


@dataclass(frozen=True)
class Node:
    id: str
    name: str
    kind: NodeKind
    x: int
    y: int
    produces: dict[str, float] = field(default_factory=dict)
    accepts: dict[str, float] = field(default_factory=dict)
    population: int = 0


@dataclass(frozen=True)
class Budget:
    max_steps: int
    max_months: int
    starting_cash: float
    max_loan: float
    interest_rate: float


@dataclass(frozen=True)
class MapConfig:
    width: int
    height: int
    terrain_cost: float = 1.0


@dataclass(frozen=True)
class Goals:
    cargo: str | None = None
    cargo_delivered: float = 0.0
    operating_profit: float = 0.0
    network_routes: int = 1
    max_debt_ratio: float = 0.8


@dataclass(frozen=True)
class Scenario:
    id: str
    name: str
    task: str
    map: MapConfig
    budget: Budget
    goals: Goals
    nodes: list[Node]
    tags: list[str] = field(default_factory=list)


@dataclass
class Route:
    id: str
    source_id: str
    destination_id: str
    cargo: str
    mode: TransportMode
    distance: float
    build_cost: float
    vehicle_cost: float
    vehicles: int = 0
    delivered: float = 0.0
    revenue: float = 0.0
    operating_cost: float = 0.0
    months_active: int = 0

    @property
    def operating_profit(self) -> float:
        return self.revenue - self.operating_cost


@dataclass
class Metrics:
    score: float = 0.0
    cargo_delivered: float = 0.0
    operating_profit: float = 0.0
    cash: float = 0.0
    loan: float = 0.0
    route_count: int = 0
    vehicles: int = 0
    invalid_actions: int = 0
    first_delivery_month: int | None = None
    utilization: float = 0.0
    cargo_by_type: dict[str, float] = field(default_factory=dict)
    breakdown: dict[str, float] = field(default_factory=dict)


@dataclass
class GameState:
    scenario_id: str
    month: int
    step: int
    cash: float
    loan: float
    routes: list[Route] = field(default_factory=list)
    metrics: Metrics = field(default_factory=Metrics)
    done: bool = False
    last_event: str = "Scenario loaded."


@dataclass
class StepResult:
    observation: dict[str, Any]
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class EnvError(RuntimeError):
    """Raised when an environment action cannot be applied."""


def distance(a: Node, b: Node) -> float:
    return abs(a.x - b.x) + abs(a.y - b.y)
