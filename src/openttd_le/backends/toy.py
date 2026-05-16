from __future__ import annotations

import random
from dataclasses import asdict
from typing import Any

from openttd_le.backends.base import Backend
from openttd_le.core.scoring import score_state
from openttd_le.core.types import EnvError, GameState, Metrics, Node, Route, Scenario, distance


MODE_CONFIG = {
    "road": {
        "track_cost": 900.0,
        "station_cost": 9_000.0,
        "vehicle_cost": 18_000.0,
        "vehicle_capacity": 26.0,
        "maintenance": 520.0,
        "speed_factor": 0.72,
    },
    "rail": {
        "track_cost": 2_150.0,
        "station_cost": 24_000.0,
        "vehicle_cost": 42_000.0,
        "vehicle_capacity": 70.0,
        "maintenance": 1_250.0,
        "speed_factor": 1.12,
    },
}

CARGO_REVENUE = {
    "coal": 40.0,
    "oil": 45.0,
    "wood": 35.0,
    "grain": 32.0,
    "goods": 50.0,
    "passengers": 28.0,
    "mail": 50.0,
}


class ToyLogisticsBackend(Backend):
    """A deterministic logistics simulator for exercising OpenTTD-LE contracts.

    This backend is intentionally small. It lets us build the benchmark, agents,
    scorecards, and artifacts before the real OpenTTD bridge is ready.
    """

    def __init__(self) -> None:
        self._scenario: Scenario | None = None
        self._state: GameState | None = None
        self._random = random.Random(0)

    def reset(self, scenario: Scenario, seed: int | None = None) -> GameState:
        self._scenario = scenario
        self._random = random.Random(seed or 0)
        self._state = GameState(
            scenario_id=scenario.id,
            month=0,
            step=0,
            cash=scenario.budget.starting_cash,
            loan=0.0,
            metrics=Metrics(cash=scenario.budget.starting_cash),
            last_event=f"{scenario.id} loaded.",
        )
        score_state(scenario, self._state)
        return self._state

    def apply(self, action: dict[str, Any]) -> GameState:
        scenario, state = self._require_state()
        previous_score = state.metrics.score
        try:
            if action["type"] == "build_route":
                self._build_route(action)
            elif action["type"] == "add_vehicle":
                self._add_vehicle(action)
            elif action["type"] == "wait":
                self._wait(int(action["months"]))
            elif action["type"] == "take_loan":
                self._take_loan(float(action["amount"]))
            elif action["type"] == "repay_loan":
                self._repay_loan(float(action["amount"]))
            else:
                raise EnvError(f"Unhandled action type: {action['type']}")
        except EnvError as exc:
            state.metrics.invalid_actions += 1
            state.last_event = f"Invalid action: {exc}"

        state.step += 1
        state.done = state.step >= scenario.budget.max_steps or state.month >= scenario.budget.max_months
        score_state(scenario, state)
        state.last_event = f"{state.last_event} Score delta {state.metrics.score - previous_score:.2f}."
        return state

    def close(self) -> None:
        self._scenario = None
        self._state = None

    def artifact_state(self) -> dict[str, Any]:
        scenario, state = self._require_state()
        return {
            "backend": "toy",
            "scenario": asdict(scenario),
            "state": asdict(state),
        }

    def _build_route(self, action: dict[str, Any]) -> None:
        scenario, state = self._require_state()
        source = self._node(action["source_id"])
        destination = self._node(action["destination_id"])
        cargo = action["cargo"]
        mode = action["mode"]
        if cargo not in source.produces:
            raise EnvError(f"{source.id} does not produce {cargo}.")
        if cargo not in destination.accepts:
            raise EnvError(f"{destination.id} does not accept {cargo}.")
        if source.id == destination.id:
            raise EnvError("source and destination must differ.")
        if self._route_exists(source.id, destination.id, cargo):
            raise EnvError("route already exists.")

        cfg = MODE_CONFIG[mode]
        route_distance = max(1.0, distance(source, destination))
        build_cost = (
            cfg["station_cost"] * 2.0
            + cfg["track_cost"] * route_distance * scenario.map.terrain_cost
        )
        if state.cash < build_cost:
            raise EnvError(f"insufficient cash for route: need {build_cost:.0f}, have {state.cash:.0f}.")

        state.cash -= build_cost
        route = Route(
            id=f"R{len(state.routes) + 1:03d}",
            source_id=source.id,
            destination_id=destination.id,
            cargo=cargo,
            mode=mode,
            distance=route_distance,
            build_cost=build_cost,
            vehicle_cost=cfg["vehicle_cost"],
        )
        state.routes.append(route)
        state.last_event = (
            f"Built {mode} route {route.id} from {source.name} to "
            f"{destination.name} for {cargo}; cost {build_cost:.0f}."
        )

    def _add_vehicle(self, action: dict[str, Any]) -> None:
        _, state = self._require_state()
        route = self._route(action["route_id"])
        count = int(action["count"])
        cost = route.vehicle_cost * count
        if state.cash < cost:
            raise EnvError(f"insufficient cash for vehicles: need {cost:.0f}, have {state.cash:.0f}.")
        state.cash -= cost
        route.vehicles += count
        state.last_event = f"Added {count} vehicle(s) to {route.id}; cost {cost:.0f}."

    def _wait(self, months: int) -> None:
        scenario, state = self._require_state()
        months = max(1, min(months, scenario.budget.max_months - state.month))
        delivered_this_wait = 0.0
        profit_this_wait = 0.0
        for _ in range(months):
            if state.month >= scenario.budget.max_months:
                break
            state.month += 1
            interest = state.loan * scenario.budget.interest_rate / 12.0
            state.cash -= interest
            for route in state.routes:
                delivered, profit = self._operate_route(route)
                delivered_this_wait += delivered
                profit_this_wait += profit
        state.last_event = (
            f"Advanced {months} month(s); delivered {delivered_this_wait:.1f} units "
            f"with operating profit {profit_this_wait:.0f}."
        )

    def _operate_route(self, route: Route) -> tuple[float, float]:
        _, state = self._require_state()
        source = self._node(route.source_id)
        destination = self._node(route.destination_id)
        cfg = MODE_CONFIG[route.mode]
        if route.vehicles <= 0:
            route.months_active += 1
            return 0.0, 0.0

        source_limit = source.produces[route.cargo]
        dest_limit = destination.accepts[route.cargo]
        vehicle_capacity = cfg["vehicle_capacity"] * cfg["speed_factor"] * route.vehicles
        distance_drag = max(0.35, 1.0 - route.distance / 240.0)
        delivered = min(source_limit, dest_limit, vehicle_capacity * distance_drag)
        revenue_rate = CARGO_REVENUE.get(route.cargo, 9.0)
        revenue = delivered * revenue_rate * (1.0 + route.distance / 120.0)
        operating_cost = route.vehicles * cfg["maintenance"] + route.distance * 8.0
        profit = revenue - operating_cost

        route.delivered += delivered
        route.revenue += revenue
        route.operating_cost += operating_cost
        route.months_active += 1
        state.cash += profit
        state.metrics.cargo_by_type[route.cargo] = state.metrics.cargo_by_type.get(route.cargo, 0.0) + delivered
        if delivered > 0 and state.metrics.first_delivery_month is None:
            state.metrics.first_delivery_month = state.month
        return delivered, profit

    def _take_loan(self, amount: float) -> None:
        scenario, state = self._require_state()
        amount = min(amount, scenario.budget.max_loan - state.loan)
        if amount <= 0:
            raise EnvError("loan cap reached.")
        state.loan += amount
        state.cash += amount
        state.last_event = f"Took loan of {amount:.0f}."

    def _repay_loan(self, amount: float) -> None:
        _, state = self._require_state()
        amount = min(amount, state.loan)
        if amount <= 0:
            raise EnvError("no outstanding loan.")
        if state.cash < amount:
            raise EnvError(f"insufficient cash to repay {amount:.0f}.")
        state.cash -= amount
        state.loan -= amount
        state.last_event = f"Repaid loan of {amount:.0f}."

    def _node(self, node_id: str) -> Node:
        scenario, _ = self._require_state()
        for node in scenario.nodes:
            if node.id == node_id:
                return node
        raise EnvError(f"unknown node: {node_id}.")

    def _route(self, route_id: str) -> Route:
        _, state = self._require_state()
        for route in state.routes:
            if route.id == route_id:
                return route
        raise EnvError(f"unknown route: {route_id}.")

    def _route_exists(self, source_id: str, destination_id: str, cargo: str) -> bool:
        _, state = self._require_state()
        return any(
            route.source_id == source_id
            and route.destination_id == destination_id
            and route.cargo == cargo
            for route in state.routes
        )

    def _require_state(self) -> tuple[Scenario, GameState]:
        if self._scenario is None or self._state is None:
            raise EnvError("backend has not been reset.")
        return self._scenario, self._state
