from __future__ import annotations

from dataclasses import asdict
from typing import Any

from .types import GameState, Scenario


def build_observation(scenario: Scenario, state: GameState) -> dict[str, Any]:
    return {
        "scenario": {
            "id": scenario.id,
            "name": scenario.name,
            "task": scenario.task,
            "tags": scenario.tags,
            "map": asdict(scenario.map),
            "goals": asdict(scenario.goals),
        },
        "time": {
            "month": state.month,
            "step": state.step,
            "max_months": scenario.budget.max_months,
            "max_steps": scenario.budget.max_steps,
        },
        "company": {
            "cash": round(state.cash, 2),
            "loan": round(state.loan, 2),
            "max_loan": scenario.budget.max_loan,
        },
        "nodes": [
            {
                "id": node.id,
                "name": node.name,
                "kind": node.kind,
                "x": node.x,
                "y": node.y,
                "produces": node.produces,
                "accepts": node.accepts,
                "population": node.population,
            }
            for node in scenario.nodes
        ],
        "routes": [
            {
                "id": route.id,
                "source_id": route.source_id,
                "destination_id": route.destination_id,
                "cargo": route.cargo,
                "mode": route.mode,
                "distance": round(route.distance, 2),
                "vehicle_cost": round(route.vehicle_cost, 2),
                "vehicles": route.vehicles,
                "delivered": round(route.delivered, 2),
                "revenue": round(route.revenue, 2),
                "operating_cost": round(route.operating_cost, 2),
                "operating_profit": round(route.operating_profit, 2),
            }
            for route in state.routes
        ],
        "metrics": {
            "score": state.metrics.score,
            "cargo_delivered": state.metrics.cargo_delivered,
            "cargo_by_type": state.metrics.cargo_by_type,
            "operating_profit": state.metrics.operating_profit,
            "route_count": state.metrics.route_count,
            "vehicles": state.metrics.vehicles,
            "invalid_actions": state.metrics.invalid_actions,
            "first_delivery_month": state.metrics.first_delivery_month,
            "utilization": state.metrics.utilization,
            "breakdown": state.metrics.breakdown,
        },
        "last_event": state.last_event,
        "allowed_actions": [
            "build_route",
            "add_vehicle",
            "wait",
            "take_loan",
            "repay_loan",
        ],
    }
