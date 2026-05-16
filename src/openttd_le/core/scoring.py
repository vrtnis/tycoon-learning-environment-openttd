from __future__ import annotations

from .types import GameState, Metrics, Scenario


def score_state(scenario: Scenario, state: GameState) -> Metrics:
    metrics = state.metrics
    cargo_target = max(1.0, scenario.goals.cargo_delivered)
    profit_target = max(1.0, scenario.goals.operating_profit)
    target_cargo = scenario.goals.cargo

    cargo_delivered = (
        metrics.cargo_by_type.get(target_cargo, 0.0)
        if target_cargo
        else metrics.cargo_delivered
    )
    operating_profit = sum(route.operating_profit for route in state.routes)
    vehicle_count = sum(route.vehicles for route in state.routes)
    route_count = len(state.routes)
    utilization = _mean_utilization(state)

    cargo_score = min(35.0, 35.0 * cargo_delivered / cargo_target)
    profit_score = min(25.0, 25.0 * max(0.0, operating_profit) / profit_target)
    debt_ratio = state.loan / max(1.0, scenario.budget.max_loan)
    debt_score = max(0.0, 15.0 * (1.0 - debt_ratio / max(0.01, scenario.goals.max_debt_ratio)))
    route_score = min(10.0, 10.0 * route_count / max(1, scenario.goals.network_routes))
    first_delivery_score = 0.0
    if metrics.first_delivery_month is not None:
        first_delivery_score = max(
            0.0,
            10.0 * (1.0 - metrics.first_delivery_month / max(1, scenario.budget.max_months)),
        )
    utilization_score = min(5.0, 5.0 * utilization)
    invalid_penalty = min(20.0, metrics.invalid_actions * 3.0)

    score = max(
        0.0,
        min(
            100.0,
            cargo_score
            + profit_score
            + debt_score
            + route_score
            + first_delivery_score
            + utilization_score
            - invalid_penalty,
        ),
    )

    metrics.score = round(score, 3)
    metrics.cargo_delivered = round(sum(metrics.cargo_by_type.values()), 3)
    metrics.operating_profit = round(operating_profit, 3)
    metrics.cash = round(state.cash, 3)
    metrics.loan = round(state.loan, 3)
    metrics.route_count = route_count
    metrics.vehicles = vehicle_count
    metrics.utilization = round(utilization, 3)
    metrics.breakdown = {
        "cargo": round(cargo_score, 3),
        "profit": round(profit_score, 3),
        "debt": round(debt_score, 3),
        "routes": round(route_score, 3),
        "first_delivery": round(first_delivery_score, 3),
        "utilization": round(utilization_score, 3),
        "invalid_penalty": round(-invalid_penalty, 3),
    }
    return metrics


def _mean_utilization(state: GameState) -> float:
    utilizations = []
    for route in state.routes:
        if route.vehicles <= 0 or route.months_active <= 0:
            continue
        theoretical = route.vehicles * route.months_active
        if theoretical <= 0:
            continue
        utilizations.append(min(1.0, route.delivered / max(1.0, theoretical * 50.0)))
    if not utilizations:
        return 0.0
    return sum(utilizations) / len(utilizations)
