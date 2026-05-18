from __future__ import annotations

from typing import Any

from openttd_le.core.logistics import CARGO_REVENUE, MODE_CONFIG, estimate_route
from openttd_le.core.schemas import ACTION_PREVIEW_SCHEMA, CANDIDATE_ACTION_SCHEMA, STEP_REWARD_SCHEMA


def candidate_actions_from_observation(
    observation: dict[str, Any],
    *,
    limit: int = 24,
) -> list[dict[str, Any]]:
    """Return a reusable action frontier for planning, reranking, and logging."""

    if "nodes" not in observation or "scenario" not in observation:
        return []

    candidates: list[dict[str, Any]] = []
    candidates.extend(_add_vehicle_candidates(observation))
    candidates.extend(_build_route_candidates(observation))
    candidates.extend(_finance_candidates(observation, candidates))
    candidates.extend(_wait_candidates(observation))
    return sorted(candidates, key=_candidate_sort_key)[:limit]


def preview_action(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    """Estimate immediate feasibility and local value without mutating the environment."""

    action_type = action.get("type")
    if action_type == "build_route":
        return _preview_build_route(observation, action)
    if action_type == "add_vehicle":
        return _preview_add_vehicle(observation, action)
    if action_type == "wait":
        return _preview_wait(observation, action)
    if action_type == "take_loan":
        amount = float(action.get("amount", 0) or 0)
        max_loan = float(observation.get("company", {}).get("max_loan", 0) or 0)
        loan = float(observation.get("company", {}).get("loan", 0) or 0)
        feasible = amount > 0 and loan + amount <= max_loan
        return {
            "schema": ACTION_PREVIEW_SCHEMA,
            "feasible": feasible,
            "estimated_score_delta": 0.0,
            "components": {"cash_delta": round(amount, 3), "loan_delta": round(amount, 3)},
            "diagnostics": [] if feasible else ["loan_cap_or_nonpositive_amount"],
        }
    if action_type == "repay_loan":
        amount = float(action.get("amount", 0) or 0)
        cash = float(observation.get("company", {}).get("cash", 0) or 0)
        loan = float(observation.get("company", {}).get("loan", 0) or 0)
        feasible = amount > 0 and amount <= cash and loan > 0
        return {
            "schema": ACTION_PREVIEW_SCHEMA,
            "feasible": feasible,
            "estimated_score_delta": 0.0,
            "components": {"cash_delta": round(-amount, 3), "loan_delta": round(-min(amount, loan), 3)},
            "diagnostics": [] if feasible else ["insufficient_cash_or_no_loan"],
        }
    return {
        "schema": ACTION_PREVIEW_SCHEMA,
        "feasible": False,
        "estimated_score_delta": 0.0,
        "components": {},
        "diagnostics": [f"unsupported_action:{action_type}"],
    }


def decompose_step_reward(
    previous_observation: dict[str, Any],
    current_observation: dict[str, Any],
    *,
    action: dict[str, Any],
    score_delta: float,
) -> dict[str, Any]:
    previous_metrics = previous_observation.get("metrics", {})
    current_metrics = current_observation.get("metrics", {})
    previous_company = previous_observation.get("company", {})
    current_company = current_observation.get("company", {})

    route_delta = int(current_metrics.get("route_count", 0) or 0) - int(previous_metrics.get("route_count", 0) or 0)
    vehicle_delta = int(current_metrics.get("vehicles", 0) or 0) - int(previous_metrics.get("vehicles", 0) or 0)
    invalid_delta = int(current_metrics.get("invalid_actions", 0) or 0) - int(previous_metrics.get("invalid_actions", 0) or 0)
    cargo_delta = float(current_metrics.get("cargo_delivered", 0) or 0) - float(previous_metrics.get("cargo_delivered", 0) or 0)
    profit_delta = float(current_metrics.get("operating_profit", 0) or 0) - float(previous_metrics.get("operating_profit", 0) or 0)
    first_delivery = (
        previous_metrics.get("first_delivery_month") is None
        and current_metrics.get("first_delivery_month") is not None
    )

    components = {
        "score_delta": round(score_delta, 4),
        "cargo_delta": round(cargo_delta, 4),
        "profit_delta": round(profit_delta, 4),
        "route_delta": route_delta,
        "vehicle_delta": vehicle_delta,
        "cash_delta": round(float(current_company.get("cash", 0) or 0) - float(previous_company.get("cash", 0) or 0), 4),
        "loan_delta": round(float(current_company.get("loan", 0) or 0) - float(previous_company.get("loan", 0) or 0), 4),
        "first_delivery": 1 if first_delivery else 0,
        "invalid_action": invalid_delta,
    }
    milestones = [
        name
        for name, active in (
            ("route_built", route_delta > 0),
            ("vehicles_added", vehicle_delta > 0),
            ("first_delivery", first_delivery),
            ("cargo_delivered", cargo_delta > 0),
            ("positive_profit_delta", profit_delta > 0),
        )
        if active
    ]
    diagnostics = []
    if invalid_delta > 0:
        diagnostics.append("action_rejected")
    if action.get("type") == "wait" and cargo_delta <= 0 and previous_metrics.get("vehicles", 0):
        diagnostics.append("wait_without_delivery")
    return {
        "schema": STEP_REWARD_SCHEMA,
        "reward": round(score_delta, 4),
        "components": components,
        "milestones": milestones,
        "diagnostics": diagnostics,
    }


def _build_route_candidates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    terrain_cost = float(observation["scenario"]["map"].get("terrain_cost", 1.0) or 1.0)
    goal_cargo = observation["scenario"]["goals"].get("cargo")
    cash = float(observation["company"]["cash"])
    loan = float(observation["company"]["loan"])
    max_loan = float(observation["company"]["max_loan"])
    finance_available = cash + max(0.0, max_loan - loan)
    existing = {
        (route["source_id"], route["destination_id"], route["cargo"])
        for route in observation.get("routes", [])
    }
    candidates: list[dict[str, Any]] = []
    for source in observation["nodes"]:
        for cargo in sorted((source.get("produces") or {}).keys()):
            if goal_cargo and cargo != goal_cargo:
                continue
            for destination in observation["nodes"]:
                if destination["id"] == source["id"] or cargo not in (destination.get("accepts") or {}):
                    continue
                if (source["id"], destination["id"], cargo) in existing:
                    continue
                for mode in ("rail", "road"):
                    estimate = estimate_route(
                        source=source,
                        destination=destination,
                        cargo=cargo,
                        mode=mode,
                        terrain_cost=terrain_cost,
                    )
                    action = {
                        "type": "build_route",
                        "source_id": source["id"],
                        "destination_id": destination["id"],
                        "cargo": cargo,
                        "mode": mode,
                    }
                    build_cost = estimate["build_cost"]
                    objective_relevance = 1.0 if not goal_cargo or cargo == goal_cargo else 0.25
                    feasible_with_financing = finance_available >= build_cost
                    candidates.append(
                        {
                            "schema": CANDIDATE_ACTION_SCHEMA,
                            "id": _action_id(action),
                            "kind": "build_route",
                            "action": action,
                            "feasible": feasible_with_financing,
                            "directly_executable": cash >= build_cost,
                            "requires_loan": max(0.0, build_cost - cash),
                            "rank_score": round(estimate["roi"] * 1000.0 + objective_relevance, 6),
                            "estimates": estimate,
                            "objective_relevance": objective_relevance,
                            "diagnostics": [] if feasible_with_financing else ["insufficient_cash_even_with_loan"],
                            "description": (
                                f"Build {mode} route for {cargo} from {source['name']} "
                                f"to {destination['name']}."
                            ),
                        }
                    )
    return candidates


def _add_vehicle_candidates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    cash = float(observation["company"]["cash"])
    candidates: list[dict[str, Any]] = []
    for route in observation.get("routes", []):
        if int(route.get("vehicles", 0) or 0) >= 5:
            continue
        count = 1
        cost = float(route.get("vehicle_cost", 0) or 0) * count
        action = {"type": "add_vehicle", "route_id": route["id"], "count": count}
        monthly_profit = _route_monthly_profit_estimate(route, count)
        candidates.append(
            {
                "schema": CANDIDATE_ACTION_SCHEMA,
                "id": _action_id(action),
                "kind": "add_vehicle",
                "action": action,
                "feasible": True,
                "directly_executable": cash >= cost,
                "requires_loan": max(0.0, cost - cash),
                "rank_score": round(monthly_profit / max(1.0, cost), 6),
                "estimates": {
                    "vehicle_cost": round(cost, 3),
                    "estimated_monthly_profit_delta": round(monthly_profit, 3),
                },
                "objective_relevance": 0.8,
                "diagnostics": [],
                "description": f"Add one vehicle to route {route['id']}.",
            }
        )
    return candidates


def _finance_candidates(observation: dict[str, Any], candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cash = float(observation["company"]["cash"])
    loan = float(observation["company"]["loan"])
    max_loan = float(observation["company"]["max_loan"])
    finance_candidates: list[dict[str, Any]] = []
    missing = [
        float(item.get("requires_loan", 0) or 0)
        for item in candidates
        if item.get("feasible") and not item.get("directly_executable") and float(item.get("requires_loan", 0) or 0) > 0
    ]
    if missing and loan < max_loan:
        amount = min(max_loan - loan, max(missing) + 10_000)
        action = {"type": "take_loan", "amount": round(amount, 2)}
        finance_candidates.append(
            {
                "schema": CANDIDATE_ACTION_SCHEMA,
                "id": _action_id(action),
                "kind": "take_loan",
                "action": action,
                "feasible": amount > 0,
                "directly_executable": amount > 0,
                "requires_loan": 0.0,
                "rank_score": 0.2,
                "estimates": {"cash_after": round(cash + amount, 3), "loan_after": round(loan + amount, 3)},
                "objective_relevance": 0.5,
                "diagnostics": [],
                "description": "Take a loan to unlock a high-ranked route or vehicle action.",
            }
        )
    if cash > 80_000 and loan > 0:
        amount = min(loan, cash - 50_000)
        action = {"type": "repay_loan", "amount": round(amount, 2)}
        finance_candidates.append(
            {
                "schema": CANDIDATE_ACTION_SCHEMA,
                "id": _action_id(action),
                "kind": "repay_loan",
                "action": action,
                "feasible": amount > 0,
                "directly_executable": amount > 0,
                "requires_loan": 0.0,
                "rank_score": 0.05,
                "estimates": {"cash_after": round(cash - amount, 3), "loan_after": round(loan - amount, 3)},
                "objective_relevance": 0.2,
                "diagnostics": [],
                "description": "Repay debt while retaining operating cash.",
            }
        )
    return finance_candidates


def _wait_candidates(observation: dict[str, Any]) -> list[dict[str, Any]]:
    months = [1, 3] if observation.get("routes") else [1]
    candidates: list[dict[str, Any]] = []
    for month_count in months:
        action = {"type": "wait", "months": month_count}
        preview = _preview_wait(observation, action)
        candidates.append(
            {
                "schema": CANDIDATE_ACTION_SCHEMA,
                "id": _action_id(action),
                "kind": "wait",
                "action": action,
                "feasible": True,
                "directly_executable": True,
                "requires_loan": 0.0,
                "rank_score": round(float(preview["components"].get("profit_delta", 0) or 0) / 1000.0, 6),
                "estimates": preview["components"],
                "objective_relevance": 0.1,
                "diagnostics": preview["diagnostics"],
                "description": f"Advance simulation by {month_count} month(s).",
            }
        )
    return candidates


def _preview_build_route(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    source = _node_by_id(observation, str(action.get("source_id", "")))
    destination = _node_by_id(observation, str(action.get("destination_id", "")))
    diagnostics = []
    if not source:
        diagnostics.append("unknown_source")
    if not destination:
        diagnostics.append("unknown_destination")
    cargo = str(action.get("cargo", ""))
    mode = str(action.get("mode", ""))
    if mode not in MODE_CONFIG:
        diagnostics.append("unknown_mode")
    if source and cargo not in (source.get("produces") or {}):
        diagnostics.append("source_does_not_produce_cargo")
    if destination and cargo not in (destination.get("accepts") or {}):
        diagnostics.append("destination_does_not_accept_cargo")
    if source and destination and source["id"] == destination["id"]:
        diagnostics.append("source_equals_destination")
    if any(
        route.get("source_id") == action.get("source_id")
        and route.get("destination_id") == action.get("destination_id")
        and route.get("cargo") == cargo
        for route in observation.get("routes", [])
    ):
        diagnostics.append("duplicate_route")
    estimates: dict[str, Any] = {}
    if source and destination and mode in MODE_CONFIG:
        estimates = estimate_route(
            source=source,
            destination=destination,
            cargo=cargo,
            mode=mode,
            terrain_cost=float(observation["scenario"]["map"].get("terrain_cost", 1.0) or 1.0),
        )
        cash = float(observation.get("company", {}).get("cash", 0) or 0)
        if cash < estimates["build_cost"]:
            diagnostics.append("insufficient_cash")
    feasible = not diagnostics
    return {
        "schema": ACTION_PREVIEW_SCHEMA,
        "feasible": feasible,
        "estimated_score_delta": 0.0,
        "components": estimates,
        "diagnostics": diagnostics,
    }


def _preview_add_vehicle(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    route = _route_by_id(observation, str(action.get("route_id", "")))
    diagnostics = []
    if not route:
        diagnostics.append("unknown_route")
        return {
            "schema": ACTION_PREVIEW_SCHEMA,
            "feasible": False,
            "estimated_score_delta": 0.0,
            "components": {},
            "diagnostics": diagnostics,
        }
    count = max(1, int(action.get("count", 1) or 1))
    cost = float(route.get("vehicle_cost", 0) or 0) * count
    cash = float(observation.get("company", {}).get("cash", 0) or 0)
    if cash < cost:
        diagnostics.append("insufficient_cash")
    profit_delta = _route_monthly_profit_estimate(route, count)
    return {
        "schema": ACTION_PREVIEW_SCHEMA,
        "feasible": not diagnostics,
        "estimated_score_delta": 0.0,
        "components": {
            "vehicle_cost": round(cost, 3),
            "estimated_monthly_profit_delta": round(profit_delta, 3),
        },
        "diagnostics": diagnostics,
    }


def _preview_wait(observation: dict[str, Any], action: dict[str, Any]) -> dict[str, Any]:
    months = max(1, int(action.get("months", 1) or 1))
    delivered = 0.0
    profit = 0.0
    for route in observation.get("routes", []):
        vehicles = int(route.get("vehicles", 0) or 0)
        if vehicles <= 0:
            continue
        monthly = _route_monthly_profit_estimate(route, 0)
        profit += monthly * months
        delivered += max(0.0, float(route.get("vehicles", 0) or 0) * 8.0 * months)
    diagnostics = []
    if observation.get("routes") and delivered <= 0:
        diagnostics.append("routes_have_no_vehicles")
    return {
        "schema": ACTION_PREVIEW_SCHEMA,
        "feasible": True,
        "estimated_score_delta": 0.0,
        "components": {
            "months": months,
            "cargo_delta": round(delivered, 3),
            "profit_delta": round(profit, 3),
        },
        "diagnostics": diagnostics,
    }


def _route_monthly_profit_estimate(route: dict[str, Any], extra_vehicles: int) -> float:
    vehicles = int(route.get("vehicles", 0) or 0) + extra_vehicles
    if vehicles <= 0:
        return 0.0
    mode = str(route.get("mode", "road"))
    cfg = MODE_CONFIG.get(mode, MODE_CONFIG["road"])
    distance = float(route.get("distance", 1) or 1)
    cargo = str(route.get("cargo", ""))
    capacity = cfg["vehicle_capacity"] * cfg["speed_factor"] * vehicles
    distance_drag = max(0.35, 1.0 - distance / 240.0)
    delivered = capacity * distance_drag
    revenue = delivered * 9.0 * (1.0 + distance / 120.0)
    operating_cost = vehicles * cfg["maintenance"] + distance * 8.0
    # Existing observation lacks source/destination production, so this is a lower-fidelity preview.
    if cargo in CARGO_REVENUE:
        revenue = delivered * CARGO_REVENUE[cargo] * (1.0 + distance / 120.0)
    return revenue - operating_cost


def _node_by_id(observation: dict[str, Any], node_id: str) -> dict[str, Any] | None:
    for node in observation.get("nodes", []):
        if str(node.get("id")) == node_id:
            return node
    return None


def _route_by_id(observation: dict[str, Any], route_id: str) -> dict[str, Any] | None:
    for route in observation.get("routes", []):
        if str(route.get("id")) == route_id:
            return route
    return None


def _action_id(action: dict[str, Any]) -> str:
    action_type = str(action.get("type", "unknown"))
    if action_type == "build_route":
        return ":".join([action_type, str(action.get("source_id")), str(action.get("destination_id")), str(action.get("cargo")), str(action.get("mode"))])
    if action_type == "add_vehicle":
        return ":".join([action_type, str(action.get("route_id")), str(action.get("count", 1))])
    if action_type in {"wait", "take_loan", "repay_loan"}:
        return ":".join([action_type, str(action.get("months", action.get("amount", "")))])
    return action_type


def _candidate_sort_key(candidate: dict[str, Any]) -> tuple[int, int, float]:
    executable_rank = 0 if candidate.get("directly_executable") else 1
    feasible_rank = 0 if candidate.get("feasible") else 1
    return (feasible_rank, executable_rank, -float(candidate.get("rank_score", 0) or 0))
