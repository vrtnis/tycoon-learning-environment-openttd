from __future__ import annotations

from typing import Any


CARGO_VALUE: dict[str, float] = {
    "PASS": 0.5,
    "MAIL": 0.5,
    "COAL": 1.0,
    "IORE": 1.0,
    "WOOD": 1.0,
    "SAND": 1.0,
    "KAOL": 1.2,
    "SCMT": 1.2,
    "FISH": 1.4,
    "FRUT": 1.4,
    "MILK": 1.4,
    "LVST": 1.5,
    "ENSP": 1.8,
    "FMSP": 1.8,
    "CHEM": 2.0,
    "FOOD": 2.4,
    "STEL": 2.8,
    "GOOD": 3.0,
    "BEER": 3.0,
}


def cargo_value(label: str) -> float:
    return CARGO_VALUE.get(label.upper(), 1.0)


def delivered_cargo_value(routes: list[dict[str, Any]]) -> float:
    total = 0.0
    for route in routes:
        delivered = float(route.get("delivered", 0) or 0)
        label = str(route.get("cargo_label", route.get("cargo", ""))).upper()
        total += delivered * cargo_value(label)
    return total


def score_snapshot(observation: dict[str, Any], objectives: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    routes = observation.get("routes", []) or []
    cargo_score = delivered_cargo_value(routes)
    route_profit = sum(float(route.get("profit", route.get("vehicle_profit", 0)) or 0) for route in routes)
    route_count = len(routes)
    delivered_routes = sum(1 for route in routes if int(route.get("delivered", 0) or 0) > 0)
    target_labels = {str(item.get("cargo", "")).upper() for item in (objectives or [])[1:] if item.get("cargo")}
    processed_deliveries = sum(
        int(route.get("delivered", 0) or 0)
        for route in routes
        if str(route.get("cargo_label", "")).upper() in target_labels
    )
    network_value = cargo_score + max(0.0, route_profit / 1000.0) + route_count * 2.0 + delivered_routes * 5.0
    return {
        "cargo_score": round(cargo_score, 3),
        "network_value": round(network_value, 3),
        "route_profit": round(route_profit, 3),
        "route_count": route_count,
        "delivered_routes": delivered_routes,
        "processed_deliveries": processed_deliveries,
    }
