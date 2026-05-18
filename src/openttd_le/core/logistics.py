from __future__ import annotations

from typing import Any


MODE_CONFIG: dict[str, dict[str, float]] = {
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

CARGO_REVENUE: dict[str, float] = {
    "coal": 40.0,
    "oil": 45.0,
    "wood": 35.0,
    "grain": 32.0,
    "goods": 50.0,
    "passengers": 28.0,
    "mail": 50.0,
}


def estimate_route(
    *,
    source: dict[str, Any],
    destination: dict[str, Any],
    cargo: str,
    mode: str,
    terrain_cost: float = 1.0,
    vehicles: int = 1,
) -> dict[str, float]:
    cfg = MODE_CONFIG[mode]
    distance = float(abs(int(source["x"]) - int(destination["x"])) + abs(int(source["y"]) - int(destination["y"])))
    distance = max(1.0, distance)
    build_cost = cfg["station_cost"] * 2.0 + cfg["track_cost"] * distance * terrain_cost
    vehicle_cost = cfg["vehicle_cost"] * vehicles
    produced = float((source.get("produces") or {}).get(cargo, 0.0) or 0.0)
    accepted = float((destination.get("accepts") or {}).get(cargo, 0.0) or 0.0)
    capacity = cfg["vehicle_capacity"] * cfg["speed_factor"] * vehicles
    distance_drag = max(0.35, 1.0 - distance / 240.0)
    monthly_delivered = min(produced, accepted, capacity * distance_drag)
    monthly_revenue = monthly_delivered * CARGO_REVENUE.get(cargo, 9.0) * (1.0 + distance / 120.0)
    monthly_operating_cost = cfg["maintenance"] * vehicles + distance * 8.0
    monthly_profit = monthly_revenue - monthly_operating_cost
    total_capex = build_cost + vehicle_cost
    roi = monthly_profit / max(1.0, total_capex)
    payback_months = total_capex / monthly_profit if monthly_profit > 0 else 9999.0
    return {
        "distance": round(distance, 3),
        "build_cost": round(build_cost, 3),
        "vehicle_cost": round(vehicle_cost, 3),
        "total_capex": round(total_capex, 3),
        "monthly_delivered": round(monthly_delivered, 3),
        "monthly_revenue": round(monthly_revenue, 3),
        "monthly_operating_cost": round(monthly_operating_cost, 3),
        "monthly_profit": round(monthly_profit, 3),
        "roi": round(roi, 6),
        "payback_months": round(payback_months, 3),
    }
