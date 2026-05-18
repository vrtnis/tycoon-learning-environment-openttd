from __future__ import annotations

from typing import Any

from openttd_le.core.logistics import CARGO_REVENUE, MODE_CONFIG

from .base import Agent


class GreedyAgent(Agent):
    name = "greedy"

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        frontier_action = self._frontier_action(observation)
        if frontier_action:
            return frontier_action

        cash = observation["company"]["cash"]
        loan = observation["company"]["loan"]
        max_loan = observation["company"]["max_loan"]

        route_needing_vehicle = self._route_needing_vehicle(observation)
        if route_needing_vehicle:
            cost = route_needing_vehicle["vehicle_cost"]
            if cash >= cost:
                return {"type": "add_vehicle", "route_id": route_needing_vehicle["id"], "count": 1}
            if loan < max_loan:
                return {"type": "take_loan", "amount": min(max_loan - loan, cost - cash + 10_000)}

        best = self._best_route_candidate(observation)
        if best:
            if cash >= best["estimated_build_cost"]:
                return {
                    "type": "build_route",
                    "source_id": best["source_id"],
                    "destination_id": best["destination_id"],
                    "cargo": best["cargo"],
                    "mode": best["mode"],
                }
            if loan < max_loan:
                return {
                    "type": "take_loan",
                    "amount": min(max_loan - loan, best["estimated_build_cost"] - cash + 10_000),
                }

        if cash > 80_000 and loan > 0:
            return {"type": "repay_loan", "amount": min(loan, cash - 50_000)}
            return {"type": "wait", "months": 3}

    def _frontier_action(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        candidates = observation.get("candidate_actions") or []
        for kind in ("add_vehicle", "build_route", "take_loan", "repay_loan", "wait"):
            for candidate in candidates:
                if candidate.get("kind") != kind:
                    continue
                if not candidate.get("directly_executable"):
                    continue
                if kind == "build_route" and float(candidate.get("estimates", {}).get("monthly_profit", 0) or 0) <= 0:
                    continue
                if kind == "wait" and observation.get("routes"):
                    return dict(candidate["action"])
                if kind != "wait":
                    return dict(candidate["action"])
        return None

    def _route_needing_vehicle(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        routes = sorted(observation["routes"], key=lambda route: route["vehicles"])
        for route in routes:
            if route["vehicles"] < 2:
                return route
        return None

    def _best_route_candidate(self, observation: dict[str, Any]) -> dict[str, Any] | None:
        if len(observation["routes"]) >= observation["scenario"]["goals"].get("network_routes", 1):
            return None
        existing = {
            (route["source_id"], route["destination_id"], route["cargo"])
            for route in observation["routes"]
        }
        goal_cargo = observation["scenario"]["goals"].get("cargo")
        terrain_cost = observation["scenario"]["map"].get("terrain_cost", 1.0)
        best = None
        for source in observation["nodes"]:
            for cargo, produced in source.get("produces", {}).items():
                if goal_cargo and cargo != goal_cargo:
                    continue
                for destination in observation["nodes"]:
                    if destination["id"] == source["id"] or cargo not in destination.get("accepts", {}):
                        continue
                    if (source["id"], destination["id"], cargo) in existing:
                        continue
                    accepted = destination["accepts"][cargo]
                    distance = abs(source["x"] - destination["x"]) + abs(source["y"] - destination["y"])
                    for mode in ("rail", "road"):
                        cfg = MODE_CONFIG[mode]
                        build_cost = cfg["station_cost"] * 2 + cfg["track_cost"] * distance * terrain_cost
                        capacity = cfg["vehicle_capacity"] * cfg["speed_factor"]
                        delivered = min(produced, accepted, capacity * max(0.35, 1.0 - distance / 240.0))
                        revenue = delivered * CARGO_REVENUE.get(cargo, 9.0) * (1.0 + distance / 120.0)
                        operating_cost = cfg["maintenance"] + distance * 8.0
                        monthly_profit = revenue - operating_cost
                        roi = monthly_profit / max(1.0, build_cost + cfg["vehicle_cost"])
                        candidate = {
                            "source_id": source["id"],
                            "destination_id": destination["id"],
                            "cargo": cargo,
                            "mode": mode,
                            "estimated_build_cost": build_cost,
                            "estimated_vehicle_cost": cfg["vehicle_cost"],
                            "estimated_monthly_profit": monthly_profit,
                            "roi": roi,
                        }
                        if monthly_profit > 0 and (best is None or candidate["roi"] > best["roi"]):
                            best = candidate
        return best
