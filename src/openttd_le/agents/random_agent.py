from __future__ import annotations

import random
from typing import Any

from .base import Agent


class RandomAgent(Agent):
    name = "random"

    def __init__(self, seed: int | None = None) -> None:
        self.random = random.Random(seed)

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        routes = observation["routes"]
        cash = observation["company"]["cash"]
        if routes and self.random.random() < 0.35:
            return {
                "type": "add_vehicle",
                "route_id": self.random.choice(routes)["id"],
                "count": 1,
            }
        if cash < 40_000 and self.random.random() < 0.4:
            return {"type": "take_loan", "amount": 50_000}
        candidates = _route_candidates(observation)
        if candidates and self.random.random() < 0.65:
            candidate = self.random.choice(candidates)
            return {
                "type": "build_route",
                "source_id": candidate["source_id"],
                "destination_id": candidate["destination_id"],
                "cargo": candidate["cargo"],
                "mode": self.random.choice(["road", "rail"]),
            }
        return {"type": "wait", "months": self.random.randint(1, 4)}


def _route_candidates(observation: dict[str, Any]) -> list[dict[str, str]]:
    routes = {
        (route["source_id"], route["destination_id"], route["cargo"])
        for route in observation["routes"]
    }
    candidates = []
    for source in observation["nodes"]:
        for cargo in source.get("produces", {}):
            for destination in observation["nodes"]:
                if destination["id"] == source["id"]:
                    continue
                if cargo not in destination.get("accepts", {}):
                    continue
                key = (source["id"], destination["id"], cargo)
                if key in routes:
                    continue
                candidates.append(
                    {
                        "source_id": source["id"],
                        "destination_id": destination["id"],
                        "cargo": cargo,
                    }
                )
    return candidates
