from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from .types import Budget, Goals, MapConfig, Node, Scenario


DEFAULT_SCENARIO_FILE = Path(__file__).resolve().parents[3] / "scenarios" / "lab_play.json"


class ScenarioRegistry:
    def __init__(self, scenarios: Iterable[Scenario]):
        self._scenarios = {scenario.id: scenario for scenario in scenarios}

    def get(self, scenario_id: str) -> Scenario:
        try:
            return self._scenarios[scenario_id]
        except KeyError as exc:
            known = ", ".join(sorted(self._scenarios))
            raise KeyError(f"Unknown scenario '{scenario_id}'. Known scenarios: {known}") from exc

    def list(self) -> list[Scenario]:
        return [self._scenarios[key] for key in sorted(self._scenarios)]


def load_registry(path: str | Path | None = None) -> ScenarioRegistry:
    scenario_file = Path(path) if path else DEFAULT_SCENARIO_FILE
    with scenario_file.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    scenarios = [_parse_scenario(item) for item in payload["scenarios"]]
    return ScenarioRegistry(scenarios)


def _parse_scenario(payload: dict) -> Scenario:
    return Scenario(
        id=payload["id"],
        name=payload["name"],
        task=payload["task"],
        map=MapConfig(**payload["map"]),
        budget=Budget(**payload["budget"]),
        goals=Goals(**payload.get("goals", {})),
        nodes=[
            Node(
                id=item["id"],
                name=item["name"],
                kind=item["kind"],
                x=item["x"],
                y=item["y"],
                produces=dict(item.get("produces", {})),
                accepts=dict(item.get("accepts", {})),
                population=int(item.get("population", 0)),
            )
            for item in payload["nodes"]
        ],
        tags=list(payload.get("tags", [])),
    )
