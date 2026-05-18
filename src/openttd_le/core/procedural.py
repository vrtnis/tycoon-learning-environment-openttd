from __future__ import annotations

import random
from typing import Iterable

from openttd_le.core.types import Budget, Goals, MapConfig, Node, Scenario


PROCEDURAL_FAMILIES = ("single_route", "low_cash", "multi_route", "chain")
PROCEDURAL_SPLITS = ("train", "dev", "test")
SPLIT_BASE_SEEDS = {"train": 10_000, "dev": 20_000, "test": 30_000}
DEFAULT_PROCEDURAL_COUNT_PER_FAMILY = 3

_CARGO_KINDS = (
    ("coal", "Coal Mine", "Power Plant", 95, 145),
    ("wood", "Forest", "Sawmill", 70, 110),
    ("oil", "Oil Wells", "Refinery", 85, 140),
    ("grain", "Farm", "Food Plant", 65, 115),
    ("goods", "Factory", "Town", 55, 95),
    ("passengers", "Town", "Town", 55, 120),
    ("mail", "Town", "Town", 22, 46),
)

_CHAIN_KINDS = (
    ("coal", "steel", "Coal Mine", "Steel Mill", "Factory"),
    ("wood", "goods", "Forest", "Sawmill", "Town"),
    ("grain", "goods", "Farm", "Food Plant", "Town"),
    ("oil", "goods", "Oil Wells", "Refinery", "Town"),
)


def generate_procedural_scenarios(
    *,
    split: str = "dev",
    families: Iterable[str] = PROCEDURAL_FAMILIES,
    count_per_family: int = DEFAULT_PROCEDURAL_COUNT_PER_FAMILY,
) -> list[Scenario]:
    if split not in PROCEDURAL_SPLITS:
        raise ValueError(f"Unknown procedural split: {split}")
    scenarios: list[Scenario] = []
    for family_index, family in enumerate(families):
        if family not in PROCEDURAL_FAMILIES:
            raise ValueError(f"Unknown procedural family: {family}")
        for ordinal in range(1, count_per_family + 1):
            seed = SPLIT_BASE_SEEDS[split] + family_index * 1000 + ordinal
            scenarios.append(_generate_family(family, split, ordinal, seed))
    return scenarios


def procedural_task_ids(
    *,
    split: str = "dev",
    families: Iterable[str] = PROCEDURAL_FAMILIES,
    count_per_family: int = DEFAULT_PROCEDURAL_COUNT_PER_FAMILY,
) -> list[str]:
    return [scenario.id for scenario in generate_procedural_scenarios(split=split, families=families, count_per_family=count_per_family)]


def _generate_family(family: str, split: str, ordinal: int, seed: int) -> Scenario:
    rng = random.Random(seed)
    if family == "single_route":
        return _single_route(split, ordinal, seed, rng)
    if family == "low_cash":
        return _low_cash(split, ordinal, seed, rng)
    if family == "multi_route":
        return _multi_route(split, ordinal, seed, rng)
    if family == "chain":
        return _chain(split, ordinal, seed, rng)
    raise AssertionError(family)


def _single_route(split: str, ordinal: int, seed: int, rng: random.Random) -> Scenario:
    cargo, source_kind, destination_kind, low, high = rng.choice(_CARGO_KINDS[:5])
    source, destination = _route_nodes(rng, cargo, source_kind, destination_kind, low, high)
    distance = _distance(source, destination)
    return Scenario(
        id=_scenario_id(split, "single_route", ordinal),
        name=f"Procedural Single Route {split}-{ordinal:03d}",
        task=f"Build and operate the best {cargo} route on a generated map.",
        map=MapConfig(width=80, height=80, terrain_cost=round(rng.uniform(0.9, 1.25), 2)),
        budget=Budget(max_steps=14, max_months=42, starting_cash=170_000, max_loan=320_000, interest_rate=0.045),
        goals=Goals(cargo=cargo, cargo_delivered=max(500.0, distance * 18.0), operating_profit=max(35_000.0, distance * 1100.0), network_routes=1, max_debt_ratio=0.72),
        nodes=[source, destination, _town(rng, "distractor_town", "Riverton")],
        tags=_tags(split, "single_route", seed, cargo),
    )


def _low_cash(split: str, ordinal: int, seed: int, rng: random.Random) -> Scenario:
    cargo, source_kind, destination_kind, low, high = rng.choice(_CARGO_KINDS[:4])
    source, destination = _route_nodes(rng, cargo, source_kind, destination_kind, low, high)
    return Scenario(
        id=_scenario_id(split, "low_cash", ordinal),
        name=f"Procedural Low Cash {split}-{ordinal:03d}",
        task="Use financing carefully to bootstrap a profitable generated route.",
        map=MapConfig(width=72, height=72, terrain_cost=round(rng.uniform(1.0, 1.35), 2)),
        budget=Budget(max_steps=16, max_months=46, starting_cash=rng.randint(55_000, 80_000), max_loan=300_000, interest_rate=0.06),
        goals=Goals(cargo=cargo, cargo_delivered=650.0, operating_profit=42_000.0, network_routes=1, max_debt_ratio=0.65),
        nodes=[source, destination, _town(rng, "support_town", "Lowbank")],
        tags=_tags(split, "low_cash", seed, cargo),
    )


def _multi_route(split: str, ordinal: int, seed: int, rng: random.Random) -> Scenario:
    first = rng.choice(_CARGO_KINDS[:5])
    second = rng.choice([item for item in _CARGO_KINDS if item[0] != first[0]])
    source_a, destination_a = _route_nodes(rng, first[0], first[1], first[2], first[3], first[4], prefix="a")
    source_b, destination_b = _route_nodes(rng, second[0], second[1], second[2], second[3], second[4], prefix="b")
    return Scenario(
        id=_scenario_id(split, "multi_route", ordinal),
        name=f"Procedural Multi Route {split}-{ordinal:03d}",
        task="Build a two-route network from mixed generated cargo opportunities.",
        map=MapConfig(width=96, height=80, terrain_cost=round(rng.uniform(0.9, 1.2), 2)),
        budget=Budget(max_steps=22, max_months=54, starting_cash=240_000, max_loan=460_000, interest_rate=0.05),
        goals=Goals(cargo=None, cargo_delivered=1300.0, operating_profit=95_000.0, network_routes=2, max_debt_ratio=0.75),
        nodes=[source_a, destination_a, source_b, destination_b],
        tags=_tags(split, "multi_route", seed, first[0], second[0]),
    )


def _chain(split: str, ordinal: int, seed: int, rng: random.Random) -> Scenario:
    raw, processed, source_kind, processor_kind, sink_kind = rng.choice(_CHAIN_KINDS)
    source = _industry(rng, "chain_source", source_kind, produces={raw: rng.randint(85, 135)})
    processor = _industry(
        rng,
        "chain_processor",
        processor_kind,
        produces={processed: rng.randint(55, 95)},
        accepts={raw: rng.randint(90, 145)},
    )
    sink_accepts = {processed: rng.randint(60, 110)}
    sink = _town(rng, "chain_sink", "Market", accepts=sink_accepts) if sink_kind == "Town" else _industry(rng, "chain_sink", sink_kind, accepts=sink_accepts)
    return Scenario(
        id=_scenario_id(split, "chain", ordinal),
        name=f"Procedural Chain {split}-{ordinal:03d}",
        task=f"Build a generated two-stage {raw} to {processed} logistics chain.",
        map=MapConfig(width=90, height=90, terrain_cost=round(rng.uniform(0.95, 1.3), 2)),
        budget=Budget(max_steps=24, max_months=60, starting_cash=260_000, max_loan=520_000, interest_rate=0.052),
        goals=Goals(cargo=None, cargo_delivered=1200.0, operating_profit=100_000.0, network_routes=2, max_debt_ratio=0.78),
        nodes=[source, processor, sink, _town(rng, "chain_distractor", "Cedar")],
        tags=_tags(split, "chain", seed, raw, processed),
    )


def _route_nodes(
    rng: random.Random,
    cargo: str,
    source_kind: str,
    destination_kind: str,
    low: int,
    high: int,
    *,
    prefix: str = "main",
) -> tuple[Node, Node]:
    source_produces = {cargo: rng.randint(low, high)}
    destination_accepts = {cargo: rng.randint(low, high)}
    source = _town(rng, f"{prefix}_source", source_kind, produces=source_produces) if source_kind == "Town" else _industry(rng, f"{prefix}_source", source_kind, produces=source_produces)
    destination = _town(rng, f"{prefix}_destination", destination_kind, accepts=destination_accepts) if destination_kind == "Town" else _industry(rng, f"{prefix}_destination", destination_kind, accepts=destination_accepts)
    return source, destination


def _industry(
    rng: random.Random,
    node_id: str,
    kind: str,
    *,
    produces: dict[str, float] | None = None,
    accepts: dict[str, float] | None = None,
) -> Node:
    return Node(
        id=node_id,
        name=f"{_name_prefix(rng)} {kind}",
        kind="industry",
        x=rng.randint(6, 84),
        y=rng.randint(6, 84),
        produces=produces or {},
        accepts=accepts or {},
    )


def _town(
    rng: random.Random,
    node_id: str,
    name: str,
    *,
    produces: dict[str, float] | None = None,
    accepts: dict[str, float] | None = None,
) -> Node:
    population = rng.randint(650, 2400)
    default_produces = {"passengers": max(35, population // 22), "mail": max(10, population // 70)}
    default_accepts = {"passengers": max(35, population // 20), "mail": max(10, population // 65)}
    return Node(
        id=node_id,
        name=f"{_name_prefix(rng)} {name}",
        kind="town",
        x=rng.randint(6, 84),
        y=rng.randint(6, 84),
        produces=produces if produces is not None else default_produces,
        accepts=accepts if accepts is not None else default_accepts,
        population=population,
    )


def _scenario_id(split: str, family: str, ordinal: int) -> str:
    return f"proc_{split}_{family}_{ordinal:03d}"


def _tags(split: str, family: str, seed: int, *labels: str) -> list[str]:
    return ["procedural", f"split:{split}", f"family:{family}", f"seed:{seed}", *labels]


def _name_prefix(rng: random.Random) -> str:
    return rng.choice(("North", "South", "East", "West", "Lake", "Fort", "New", "Old", "Grand", "Little"))


def _distance(a: Node, b: Node) -> float:
    return float(abs(a.x - b.x) + abs(a.y - b.y))
