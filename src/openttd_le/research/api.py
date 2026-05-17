from __future__ import annotations

from dataclasses import dataclass
from typing import Any


class Prototype:
    class Cargo:
        Alcohol = "BEER"
        Chemicals = "CHEM"
        Coal = "COAL"
        EngineeringSupplies = "ENSP"
        FarmSupplies = "FMSP"
        Food = "FOOD"
        Goods = "GOOD"
        IronOre = "IORE"
        Steel = "STEL"
        Wood = "WOOD"

    class Industry:
        CoalMine = "Coal Mine"
        DairyFarm = "Dairy Farm"
        GeneralStore = "General Store"
        IronOreMine = "Iron Ore Mine"
        Port = "Port"
        SteelMill = "Steel Mill"


@dataclass(frozen=True)
class Cargo:
    id: int
    label: str
    name: str
    value: float = 1.0


@dataclass(frozen=True)
class Industry:
    id: int
    name: str
    x: int | None = None
    y: int | None = None

    @property
    def type(self) -> str:
        for suffix in (
            "Coal Mine",
            "Iron Ore Mine",
            "Steel Mill",
            "Port",
            "General Store",
            "Dairy Farm",
            "Glass Works",
            "Dredging Site",
            "Clay Pit",
        ):
            if suffix.lower() in self.name.lower():
                return suffix
        return self.name


@dataclass(frozen=True)
class CargoChain:
    source: Industry
    destination: Industry
    cargo: Cargo
    distance: int | None = None
    production: int | None = None

    @property
    def route_key(self) -> tuple[int, int, int]:
        return self.source.id, self.destination.id, self.cargo.id


@dataclass(frozen=True)
class Route:
    id: str
    source: Industry
    destination: Industry
    cargo: Cargo
    vehicles: int
    delivered: int
    profit: float
    source_waiting: int = 0
    source_rating: int | None = None


@dataclass(frozen=True)
class Finance:
    bank_balance: float = 0.0
    loan: float = 0.0
    route_profit: float = 0.0


def api_from_observation(observation: dict[str, Any], cargo_values: dict[str, float] | None = None) -> dict[str, Any]:
    values = cargo_values or {}
    return {
        "industries": get_industries(observation),
        "cargo_chains": get_cargo_chains(observation, values),
        "routes": get_routes(observation, values),
        "finance": get_finance(observation),
    }


def get_industries(observation: dict[str, Any]) -> list[Industry]:
    by_id: dict[int, Industry] = {}
    for field in ("industry_graph", "industry_inputs", "industry_outputs"):
        for item in observation.get(field, []) or []:
            industry_id = item.get("industry_id", item.get("source_id"))
            industry_name = item.get("industry_name", item.get("source_name"))
            if isinstance(industry_id, int) and industry_name:
                by_id[industry_id] = Industry(
                    id=industry_id,
                    name=str(industry_name),
                    x=_maybe_int(item.get("source_x")),
                    y=_maybe_int(item.get("source_y")),
                )
            destination_id = item.get("destination_id")
            destination_name = item.get("destination_name")
            if isinstance(destination_id, int) and destination_name:
                by_id[destination_id] = Industry(
                    id=destination_id,
                    name=str(destination_name),
                    x=_maybe_int(item.get("destination_x")),
                    y=_maybe_int(item.get("destination_y")),
                )
    return sorted(by_id.values(), key=lambda industry: industry.id)


def get_cargo_chains(observation: dict[str, Any], cargo_values: dict[str, float] | None = None) -> list[CargoChain]:
    values = cargo_values or {}
    chains: list[CargoChain] = []
    for item in observation.get("industry_graph", []) or []:
        source_id = item.get("source_id")
        destination_id = item.get("destination_id")
        cargo_id = item.get("cargo_id")
        if not isinstance(source_id, int) or not isinstance(destination_id, int) or not isinstance(cargo_id, int):
            continue
        cargo_label = str(item.get("cargo_label", item.get("cargo", ""))).upper()
        chains.append(
            CargoChain(
                source=Industry(source_id, str(item.get("source_name", item.get("source_type", ""))), _maybe_int(item.get("source_x")), _maybe_int(item.get("source_y"))),
                destination=Industry(destination_id, str(item.get("destination_name", item.get("destination_type", ""))), _maybe_int(item.get("destination_x")), _maybe_int(item.get("destination_y"))),
                cargo=Cargo(cargo_id, cargo_label, str(item.get("cargo_name", cargo_label)), float(values.get(cargo_label, 1.0))),
                distance=_maybe_int(item.get("distance")),
                production=_maybe_int(item.get("production")),
            )
        )
    return chains


def get_routes(observation: dict[str, Any], cargo_values: dict[str, float] | None = None) -> list[Route]:
    values = cargo_values or {}
    routes: list[Route] = []
    for item in observation.get("routes", []) or []:
        cargo_label = str(item.get("cargo_label", item.get("cargo", ""))).upper()
        cargo_id = _maybe_int(item.get("cargo_id")) or -1
        routes.append(
            Route(
                id=str(item.get("route_id", item.get("id", ""))),
                source=Industry(_maybe_int(item.get("source_id")) or -1, str(item.get("source_name", item.get("source", "")))),
                destination=Industry(_maybe_int(item.get("destination_id")) or -1, str(item.get("destination_name", item.get("destination", "")))),
                cargo=Cargo(cargo_id, cargo_label, str(item.get("cargo_name", cargo_label)), float(values.get(cargo_label, 1.0))),
                vehicles=_maybe_int(item.get("vehicles")) or 0,
                delivered=_maybe_int(item.get("delivered")) or 0,
                profit=float(item.get("profit", item.get("vehicle_profit", 0)) or 0),
                source_waiting=_maybe_int(item.get("source_waiting", item.get("waiting_source", 0))) or 0,
                source_rating=_maybe_int(item.get("source_rating")),
            )
        )
    return routes


def get_finance(observation: dict[str, Any]) -> Finance:
    finances = observation.get("company_finances", {}) or {}
    routes = observation.get("routes", []) or []
    return Finance(
        bank_balance=float(finances.get("bank_balance", observation.get("bank_balance", 0)) or 0),
        loan=float(finances.get("loan", 0) or 0),
        route_profit=sum(float(route.get("profit", route.get("vehicle_profit", 0)) or 0) for route in routes),
    )


def _maybe_int(value: Any) -> int | None:
    try:
        if value is None or value == "":
            return None
        return int(value)
    except (TypeError, ValueError):
        return None
