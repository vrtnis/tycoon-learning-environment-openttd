from __future__ import annotations

from pathlib import Path
from typing import Any

from openttd_le.backends.firs import FIRSRunConfig, config_from_workbook_fields
from openttd_le.workbooks.xlsx import Sheet, read_xlsx, write_xlsx


def create_firs_ops_workbook(path: str | Path, config: FIRSRunConfig | None = None) -> Path:
    cfg = config or FIRSRunConfig()
    sheets = [
        Sheet("Scenario", _scenario_rows(cfg), widths={1: 24, 2: 24, 3: 64}),
        Sheet("Objectives", _objective_rows(cfg), widths={1: 12, 2: 22, 3: 24, 4: 14, 5: 16, 6: 18, 7: 12}),
        Sheet("Plan", _plan_rows(cfg), widths={1: 12, 2: 22, 3: 22, 4: 22, 5: 14, 6: 12, 7: 16, 8: 18, 9: 18}),
        Sheet("Routes", [["Routes"], [], ["Route ID", "Cargo", "Source", "Destination", "Vehicles", "Delivered", "Profit", "Waiting Source"]]),
        Sheet("Financials", [["Financials"], [], ["Year", "Revenue", "Opex", "Capex", "Loan", "Bank Balance", "Profit"]]),
        Sheet("Bottlenecks", [["Bottlenecks"], [], ["Route ID", "Cargo", "Waiting Cargo", "Station Rating", "Idle Vehicles", "Action"]]),
        Sheet("Scorecard", _scorecard_rows()),
    ]
    return write_xlsx(path, sheets)


def read_firs_ops_workbook(path: str | Path) -> tuple[FIRSRunConfig, dict[str, Any]]:
    workbook = read_xlsx(path)
    scenario_sheet = workbook.get("Scenario")
    if scenario_sheet is None:
        raise ValueError("Workbook is missing required sheet: Scenario")
    fields = _scenario_fields(scenario_sheet["rows"])
    objectives = _objective_records(workbook.get("Objectives", {}).get("rows", []))
    return config_from_workbook_fields(fields, objectives), {"fields": fields, "objectives": objectives}


def _scenario_rows(config: FIRSRunConfig) -> list[list[Any]]:
    return [
        ["Scenario"],
        ["Excel workbook -> GPT plan -> OpenTTD/FIRS execution -> updated Excel scorecard"],
        ["Field", "Value", "Notes"],
        ["seed", config.seed, "OpenTTD generation_seed for reproducible maps."],
        ["economy", config.economy, "FIRS economy. First slice uses basic_temperate for Iron Ore Mine -> Steel Mill."],
        ["budget", config.budget, "Planning budget visible to GPT."],
        ["years", config.years, "Objective horizon."],
        ["allowed_modes", ",".join(config.allowed_modes), "Currently the GameScript macro builder uses road vehicles."],
        ["map_x", config.map_x, "OpenTTD map size exponent."],
        ["map_y", config.map_y, "OpenTTD map size exponent."],
        ["landscape", config.landscape, "basic_temperate works best with temperate."],
        ["starting_year", config.starting_year, "Scenario start year."],
        ["towns", config.towns, "Town count setting."],
        ["industries", config.industries, "Industry density setting."],
        ["vehicles_per_route", config.vehicles_per_route, "Default truck count when GPT builds a route."],
    ]


def _objective_rows(config: FIRSRunConfig) -> list[list[Any]]:
    rows: list[list[Any]] = [
        ["Objectives"],
        ["Chain targets and weights"],
        ["Step", "Source Type", "Destination Type", "Cargo Label", "Deadline Year", "Required Delivered", "Weight"],
    ]
    for target in config.target_chain:
        rows.append(
            [
                target.get("step", len(rows) - 2),
                target.get("source_type", ""),
                target.get("destination_type", ""),
                target.get("cargo", ""),
                target.get("deadline_year", config.years),
                target.get("required_delivered", 1),
                40 if int(target.get("step", 1)) == 2 else 30,
            ]
        )
    rows.append(["Financial", "Total Network", "Positive Route Profit", "PROFIT", config.years, 1, 30])
    return rows


def _plan_rows(config: FIRSRunConfig) -> list[list[Any]]:
    rows = [
        ["Plan"],
        ["GPT can overwrite these rows before execution; actuals land in Routes and Financials."],
        ["Step", "Action", "Source Type", "Destination Type", "Cargo", "Vehicles", "Expected Cost", "Expected Revenue", "Status"],
    ]
    for target in config.target_chain:
        rows.append(
            [
                target.get("step", len(rows) - 2),
                "build_cargo_route",
                target.get("source_type", ""),
                target.get("destination_type", ""),
                target.get("cargo", ""),
                config.vehicles_per_route,
                "",
                "",
                "planned",
            ]
        )
    rows.append([len(config.target_chain) + 1, "inspect_bottlenecks", "", "", "", "", "", "", "planned"])
    return rows


def _scorecard_rows() -> list[list[Any]]:
    return [
        ["Scorecard"],
        ["Run summary is populated by tycoonle-openttd export-xlsx or play-firs-live."],
        ["Metric", "Value", "Target", "Status"],
        ["First route delivered", "", 1, ""],
        ["Second route delivered", "", 1, ""],
        ["Positive route profit", "", 1, ""],
        ["Total score", "", 100, ""],
        ["Notes", "", "", ""],
    ]


def _scenario_fields(rows: list[list[Any]]) -> dict[str, Any]:
    fields: dict[str, Any] = {}
    for row in rows[3:]:
        if len(row) < 2 or row[0] in (None, ""):
            continue
        fields[str(row[0])] = row[1]
    return fields


def _objective_records(rows: list[list[Any]]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows[3:]:
        if len(row) < 6 or not row[0]:
            continue
        if str(row[0]).lower() == "financial":
            continue
        records.append(
            {
                "step": row[0],
                "source_type": row[1],
                "destination_type": row[2],
                "cargo": row[3],
                "deadline_year": row[4],
                "required_delivered": row[5],
                "weight": row[6] if len(row) > 6 else None,
            }
        )
    return records
