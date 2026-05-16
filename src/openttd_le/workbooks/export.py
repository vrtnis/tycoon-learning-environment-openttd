from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openttd_le.workbooks.template import read_firs_ops_workbook
from openttd_le.workbooks.xlsx import Sheet, write_xlsx


def export_run_to_xlsx(
    run_dir: str | Path,
    out: str | Path,
    *,
    source_workbook: str | Path | None = None,
) -> Path:
    run_path = Path(run_dir)
    summary = _read_json(run_path / "summary.json")
    events = _read_events(run_path)
    workbook_meta: dict[str, Any] = {}
    if source_workbook:
        try:
            _, workbook_meta = read_firs_ops_workbook(source_workbook)
        except Exception as exc:
            workbook_meta = {"workbook_error": str(exc)}

    observations = [event for event in events if event.get("event") in {"initial_observation", "observation"}]
    final_observation = observations[-1].get("data", {}) if observations else summary.get("final_observation", {})
    actions = [event for event in events if event.get("event") == "action"]
    routes = final_observation.get("routes", []) or summary.get("routes", [])
    finances = final_observation.get("company_finances", {})
    bottlenecks = _bottleneck_rows(final_observation)
    sheets = [
        Sheet("Scenario", _scenario_rows(summary, workbook_meta), widths={1: 24, 2: 48, 3: 64}),
        Sheet("Objectives", _objectives_rows(workbook_meta), widths={1: 12, 2: 24, 3: 24, 4: 14, 5: 16, 6: 18, 7: 12}),
        Sheet("Plan", _actions_rows(actions), widths={1: 10, 2: 22, 3: 80}),
        Sheet("Routes", _routes_rows(routes), widths={1: 14, 2: 12, 3: 24, 4: 24, 5: 12, 6: 12, 7: 14, 8: 16}),
        Sheet("Financials", _financial_rows(summary, finances), widths={1: 24, 2: 18, 3: 18, 4: 18, 5: 18}),
        Sheet("Bottlenecks", bottlenecks, widths={1: 14, 2: 12, 3: 18, 4: 18, 5: 18, 6: 28}),
        Sheet("Scorecard", _scorecard_rows(summary, routes), widths={1: 28, 2: 18, 3: 18, 4: 18}),
    ]
    return write_xlsx(out, sheets)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _read_events(run_dir: Path) -> list[dict[str, Any]]:
    trace_paths = [run_dir / "firs_trace.jsonl", run_dir / "coal_trace.jsonl", run_dir / "live_trace.jsonl"]
    for path in trace_paths:
        if not path.exists():
            continue
        events = []
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    events.append(json.loads(line))
        return events
    return []


def _scenario_rows(summary: dict[str, Any], workbook_meta: dict[str, Any]) -> list[list[Any]]:
    fields = workbook_meta.get("fields", {})
    rows = [
        ["Scenario"],
        ["Run inputs and outputs"],
        ["Field", "Value", "Notes"],
    ]
    for key in ["seed", "economy", "budget", "years", "allowed_modes", "map_x", "map_y", "landscape"]:
        rows.append([key, fields.get(key, summary.get(key, "")), ""])
    rows.extend(
        [
            ["run_dir", summary.get("run_dir", ""), ""],
            ["model", summary.get("model", ""), ""],
            ["completed", summary.get("completed", False), ""],
        ]
    )
    if workbook_meta.get("workbook_error"):
        rows.append(["source_workbook_error", workbook_meta["workbook_error"], ""])
    return rows


def _objectives_rows(workbook_meta: dict[str, Any]) -> list[list[Any]]:
    rows = [
        ["Objectives"],
        ["Workbook objective rows"],
        ["Step", "Source Type", "Destination Type", "Cargo Label", "Deadline Year", "Required Delivered", "Weight"],
    ]
    for item in workbook_meta.get("objectives", []):
        rows.append(
            [
                item.get("step"),
                item.get("source_type"),
                item.get("destination_type"),
                item.get("cargo"),
                item.get("deadline_year"),
                item.get("required_delivered"),
                item.get("weight"),
            ]
        )
    return rows


def _actions_rows(actions: list[dict[str, Any]]) -> list[list[Any]]:
    rows = [["Plan"], ["GPT action trace"], ["Step", "Action Type", "Action JSON"]]
    for event in actions:
        action = event.get("data", {})
        rows.append([event.get("step", ""), action.get("type", ""), json.dumps(action, separators=(",", ":"))])
    return rows


def _routes_rows(routes: list[dict[str, Any]]) -> list[list[Any]]:
    rows = [
        ["Routes"],
        ["Actual route registry from GameScript"],
        ["Route ID", "Cargo", "Source", "Destination", "Vehicles", "Delivered", "Profit", "Waiting Source"],
    ]
    for route in routes:
        rows.append(
            [
                route.get("route_id", route.get("id", "")),
                route.get("cargo_label", route.get("cargo", "")),
                route.get("source_name", route.get("source", "")),
                route.get("destination_name", route.get("destination", "")),
                route.get("vehicles", 0),
                route.get("delivered", 0),
                route.get("profit", route.get("vehicle_profit", 0)),
                route.get("source_waiting", route.get("waiting_source", 0)),
            ]
        )
    return rows


def _financial_rows(summary: dict[str, Any], finances: dict[str, Any]) -> list[list[Any]]:
    return [
        ["Financials"],
        ["Final financial snapshot"],
        ["Metric", "Value", "Notes"],
        ["bank_balance", finances.get("bank_balance", summary.get("final_bank_balance", "")), ""],
        ["loan", finances.get("loan", summary.get("loan", "")), ""],
        ["route_profit", summary.get("route_profit", ""), "Sum of route vehicle profit when available."],
        ["completed", summary.get("completed", False), ""],
    ]


def _bottleneck_rows(observation: dict[str, Any]) -> list[list[Any]]:
    rows = [
        ["Bottlenecks"],
        ["Waiting cargo and station ratings"],
        ["Route ID", "Cargo", "Waiting Cargo", "Station Rating", "Idle Vehicles", "Action"],
    ]
    for route in observation.get("routes", []):
        waiting = route.get("source_waiting", 0)
        rating = route.get("source_rating", "")
        action = "add_vehicles" if isinstance(waiting, (int, float)) and waiting > 50 else "monitor"
        rows.append(
            [
                route.get("route_id", ""),
                route.get("cargo_label", ""),
                waiting,
                rating,
                _idle_count(route.get("vehicle_details", [])),
                action,
            ]
        )
    return rows


def _scorecard_rows(summary: dict[str, Any], routes: list[dict[str, Any]]) -> list[list[Any]]:
    delivered_routes = sum(1 for route in routes if route.get("delivered", 0) > 0)
    profit = sum(float(route.get("profit", route.get("vehicle_profit", 0)) or 0) for route in routes)
    completed = bool(summary.get("completed")) or (delivered_routes >= 2 and profit > 0)
    score = min(100, delivered_routes * 35 + (30 if profit > 0 else 0))
    return [
        ["Scorecard"],
        ["Actual vs target"],
        ["Metric", "Value", "Target", "Status"],
        ["Routes with delivery", delivered_routes, 2, "PASS" if delivered_routes >= 2 else "OPEN"],
        ["Positive route profit", round(profit, 2), "> 0", "PASS" if profit > 0 else "OPEN"],
        ["Completed", completed, True, "PASS" if completed else "OPEN"],
        ["Total score", score, 100, ""],
        ["Notes", summary.get("note", ""), "", ""],
    ]


def _idle_count(vehicles: list[dict[str, Any]]) -> int:
    return sum(1 for vehicle in vehicles if vehicle.get("speed", 0) == 0 or vehicle.get("in_depot"))
