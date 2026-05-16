from __future__ import annotations

from typing import Any

from .types import ActionType, EnvError


ACTION_TYPES: set[ActionType] = {
    "build_route",
    "add_vehicle",
    "wait",
    "take_loan",
    "repay_loan",
}


def normalize_action(action: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(action, dict):
        raise EnvError("Action must be a JSON object.")
    action_type = action.get("type")
    if action_type not in ACTION_TYPES:
        raise EnvError(f"Unknown action type: {action_type!r}.")
    if action_type == "wait":
        months = int(action.get("months", 1))
        return {"type": "wait", "months": max(1, min(12, months))}
    if action_type in {"take_loan", "repay_loan"}:
        amount = float(action.get("amount", 0))
        if amount <= 0:
            raise EnvError(f"{action_type} requires a positive amount.")
        return {"type": action_type, "amount": amount}
    if action_type == "build_route":
        required = ["source_id", "destination_id", "cargo", "mode"]
        missing = [key for key in required if not action.get(key)]
        if missing:
            raise EnvError(f"build_route missing fields: {', '.join(missing)}")
        mode = str(action["mode"])
        if mode not in {"road", "rail"}:
            raise EnvError("build_route mode must be 'road' or 'rail'.")
        return {
            "type": "build_route",
            "source_id": str(action["source_id"]),
            "destination_id": str(action["destination_id"]),
            "cargo": str(action["cargo"]),
            "mode": mode,
        }
    if action_type == "add_vehicle":
        route_id = action.get("route_id")
        if not route_id:
            raise EnvError("add_vehicle requires route_id.")
        count = int(action.get("count", 1))
        return {"type": "add_vehicle", "route_id": str(route_id), "count": max(1, min(20, count))}
    raise EnvError(f"Unhandled action type: {action_type!r}.")
