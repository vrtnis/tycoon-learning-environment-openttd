from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any


VOLATILE_KEYS = {
    "admin_port",
    "client_pid",
    "created_at",
    "game_port",
    "last_scroll",
    "openttd_user_dir",
    "pid",
    "source_waiting",
    "destination_waiting",
    "recording",
    "run_dir",
    "server_pid",
    "station_ratings",
    "tick",
    "timelapse",
    "trace",
    "vehicle_details",
    "vehicle_profit",
}

VOLATILE_ROUTE_KEYS = {
    "depot_tile",
    "destination_station",
    "source_rating",
    "source_waiting",
    "source_station",
    "vehicle_details",
}

VOLATILE_RESULT_KEYS = {
    "depot_tile",
    "destination_station",
    "source_waiting",
    "destination_waiting",
    "source_station",
}

MONEY_KEYS = {
    "bank_balance",
    "profit",
    "route_profit",
    "vehicle_profit",
}
MONEY_QUANTUM = 100

DETERMINISM_CONTRACT = {
    "schema": "openttd-le-determinism-contract-v1",
    "allowed_runtime_differences": [
        "absolute local paths",
        "run directories",
        "process ids",
        "ephemeral ports",
        "timestamps",
    ],
    "strict_public_trace": [
        "encoded observations",
        "action masks",
        "candidate ordering",
        "selected action indices",
        "selected action payloads",
        "rewards",
        "terminated flags",
        "truncated flags",
        "route outcomes exposed through public info",
        "cargo delivered",
        "money and profit values exposed through public info",
    ],
}

RUNTIME_LOCK_VOLATILE_KEYS = {
    "admin_port",
    "benchmark_file",
    "cfg",
    "cfg_sha256",
    "company_ai_dir",
    "firs_newgrf",
    "game_port",
    "gamescript_dir",
    "opengfx_baseset",
    "openttd_executable",
    "openttd_user_dir",
    "process_cwd",
    "run_dir",
    "server_command",
    "workbook",
}

TRACE_RUNTIME_VOLATILE_KEYS = {
    "admin_port",
    "client_pid",
    "created_at",
    "game_port",
    "openttd_user_dir",
    "pid",
    "recording",
    "run_dir",
    "server_pid",
    "timelapse",
    "trace",
}


def normalize_gym_info(info: dict[str, Any]) -> dict[str, Any]:
    keep = {
        "actions",
        "action_mask",
        "candidate_actions",
        "candidate_planning",
        "deterministic",
        "invalid_action",
        "native_observation",
        "result",
        "reward_details",
        "selected_action",
        "selected_preview",
        "snapshot",
        "task",
    }
    payload = {key: value for key, value in info.items() if key in keep}
    if isinstance(payload.get("actions"), list):
        payload["actions"] = [
            {
                "action": item.get("action"),
                "result": item.get("result"),
            }
            for item in payload["actions"]
            if isinstance(item, dict)
        ]
    return normalize_value(payload)


def normalize_observation(observation: dict[str, Any], *, decision_step: int | None = None) -> dict[str, Any]:
    normalized = normalize_value(observation)
    if decision_step is not None:
        normalized["decision_step"] = int(decision_step)
    normalized.pop("tick", None)
    normalized.pop("reason", None)
    normalized.pop("last_scroll", None)
    return normalized


def normalize_candidate_actions(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [normalize_value(candidate) for candidate in candidates]


def normalize_result(result: dict[str, Any] | None) -> dict[str, Any] | None:
    if result is None:
        return None
    return normalize_value({key: value for key, value in result.items() if key not in VOLATILE_RESULT_KEYS})


def normalize_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return normalize_value(value.tolist())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if key in VOLATILE_KEYS:
                continue
            item = value[key]
            if key in MONEY_KEYS and isinstance(item, (int, float)):
                result[key] = _quantize_money(item)
                continue
            if key == "routes" and isinstance(item, list):
                result[key] = _normalize_routes(item)
            elif key == "towns" and isinstance(item, list):
                result[key] = _normalize_towns(item)
            elif key == "result" and isinstance(item, dict):
                result[key] = normalize_result(item)
            elif key == "native_observation" and isinstance(item, dict):
                result[key] = normalize_observation(item)
            else:
                result[key] = normalize_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [normalize_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def normalize_determinism_trace(value: Any, *, mode: str = "strict") -> Any:
    """Normalize a public Gym trace for deterministic rollout comparison.

    Strict mode removes only runtime-only fields. Semantic mode preserves the
    older smoke-test behavior for reports that intentionally ignore simulator
    internals such as station ids and waiting cargo.
    """

    if mode not in {"strict", "semantic"}:
        raise ValueError("mode must be 'strict' or 'semantic'")
    if mode == "semantic":
        return normalize_value(value)
    return _normalize_strict_value(value)


def normalize_runtime_lock(lock: dict[str, Any] | None) -> dict[str, Any]:
    """Return the comparable part of a runtime lock.

    The exact lock still records paths, ports and command lines for auditability.
    This normalized form keeps only values that must be identical across repeat
    runs for a fixed seed and action sequence.
    """

    if not lock:
        return {}
    normalized = {}
    for key in sorted(lock):
        if key in RUNTIME_LOCK_VOLATILE_KEYS:
            continue
        value = lock[key]
        if key.endswith("_path") or key.endswith("_dir"):
            continue
        normalized[key] = _normalize_strict_value(value)
    return normalized


def stable_json_sha256(value: Any) -> str:
    payload = json.dumps(_normalize_strict_value(value), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def determinism_contract() -> dict[str, Any]:
    return dict(DETERMINISM_CONTRACT)


def _quantize_money(value: int | float) -> int:
    return int(round(float(value) / MONEY_QUANTUM) * MONEY_QUANTUM)


def first_diff(left: Any, right: Any, *, path: str = "$") -> dict[str, Any] | None:
    if type(left) is not type(right):
        return {"path": path, "left": left, "right": right, "reason": "type_mismatch"}
    if isinstance(left, dict):
        left_keys = set(left)
        right_keys = set(right)
        if left_keys != right_keys:
            return {
                "path": path,
                "left": sorted(left_keys - right_keys),
                "right": sorted(right_keys - left_keys),
                "reason": "keys_mismatch",
            }
        for key in sorted(left):
            diff = first_diff(left[key], right[key], path=f"{path}.{key}")
            if diff is not None:
                return diff
        return None
    if isinstance(left, list):
        if len(left) != len(right):
            return {"path": path, "left": len(left), "right": len(right), "reason": "length_mismatch"}
        for index, (left_item, right_item) in enumerate(zip(left, right, strict=True)):
            diff = first_diff(left_item, right_item, path=f"{path}[{index}]")
            if diff is not None:
                return diff
        return None
    if left != right:
        return {"path": path, "left": left, "right": right, "reason": "value_mismatch"}
    return None


def _normalize_strict_value(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return _normalize_strict_value(value.tolist())
    if isinstance(value, Path):
        return "<path>"
    if isinstance(value, dict):
        result: dict[str, Any] = {}
        for key in sorted(value):
            if key in TRACE_RUNTIME_VOLATILE_KEYS:
                continue
            item = value[key]
            if isinstance(item, Path):
                result[key] = "<path>"
            else:
                result[key] = _normalize_strict_value(item)
        return result
    if isinstance(value, (list, tuple)):
        return [_normalize_strict_value(item) for item in value]
    if isinstance(value, float):
        return round(value, 6)
    return value


def file_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    file_path = Path(path)
    if not file_path.exists() or not file_path.is_file():
        return None
    digest = hashlib.sha256()
    with file_path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def directory_sha256(path: Path | str | None) -> str | None:
    if path is None:
        return None
    root = Path(path)
    if not root.exists() or not root.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in root.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(root).as_posix().encode("utf-8"))
        digest.update(b"\0")
        file_digest = file_sha256(file_path)
        if file_digest:
            digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def _normalize_routes(routes: list[Any]) -> list[Any]:
    normalized = []
    for route in routes:
        if not isinstance(route, dict):
            normalized.append(normalize_value(route))
            continue
        item = {key: value for key, value in route.items() if key not in VOLATILE_ROUTE_KEYS}
        normalized.append(normalize_value(item))
    return sorted(normalized, key=lambda route: str(route.get("route_id", "")) if isinstance(route, dict) else str(route))


def _normalize_towns(towns: list[Any]) -> list[Any]:
    normalized = []
    for town in towns:
        if not isinstance(town, dict):
            normalized.append(normalize_value(town))
            continue
        item = {
            key: value
            for key, value in town.items()
            if key not in {"population", "houses", "ratings"}
        }
        normalized.append(normalize_value(item))
    return sorted(normalized, key=lambda town: str(town.get("town_id", town.get("name", ""))) if isinstance(town, dict) else str(town))
