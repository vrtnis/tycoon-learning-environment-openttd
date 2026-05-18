from __future__ import annotations

import json
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRACE_NAMES = ("firs_trace.jsonl", "coal_trace.jsonl", "live_trace.jsonl")


def load_replay(path: str | Path) -> dict[str, Any]:
    replay_path = Path(path)
    return json.loads(replay_path.read_text(encoding="utf-8"))


def replay_actions(replay: dict[str, Any]) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for step in replay.get("steps", []) or []:
        for action in step.get("actions", []) or []:
            if isinstance(action, dict) and isinstance(action.get("type"), str):
                actions.append(dict(action))
    return actions


def export_replay(run_dir: str | Path, out: str | Path | None = None) -> Path:
    root = Path(run_dir)
    if not root.exists():
        raise FileNotFoundError(f"Run directory not found: {root}")
    trace_path = _find_trace(root)
    summary = _read_json(root / "summary.json")
    launch = _read_json(root / "launch.json")
    payload = _build_replay(root, trace_path, summary, launch)
    output_path = Path(out) if out else root / "replay.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def _build_replay(
    run_dir: Path,
    trace_path: Path,
    summary: dict[str, Any],
    launch: dict[str, Any],
) -> dict[str, Any]:
    steps: dict[int, dict[str, Any]] = defaultdict(lambda: {"actions": [], "results": [], "observations": []})
    initial_observation: dict[str, Any] | None = None
    events: list[dict[str, Any]] = []
    for event in _read_jsonl(trace_path):
        events.append({"event": event.get("event"), "step": event.get("step")})
        event_name = str(event.get("event", ""))
        data = event.get("data", {})
        step = _step_number(event)
        if event_name == "initial_observation":
            initial_observation = data if isinstance(data, dict) else None
            continue
        bucket = steps[step]
        bucket["step"] = step
        if event_name == "repl_program":
            bucket["program"] = data.get("code") if isinstance(data, dict) else data
        elif event_name == "repl_feedback":
            if isinstance(data, dict):
                bucket["stdout"] = data.get("stdout", "")
                bucket["stderr"] = data.get("stderr", "")
                bucket["repl_action_count"] = data.get("actions")
            else:
                bucket["feedback"] = data
        elif event_name == "action":
            bucket["actions"].append(data)
        elif event_name == "result":
            bucket["results"].append(data)
        elif event_name == "observation":
            bucket["observations"].append(data)
        else:
            bucket.setdefault("events", []).append({"event": event_name, "data": data})

    ordered_steps = [steps[key] for key in sorted(steps)]
    for item in ordered_steps:
        if item["observations"]:
            item["final_observation"] = item["observations"][-1]

    return {
        "schema": "openttd-le-replay-v1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_dir),
        "trace": str(trace_path),
        "scenario": {
            "objective": summary.get("objective"),
            "model": summary.get("model") or launch.get("model"),
            "seed": summary.get("seed"),
            "economy": summary.get("economy"),
            "workbook": summary.get("workbook"),
            "firs_newgrf": summary.get("firs_newgrf") or launch.get("firs_newgrf"),
            "openttd_user_dir": summary.get("openttd_user_dir") or launch.get("openttd_user_dir"),
        },
        "media": {
            "recording": launch.get("recording"),
            "timelapse": launch.get("timelapse"),
            "report": launch.get("report") or str(run_dir / "report.xlsx"),
        },
        "summary": summary,
        "initial_observation": initial_observation,
        "steps": ordered_steps,
        "event_index": events,
        "replay_notes": [
            "This file captures GPT-generated REPL programs, macro-actions, results, and observation checkpoints.",
            "Re-execution requires the same OpenTTD/FIRS setup, seed, config, and bridge version.",
            "Use action entries as the deterministic macro-action script; use program/stdout/stderr entries to inspect GPT reasoning.",
        ],
    }


def _find_trace(run_dir: Path) -> Path:
    for name in TRACE_NAMES:
        path = run_dir / name
        if path.exists():
            return path
    raise FileNotFoundError(f"No trace JSONL found under {run_dir}. Expected one of: {', '.join(TRACE_NAMES)}")


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        if isinstance(item, dict):
            rows.append(item)
    return rows


def _step_number(event: dict[str, Any]) -> int:
    try:
        return int(event.get("step") or 0)
    except (TypeError, ValueError):
        return 0
