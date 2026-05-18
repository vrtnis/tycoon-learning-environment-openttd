from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openttd_le.core.schemas import EPISODE_TRACE_SCHEMA, REPLAY_SCHEMA, schema_manifest


class RunArtifacts:
    def __init__(self, root: Path, scenario_id: str, agent_name: str, seed: int) -> None:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        self.run_dir = root / f"{timestamp}_{scenario_id}_{agent_name}_seed{seed}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        (self.run_dir / "screenshots").mkdir(exist_ok=True)
        self.actions_path = self.run_dir / "actions.jsonl"
        self.episode_path = self.run_dir / "episode.jsonl"
        self.observations_path = self.run_dir / "observations.jsonl"
        self.candidate_actions_path = self.run_dir / "candidate_actions.jsonl"
        self.rewards_path = self.run_dir / "rewards.jsonl"
        self.diagnostics_path = self.run_dir / "diagnostics.jsonl"
        self.metrics_path = self.run_dir / "metrics.csv"
        self.trace_path = self.run_dir / "agent_trace.md"
        self._metrics_rows: list[dict[str, Any]] = []
        self._replay_steps: list[dict[str, Any]] = []

    def log_step(
        self,
        step: int,
        observation: dict[str, Any],
        action: dict[str, Any],
        reward: float,
        info: dict[str, Any],
        *,
        previous_observation: dict[str, Any] | None = None,
        candidate_actions: list[dict[str, Any]] | None = None,
        preview: dict[str, Any] | None = None,
    ) -> None:
        row = {
            "step": step,
            "month": observation["time"]["month"],
            "action": action,
            "reward": round(reward, 4),
            "score": info["score"],
            "last_event": info["last_event"],
        }
        with self.actions_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")
        if previous_observation is not None:
            _append_jsonl(
                self.observations_path,
                {
                    "step": step,
                    "phase": "before",
                    "observation": _strip_candidate_actions(previous_observation),
                },
            )
        _append_jsonl(
            self.observations_path,
            {
                "step": step,
                "phase": "after",
                "observation": _strip_candidate_actions(observation),
            },
        )
        _append_jsonl(
            self.candidate_actions_path,
            {
                "step": step,
                "candidate_actions": candidate_actions if candidate_actions is not None else previous_observation.get("candidate_actions", []) if previous_observation else [],
            },
        )
        reward_details = info.get("reward_details", {"reward": round(reward, 4), "components": {}})
        _append_jsonl(self.rewards_path, {"step": step, **reward_details})
        diagnostics = {
            "step": step,
            "preview": preview or {},
            "reward_diagnostics": reward_details.get("diagnostics", []),
            "last_event": info.get("last_event"),
        }
        _append_jsonl(self.diagnostics_path, diagnostics)
        episode_row = {
            "schema": EPISODE_TRACE_SCHEMA,
            "step": step,
            "before": _strip_candidate_actions(previous_observation) if previous_observation else None,
            "candidate_actions": candidate_actions if candidate_actions is not None else previous_observation.get("candidate_actions", []) if previous_observation else [],
            "chosen_action": action,
            "preview": preview or {},
            "after": _strip_candidate_actions(observation),
            "reward": reward_details,
            "info": info,
        }
        _append_jsonl(self.episode_path, episode_row)
        self._replay_steps.append(
            {
                "step": step,
                "action": action,
                "reward": reward_details,
                "last_event": info.get("last_event"),
            }
        )
        metrics = observation["metrics"]
        self._metrics_rows.append(
            {
                "step": step,
                "month": observation["time"]["month"],
                "score": metrics["score"],
                "cargo_delivered": metrics["cargo_delivered"],
                "operating_profit": metrics["operating_profit"],
                "cash": observation["company"]["cash"],
                "loan": observation["company"]["loan"],
                "routes": metrics["route_count"],
                "vehicles": metrics["vehicles"],
                "invalid_actions": metrics["invalid_actions"],
            }
        )

    def write_final(
        self,
        summary: dict[str, Any],
        final_state: dict[str, Any],
        final_observation: dict[str, Any],
    ) -> None:
        _write_json(self.run_dir / "summary.json", summary)
        _write_json(self.run_dir / "final_state.json", final_state)
        _write_json(
            self.run_dir / "replay.json",
            {
                "schema": REPLAY_SCHEMA,
                "schemas": schema_manifest(),
                "summary": summary,
                "steps": self._replay_steps,
            },
        )
        _write_metrics_csv(self.metrics_path, self._metrics_rows)
        _write_map_svg(self.run_dir / "screenshots" / "final_map.svg", final_observation)
        _write_trace(self.trace_path, summary, final_observation)


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
        handle.write("\n")


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def _strip_candidate_actions(observation: dict[str, Any] | None) -> dict[str, Any] | None:
    if observation is None:
        return None
    return {key: value for key, value in observation.items() if key != "candidate_actions"}


def _write_metrics_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _write_trace(path: Path, summary: dict[str, Any], observation: dict[str, Any]) -> None:
    lines = [
        f"# Run {summary['run_id']}",
        "",
        f"- Scenario: `{summary['scenario_id']}`",
        f"- Agent: `{summary['agent']}`",
        f"- Score: `{summary['score']}`",
        f"- Cargo delivered: `{observation['metrics']['cargo_delivered']}`",
        f"- Operating profit: `{observation['metrics']['operating_profit']}`",
        "",
        "## Final Event",
        "",
        observation["last_event"],
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def _write_map_svg(path: Path, observation: dict[str, Any]) -> None:
    width = int(observation["scenario"]["map"]["width"])
    height = int(observation["scenario"]["map"]["height"])
    scale = 10
    margin = 24
    svg_w = width * scale + margin * 2
    svg_h = height * scale + margin * 2
    nodes = {node["id"]: node for node in observation["nodes"]}
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="#f6f3ea"/>',
        f'<rect x="{margin}" y="{margin}" width="{width * scale}" height="{height * scale}" fill="#fffdf8" stroke="#b8aa91"/>',
    ]
    for route in observation["routes"]:
        src = nodes[route["source_id"]]
        dst = nodes[route["destination_id"]]
        color = "#7b4f27" if route["mode"] == "rail" else "#3f6f8f"
        lines.append(
            f'<line x1="{margin + src["x"] * scale}" y1="{margin + src["y"] * scale}" '
            f'x2="{margin + dst["x"] * scale}" y2="{margin + dst["y"] * scale}" '
            f'stroke="{color}" stroke-width="{max(2, 2 + route["vehicles"])}" opacity="0.72"/>'
        )
    for node in observation["nodes"]:
        x = margin + node["x"] * scale
        y = margin + node["y"] * scale
        fill = "#315c38" if node["kind"] == "town" else "#72523a"
        lines.append(f'<circle cx="{x}" cy="{y}" r="5" fill="{fill}" stroke="#1d1d1d"/>')
        lines.append(
            f'<text x="{x + 7}" y="{y - 7}" font-family="Arial" font-size="11" fill="#1f2933">'
            f'{_escape(node["name"])}</text>'
        )
    lines.append("</svg>")
    path.write_text("\n".join(lines), encoding="utf-8")


def _escape(value: str) -> str:
    return (
        value.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
