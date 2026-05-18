from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from openttd_le.core.schemas import DATASET_SCHEMA


def export_core_dataset(source: Path | str, out: Path | str, *, output_format: str | None = None) -> Path:
    source_path = Path(source)
    output_path = Path(out)
    selected_format = _resolve_format(output_path, output_format)
    if selected_format == "parquet":
        return _export_parquet(source_path, output_path)
    if selected_format != "jsonl":
        raise ValueError(f"Unsupported dataset format: {selected_format}")
    return _export_jsonl(source_path, output_path)


def _export_jsonl(source_path: Path, output_path: Path) -> Path:
    run_dirs = _find_run_dirs(source_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for run_dir in run_dirs:
            summary = _read_json(run_dir / "summary.json")
            for row in _read_jsonl(run_dir / "episode.jsonl"):
                handle.write(
                    json.dumps(
                        {
                            **row,
                            "schema": DATASET_SCHEMA,
                            "episode_schema": row.get("schema"),
                            "run_id": summary.get("run_id", run_dir.name),
                            "scenario_id": summary.get("scenario_id"),
                            "agent": summary.get("agent"),
                            "model": summary.get("model"),
                            "seed": summary.get("seed"),
                            "completed_score": summary.get("score"),
                        },
                        separators=(",", ":"),
                    )
                    + "\n"
                )
    return output_path


def _export_parquet(source_path: Path, output_path: Path) -> Path:
    try:
        import pandas as pd  # type: ignore[import-not-found]
    except ImportError as exc:
        raise RuntimeError("Parquet export requires the optional 'parquet' extra: pip install -e .[parquet]") from exc
    rows: list[dict[str, Any]] = []
    for run_dir in _find_run_dirs(source_path):
        summary = _read_json(run_dir / "summary.json")
        for row in _read_jsonl(run_dir / "episode.jsonl"):
            rows.append(
                {
                    "schema": DATASET_SCHEMA,
                    "episode_schema": row.get("schema"),
                    "run_id": summary.get("run_id", run_dir.name),
                    "scenario_id": summary.get("scenario_id"),
                    "agent": summary.get("agent"),
                    "model": summary.get("model"),
                    "seed": summary.get("seed"),
                    "completed_score": summary.get("score"),
                    "step": row.get("step"),
                    "chosen_action": json.dumps(row.get("chosen_action", {}), separators=(",", ":")),
                    "candidate_actions": json.dumps(row.get("candidate_actions", []), separators=(",", ":")),
                    "before": json.dumps(row.get("before", {}), separators=(",", ":")),
                    "after": json.dumps(row.get("after", {}), separators=(",", ":")),
                    "preview": json.dumps(row.get("preview", {}), separators=(",", ":")),
                    "reward": json.dumps(row.get("reward", {}), separators=(",", ":")),
                }
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(output_path, index=False)
    return output_path


def _resolve_format(output_path: Path, output_format: str | None) -> str:
    if output_format:
        return output_format.lower()
    if output_path.suffix.lower() == ".parquet":
        return "parquet"
    return "jsonl"


def _find_run_dirs(source: Path) -> list[Path]:
    if (source / "episode.jsonl").exists():
        return [source]
    return sorted(path.parent for path in source.glob("*/episode.jsonl"))


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows
