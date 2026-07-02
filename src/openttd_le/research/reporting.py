from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


def write_benchmark_report(
    *,
    validity_report: dict[str, Any] | Path | str | None = None,
    training_report: dict[str, Any] | Path | str | None = None,
    route_builder_report: dict[str, Any] | Path | str | None = None,
    output_dir: Path | str,
    title: str = "TycoonLE OpenTTD FIRS Benchmark Report",
) -> dict[str, Any]:
    """Write researcher-facing Markdown, CSV tables, and simple SVG curves."""

    out = Path(output_dir)
    tables_dir = out / "tables"
    curves_dir = out / "curves"
    tables_dir.mkdir(parents=True, exist_ok=True)
    curves_dir.mkdir(parents=True, exist_ok=True)

    validity = _load_report(validity_report)
    training = _load_report(training_report)
    route_builder = _load_report(route_builder_report)

    artifacts: dict[str, Any] = {
        "tables": {},
        "curves": {},
    }
    lines = [f"# {title}", ""]

    if validity:
        lines.extend(_validity_markdown(validity))
        validity_tables = _write_validity_tables(validity, tables_dir)
        artifacts["tables"].update(validity_tables)

    if training:
        lines.extend(_training_markdown(training))
        training_tables = _write_training_tables(training, tables_dir)
        artifacts["tables"].update(training_tables)
        curve_rows = _training_curve_rows(training)
        if curve_rows:
            curve_path = curves_dir / "learning_curves.svg"
            curve_path.write_text(_learning_curve_svg(curve_rows), encoding="utf-8")
            artifacts["curves"]["learning_curves"] = str(curve_path)
            lines.extend(["", "## Learning Curves", "", f"![Learning curves]({curve_path.as_posix()})"])

    if route_builder:
        lines.extend(_route_builder_markdown(route_builder))
        route_tables = _write_route_builder_tables(route_builder, tables_dir)
        artifacts["tables"].update(route_tables)

    if not validity and not training and not route_builder:
        lines.extend(["No validity or training report was supplied."])

    report_path = out / "benchmark_report.md"
    report_path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
    payload = {
        "schema": "openttd-le-benchmark-report-artifacts-v1",
        "report": str(report_path),
        **artifacts,
    }
    (out / "benchmark_report_artifacts.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_report(report: dict[str, Any] | Path | str | None) -> dict[str, Any]:
    if report is None:
        return {}
    if isinstance(report, dict):
        return report
    path = Path(report)
    return json.loads(path.read_text(encoding="utf-8"))


def _validity_markdown(report: dict[str, Any]) -> list[str]:
    sections = report.get("sections", {})
    lines = [
        "## Validity Pack",
        "",
        f"- Suite: `{report.get('suite', '-')}`",
        f"- Tasks: {len(report.get('tasks', []))}",
        f"- Splits: {_split_summary(report.get('task_metadata') or [])}",
        f"- Seeds: {', '.join(str(seed) for seed in report.get('seeds', [])) or '-'}",
        f"- Overall pass: `{bool(report.get('ok'))}`",
        "",
        "| Section | OK | Runs | Key metric |",
        "| --- | ---: | ---: | --- |",
    ]
    for name, section in sections.items():
        runs = section.get("runs", "-")
        if section.get("skipped"):
            metric = "skipped"
        elif name == "route_builder":
            metric = f"success={section.get('median_operational_success_rate', 0)}"
        elif name == "throughput":
            metric = f"median step={section.get('median_step_seconds', 0)}s"
        elif name == "determinism":
            metric = f"passed={section.get('passed', 0)}"
        else:
            metric = "see CSV"
        lines.append(f"| {name} | `{bool(section.get('ok', True))}` | {runs} | {metric} |")
    return lines


def _training_markdown(report: dict[str, Any]) -> list[str]:
    aggregate = report.get("aggregate", {})
    lines = [
        "",
        "## RL Training And Eval",
        "",
        f"- Task: `{report.get('task_id', '-')}`",
        f"- Algorithms: {', '.join(report.get('algorithms', [])) or '-'}",
        f"- Total runs: {aggregate.get('runs', 0)}",
        "",
        "| Algorithm | Runs | Best mean reward | Final mean reward |",
        "| --- | ---: | ---: | ---: |",
    ]
    for algorithm, item in sorted((aggregate.get("per_algorithm") or {}).items()):
        lines.append(
            f"| {algorithm} | {item.get('runs', 0)} | "
            f"{item.get('best_mean_reward', 0)} | {item.get('final_mean_reward', 0)} |"
        )
    return lines


def _route_builder_markdown(report: dict[str, Any]) -> list[str]:
    aggregate = report.get("aggregate", {}) or {}
    failure_counts = aggregate.get("failure_counts", {}) or {}
    failures = ", ".join(f"{key}={value}" for key, value in sorted(failure_counts.items())) or "-"
    return [
        "",
        "## Route Builder Reliability",
        "",
        f"- Seed: `{report.get('seed', '-')}`",
        f"- Attempts: {aggregate.get('attempts', report.get('attempts_executed', 0))} / requested {report.get('attempts_requested', '-')}",
        f"- Infeasible candidates skipped: {report.get('skipped_infeasible', 0)}",
        f"- Operational success rate: {aggregate.get('operational_success_rate', 0)}",
        f"- Feasible attempts: {aggregate.get('feasible_attempts', '-')}",
        f"- Feasible operational success rate: {aggregate.get('feasible_operational_success_rate', 0)}",
        f"- Target success rate: {aggregate.get('target_success_rate', 0)}",
        f"- Gate pass: `{bool(aggregate.get('level1_pass'))}`",
        f"- Feasible gate pass: `{bool(aggregate.get('feasible_level1_pass'))}`",
        f"- Failures: {failures}",
    ]


def _write_validity_tables(report: dict[str, Any], tables_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    task_metadata = report.get("task_metadata") or []
    if task_metadata:
        path = tables_dir / "tasks.csv"
        _write_csv(path, task_metadata)
        artifacts["tasks"] = str(path)

    sections_path = tables_dir / "validity_sections.csv"
    with sections_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["section", "ok", "skipped", "runs", "metric", "value"])
        writer.writeheader()
        for name, section in (report.get("sections") or {}).items():
            metric, value = _section_metric(name, section)
            writer.writerow(
                {
                    "section": name,
                    "ok": section.get("ok", True),
                    "skipped": section.get("skipped", False),
                    "runs": section.get("runs", ""),
                    "metric": metric,
                    "value": value,
                }
            )
    artifacts["validity_sections"] = str(sections_path)

    baseline_rows = []
    for task_id, aggregate in ((report.get("sections", {}).get("baselines", {}) or {}).get("per_task") or {}).items():
        for agent, row in (aggregate.get("per_agent") or {}).items():
            baseline_rows.append({"task_id": task_id, "agent": agent, **row})
    if baseline_rows:
        path = tables_dir / "baseline_results.csv"
        _write_csv(path, baseline_rows)
        artifacts["baseline_results"] = str(path)
    return artifacts


def _split_summary(task_metadata: list[dict[str, Any]]) -> str:
    if not task_metadata:
        return "-"
    counts: dict[str, int] = {}
    for item in task_metadata:
        split = str(item.get("split") or "unknown")
        counts[split] = counts.get(split, 0) + 1
    return ", ".join(f"{name}={count}" for name, count in sorted(counts.items()))


def _write_training_tables(report: dict[str, Any], tables_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    runs = report.get("runs") or []
    if runs:
        summary_rows = [
            {
                "algorithm": run.get("algorithm"),
                "seed": run.get("seed"),
                "timesteps": run.get("timesteps"),
                "best_mean_reward": run.get("best_mean_reward"),
                "final_mean_reward": run.get("final_mean_reward"),
                "curve": run.get("learning_curve"),
                "run_dir": run.get("run_dir"),
            }
            for run in runs
        ]
        path = tables_dir / "training_summary.csv"
        _write_csv(path, summary_rows)
        artifacts["training_summary"] = str(path)

    curve_rows = _training_curve_rows(report)
    if curve_rows:
        path = tables_dir / "learning_curve.csv"
        _write_csv(path, curve_rows)
        artifacts["learning_curve"] = str(path)
    return artifacts


def _write_route_builder_tables(report: dict[str, Any], tables_dir: Path) -> dict[str, str]:
    artifacts: dict[str, str] = {}
    aggregate = report.get("aggregate", {}) or {}
    path = tables_dir / "route_builder_summary.csv"
    _write_csv(
        path,
        [
            {
                "seed": report.get("seed"),
                "economy": report.get("economy"),
                **aggregate,
            }
        ],
    )
    artifacts["route_builder_summary"] = str(path)

    attempts_path = report.get("attempts")
    if attempts_path:
        attempts = []
        source = Path(str(attempts_path))
        if source.exists():
            for line in source.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                item = json.loads(line)
                attempts.append(
                    {
                        "attempt": item.get("attempt"),
                        "source": item.get("source_name"),
                        "destination": item.get("destination_name"),
                        "cargo": item.get("cargo_label"),
                        "build_success": item.get("build_success"),
                        "active_success": item.get("active_success"),
                        "operational_success": item.get("operational_success"),
                        "delivered": item.get("delivered"),
                        "profit": item.get("profit"),
                        "failure_reason": item.get("failure_reason"),
                        "error": item.get("error"),
                    }
                )
        if attempts:
            attempts_table = tables_dir / "route_builder_attempts.csv"
            _write_csv(attempts_table, attempts)
            artifacts["route_builder_attempts"] = str(attempts_table)
    return artifacts


def _training_curve_rows(report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for run in report.get("runs") or []:
        for point in run.get("curve_points") or []:
            rows.append(
                {
                    "algorithm": run.get("algorithm"),
                    "seed": run.get("seed"),
                    "timesteps": point.get("timesteps"),
                    "mean_reward": point.get("mean_reward"),
                    "success_rate": point.get("success_rate"),
                }
            )
    return rows


def _section_metric(name: str, section: dict[str, Any]) -> tuple[str, Any]:
    if section.get("skipped"):
        return "skipped", True
    if name == "route_builder":
        return "median_operational_success_rate", section.get("median_operational_success_rate")
    if name == "throughput":
        return "median_step_seconds", section.get("median_step_seconds")
    if name == "determinism":
        return "passed", section.get("passed")
    return "runs", section.get("runs")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames: list[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _learning_curve_svg(rows: list[dict[str, Any]]) -> str:
    width = 900
    height = 420
    margin_left = 70
    margin_bottom = 55
    margin_top = 30
    margin_right = 30
    plot_w = width - margin_left - margin_right
    plot_h = height - margin_top - margin_bottom
    grouped: dict[str, list[tuple[float, float]]] = {}
    for row in rows:
        try:
            x = float(row.get("timesteps", 0) or 0)
            y = float(row.get("mean_reward", 0) or 0)
        except (TypeError, ValueError):
            continue
        grouped.setdefault(str(row.get("algorithm")), []).append((x, y))
    if not grouped:
        return "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"900\" height=\"420\"></svg>\n"
    xs = [x for points in grouped.values() for x, _ in points]
    ys = [y for points in grouped.values() for _, y in points]
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    if x_min == x_max:
        x_max = x_min + 1.0
    if y_min == y_max:
        y_max = y_min + 1.0
    colors = ["#2563eb", "#dc2626", "#16a34a", "#9333ea", "#ea580c", "#0891b2"]

    def scale_x(value: float) -> float:
        return margin_left + (value - x_min) / (x_max - x_min) * plot_w

    def scale_y(value: float) -> float:
        return margin_top + plot_h - (value - y_min) / (y_max - y_min) * plot_h

    svg = [
        f"<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"{width}\" height=\"{height}\" viewBox=\"0 0 {width} {height}\">",
        "<rect width=\"100%\" height=\"100%\" fill=\"#ffffff\"/>",
        f"<line x1=\"{margin_left}\" y1=\"{margin_top + plot_h}\" x2=\"{margin_left + plot_w}\" y2=\"{margin_top + plot_h}\" stroke=\"#111827\"/>",
        f"<line x1=\"{margin_left}\" y1=\"{margin_top}\" x2=\"{margin_left}\" y2=\"{margin_top + plot_h}\" stroke=\"#111827\"/>",
        f"<text x=\"{width / 2}\" y=\"{height - 15}\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">Environment steps</text>",
        f"<text x=\"18\" y=\"{height / 2}\" transform=\"rotate(-90 18 {height / 2})\" text-anchor=\"middle\" font-family=\"Arial\" font-size=\"14\">Mean episode reward</text>",
    ]
    for index, (name, points) in enumerate(sorted(grouped.items())):
        points = sorted(points)
        color = colors[index % len(colors)]
        path = " ".join(f"{scale_x(x):.1f},{scale_y(y):.1f}" for x, y in points)
        svg.append(f"<polyline fill=\"none\" stroke=\"{color}\" stroke-width=\"3\" points=\"{path}\"/>")
        for x, y in points:
            svg.append(f"<circle cx=\"{scale_x(x):.1f}\" cy=\"{scale_y(y):.1f}\" r=\"4\" fill=\"{color}\"/>")
        legend_y = margin_top + index * 22
        svg.append(f"<rect x=\"{width - 190}\" y=\"{legend_y - 10}\" width=\"12\" height=\"12\" fill=\"{color}\"/>")
        svg.append(f"<text x=\"{width - 172}\" y=\"{legend_y}\" font-family=\"Arial\" font-size=\"13\">{_escape_xml(name)}</text>")
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def _escape_xml(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
