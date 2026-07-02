from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Any


def render_core_replay(
    *,
    episode: Path | str | None = None,
    replay: Path | str | None = None,
    out: Path | str,
    fps: int = 1,
) -> dict[str, Any]:
    episode_path = _resolve_episode_path(episode=episode, replay=replay)
    output_path = Path(out)
    rows = _read_jsonl(episode_path)
    if not rows:
        raise ValueError(f"No episode rows found in {episode_path}")
    if output_path.suffix.lower() == ".mp4":
        frame_dir = output_path.with_suffix("")
        frames = _render_svg_frames(rows, frame_dir)
        mp4 = _render_mp4_from_episode(rows, output_path, fps=fps)
        return {
            "episode": str(episode_path),
            "frames_dir": str(frame_dir),
            "frames": len(frames),
            "mp4": str(mp4) if mp4 else None,
            "note": "MP4 rendering requires ffmpeg; SVG frames are always written.",
        }
    frames = _render_svg_frames(rows, output_path)
    return {
        "episode": str(episode_path),
        "frames_dir": str(output_path),
        "frames": len(frames),
        "index": str(output_path / "index.html"),
        "mp4": None,
    }


def _resolve_episode_path(*, episode: Path | str | None, replay: Path | str | None) -> Path:
    if episode:
        path = Path(episode)
        if not path.exists():
            raise FileNotFoundError(path)
        return path
    if not replay:
        raise ValueError("Either episode or replay is required.")
    replay_path = Path(replay)
    if not replay_path.exists():
        raise FileNotFoundError(replay_path)
    sibling = replay_path.parent / "episode.jsonl"
    if sibling.exists():
        return sibling
    raise FileNotFoundError(f"Could not locate sibling episode.jsonl next to {replay_path}")


def _render_svg_frames(rows: list[dict[str, Any]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    frames: list[Path] = []
    for index, row in enumerate(rows, start=1):
        observation = row.get("after") or row.get("before") or {}
        path = out_dir / f"frame_{index:04d}.svg"
        path.write_text(_render_svg(row, observation), encoding="utf-8")
        frames.append(path)
    (out_dir / "index.html").write_text(_render_index(frames), encoding="utf-8")
    return frames


def _render_svg(row: dict[str, Any], observation: dict[str, Any]) -> str:
    scenario = observation.get("scenario", {})
    map_info = scenario.get("map", {})
    width = int(map_info.get("width", 80) or 80)
    height = int(map_info.get("height", 80) or 80)
    scale = max(5, min(10, int(640 / max(width, height, 1))))
    margin = 28
    panel_h = 94
    svg_w = width * scale + margin * 2
    svg_h = height * scale + margin * 2 + panel_h
    nodes = {node["id"]: node for node in observation.get("nodes", [])}
    metrics = observation.get("metrics", {})
    reward = row.get("reward", {})
    action = row.get("chosen_action", {})
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{svg_w}" height="{svg_h}" viewBox="0 0 {svg_w} {svg_h}">',
        '<rect width="100%" height="100%" fill="#f7f4ec"/>',
        f'<rect x="{margin}" y="{margin}" width="{width * scale}" height="{height * scale}" fill="#fffef9" stroke="#9d927e"/>',
    ]
    for route in observation.get("routes", []):
        src = nodes.get(route.get("source_id"))
        dst = nodes.get(route.get("destination_id"))
        if not src or not dst:
            continue
        color = "#6d4b2f" if route.get("mode") == "rail" else "#2d6683"
        x1, y1 = margin + int(src["x"]) * scale, margin + int(src["y"]) * scale
        x2, y2 = margin + int(dst["x"]) * scale, margin + int(dst["y"]) * scale
        lines.append(
            f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" '
            f'stroke-width="{max(2, 2 + int(route.get("vehicles", 0) or 0))}" opacity="0.75"/>'
        )
    for node in observation.get("nodes", []):
        x = margin + int(node["x"]) * scale
        y = margin + int(node["y"]) * scale
        fill = "#2f6b3f" if node.get("kind") == "town" else "#79543b"
        lines.append(f'<circle cx="{x}" cy="{y}" r="6" fill="{fill}" stroke="#1e1e1e" stroke-width="1.2"/>')
        lines.append(
            f'<text x="{x + 8}" y="{y - 8}" font-family="Arial" font-size="11" fill="#232323">'
            f'{_escape(str(node.get("name", "")))}</text>'
        )
    panel_y = height * scale + margin * 2
    lines.extend(
        [
            f'<rect x="0" y="{panel_y}" width="{svg_w}" height="{panel_h}" fill="#1f2933"/>',
            f'<text x="20" y="{panel_y + 24}" font-family="Arial" font-size="16" fill="#ffffff">Step {row.get("step")} - {_escape(str(scenario.get("id", "")))}</text>',
            f'<text x="20" y="{panel_y + 48}" font-family="Arial" font-size="13" fill="#dbe3ea">Action: {_escape(_compact_action(action))}</text>',
            f'<text x="20" y="{panel_y + 68}" font-family="Arial" font-size="13" fill="#dbe3ea">Score {metrics.get("score", 0)} | Cargo {metrics.get("cargo_delivered", 0)} | Profit {metrics.get("operating_profit", 0)} | Reward {reward.get("reward", 0)}</text>',
            f'<text x="20" y="{panel_y + 86}" font-family="Arial" font-size="12" fill="#b8c7d4">{_escape(str(observation.get("last_event", ""))[:180])}</text>',
            "</svg>",
        ]
    )
    return "\n".join(lines)


def _render_index(frames: list[Path]) -> str:
    links = "\n".join(
        f'<li><a href="{path.name}">{path.name}</a><br><img src="{path.name}" width="720"></li>' for path in frames
    )
    return f"<!doctype html><meta charset='utf-8'><title>TycoonLE OpenTTD Replay</title><ol>{links}</ol>"


def _render_mp4_from_episode(rows: list[dict[str, Any]], output_path: Path, *, fps: int) -> Path | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    ppm_dir = output_path.with_suffix(".ppm_frames")
    ppm_dir.mkdir(parents=True, exist_ok=True)
    for index, row in enumerate(rows, start=1):
        observation = row.get("after") or row.get("before") or {}
        _write_ppm(ppm_dir / f"frame_{index:04d}.ppm", row, observation)
    cmd = [
        ffmpeg,
        "-y",
        "-framerate",
        str(max(1, fps)),
        "-i",
        str(ppm_dir / "frame_%04d.ppm"),
        "-pix_fmt",
        "yuv420p",
        str(output_path),
    ]
    result = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
    return output_path if result.returncode == 0 and output_path.exists() else None


def _write_ppm(path: Path, row: dict[str, Any], observation: dict[str, Any]) -> None:
    width, height = 960, 640
    pixels = bytearray([246, 243, 235] * width * height)
    nodes = {node["id"]: node for node in observation.get("nodes", [])}
    map_info = observation.get("scenario", {}).get("map", {})
    map_w = max(1, int(map_info.get("width", 80) or 80))
    map_h = max(1, int(map_info.get("height", 80) or 80))

    def put(x: int, y: int, color: tuple[int, int, int]) -> None:
        if 0 <= x < width and 0 <= y < height:
            idx = (y * width + x) * 3
            pixels[idx : idx + 3] = bytes(color)

    def project(node: dict[str, Any]) -> tuple[int, int]:
        return 40 + int(int(node["x"]) / map_w * 860), 40 + int(int(node["y"]) / map_h * 500)

    for route in observation.get("routes", []):
        src = nodes.get(route.get("source_id"))
        dst = nodes.get(route.get("destination_id"))
        if src and dst:
            _draw_line(put, *project(src), *project(dst), (44, 102, 131) if route.get("mode") != "rail" else (109, 75, 47))
    for node in observation.get("nodes", []):
        x, y = project(node)
        _draw_disc(put, x, y, 6, (47, 107, 63) if node.get("kind") == "town" else (121, 84, 59))
    # Step progress bar; text stays in SVG/HTML, MP4 is intentionally map-first.
    step = int(row.get("step", 1) or 1)
    bar_w = min(880, step * 48)
    for y in range(585, 604):
        for x in range(40, 40 + bar_w):
            put(x, y, (31, 41, 51))
    path.write_bytes(f"P6\n{width} {height}\n255\n".encode("ascii") + pixels)


def _draw_line(put: Any, x1: int, y1: int, x2: int, y2: int, color: tuple[int, int, int]) -> None:
    dx = abs(x2 - x1)
    dy = -abs(y2 - y1)
    sx = 1 if x1 < x2 else -1
    sy = 1 if y1 < y2 else -1
    err = dx + dy
    while True:
        for ox in (-1, 0, 1):
            for oy in (-1, 0, 1):
                put(x1 + ox, y1 + oy, color)
        if x1 == x2 and y1 == y2:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy
            x1 += sx
        if e2 <= dx:
            err += dx
            y1 += sy


def _draw_disc(put: Any, cx: int, cy: int, radius: int, color: tuple[int, int, int]) -> None:
    for y in range(cy - radius, cy + radius + 1):
        for x in range(cx - radius, cx + radius + 1):
            if (x - cx) ** 2 + (y - cy) ** 2 <= radius**2:
                put(x, y, color)


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _compact_action(action: dict[str, Any]) -> str:
    return json.dumps(action, separators=(",", ":"))[:220]


def _escape(value: str) -> str:
    return value.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
