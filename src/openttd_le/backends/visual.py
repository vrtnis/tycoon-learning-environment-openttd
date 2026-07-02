from __future__ import annotations

import json
import os
import shutil
import subprocess
import sysconfig
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openttd_le.backends.openttd import _find_openttd
from openttd_le.core.types import EnvError


BRIDGE_NAME = "OpenTTDLEBridge"
COMPANY_AI_NAME = "OpenTTDLECompany"
GAMESCRIPT_NAME = "OpenTTDLEGameScript"
OPENGFX_URL = "https://cdn.openttd.org/opengfx-releases/8.0/opengfx-8.0-all.zip"


def launch_watch_game(
    *,
    executable: str | None = None,
    output_root: Path | str = "runs_watch",
    seed: int = 1,
    resolution: str = "1280x800",
    model: str = "gpt-5.5",
    write_plan: bool = True,
    launch: bool = True,
) -> dict[str, Any]:
    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")

    run_dir = _new_run_dir(Path(output_root))
    bridge_dir = install_bridge()
    baseset_path = ensure_opengfx()
    cfg_path = run_dir / "openttd.cfg"
    cfg_path.write_text(_watch_config(seed), encoding="ascii")

    plan = _watch_plan(model) if write_plan else {"model": model, "used_llm": False}
    (run_dir / "gpt_plan.json").write_text(json.dumps(plan, indent=2), encoding="utf-8")

    command = [
        str(exe),
        "-g",
        "-G",
        str(seed),
        "-c",
        str(cfg_path),
        "-x",
        "-X",
        "-I",
        "OpenGFX",
        "-S",
        "NoSound",
        "-M",
        "NoMusic",
        "-r",
        resolution,
    ]

    pid: int | None = None
    if launch:
        process = subprocess.Popen(
            command,
            cwd=str(Path(exe).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        pid = process.pid

    launch_info = {
        "run_dir": str(run_dir),
        "pid": pid,
        "launched": launch,
        "executable": str(exe),
        "command": command,
        "config": str(cfg_path),
        "bridge_dir": str(bridge_dir),
        "baseset": str(baseset_path),
        "seed": seed,
        "resolution": resolution,
        "plan": plan,
        "note": (
            "A visible OpenTTD window should open. The bridge AI is configured "
            "as the first competitor and will mark its decisions with map signs."
        ),
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
    return launch_info


def install_bridge(source_dir: Path | None = None) -> Path:
    return _install_script_dir(BRIDGE_NAME, "ai", source_dir)


def install_live_bridge() -> dict[str, str]:
    company_dir = _install_script_dir(COMPANY_AI_NAME, "ai", None)
    gamescript_dir = _install_script_dir(GAMESCRIPT_NAME, "game", None)
    return {"company_ai_dir": str(company_dir), "gamescript_dir": str(gamescript_dir)}


def _install_script_dir(name: str, kind: str, source_dir: Path | None = None) -> Path:
    source = source_dir or _bridge_source_root() / name
    if not source.exists():
        raise EnvError(f"Bridge source not found: {source}")
    target = _openttd_user_dir() / kind / name
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file():
            destination = target / item.name
            if _same_file_bytes(item, destination):
                continue
            try:
                shutil.copy2(item, destination)
            except PermissionError:
                if destination.exists():
                    continue
                raise
    return target


def _bridge_source_root() -> Path:
    candidates = [
        _project_root() / "openttd_bridge",
        Path(sysconfig.get_path("data")) / "openttd_bridge",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def _same_file_bytes(source: Path, destination: Path) -> bool:
    if not destination.exists() or not destination.is_file():
        return False
    try:
        if source.stat().st_size != destination.stat().st_size:
            return False
        return source.read_bytes() == destination.read_bytes()
    except OSError:
        return True


def ensure_opengfx() -> Path:
    baseset_dir = _openttd_user_dir() / "baseset"
    baseset_dir.mkdir(parents=True, exist_ok=True)
    existing = sorted(baseset_dir.glob("opengfx*.tar"))
    if existing:
        return existing[-1]

    tmp_zip = baseset_dir / "opengfx-8.0-all.zip"
    request = urllib.request.Request(OPENGFX_URL, headers={"User-Agent": "TycoonLE-OpenTTD/0.1"})
    with urllib.request.urlopen(request, timeout=120) as response, tmp_zip.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    with zipfile.ZipFile(tmp_zip) as archive:
        tar_members = [name for name in archive.namelist() if name.lower().endswith(".tar")]
        if not tar_members:
            raise EnvError("OpenGFX download did not contain a .tar base graphics set.")
        member = tar_members[0]
        target = baseset_dir / Path(member).name
        with archive.open(member) as source, target.open("wb") as dest:
            shutil.copyfileobj(source, dest)
    tmp_zip.unlink(missing_ok=True)
    return target


def _watch_plan(model: str) -> dict[str, Any]:
    api_key = os.environ.get("OPENAI_API_KEY")
    prompt = (
        "You are choosing the first visible move for an OpenTTD learning-environment demo. "
        "The bridge currently supports one executable macro-plan: connect the two largest "
        "towns by road, leaving visible signs at each decision point. Return compact JSON "
        "with keys plan_id and rationale."
    )
    if not api_key:
        return {
            "model": model,
            "used_llm": False,
            "plan_id": "largest_town_road",
            "rationale": "Fallback bridge plan: connect the two largest towns by road.",
        }

    payload = {
        "model": model,
        "input": [
            {"role": "system", "content": "Return only JSON. No prose."},
            {"role": "user", "content": prompt},
        ],
        "max_output_tokens": 200,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        text = data.get("output_text") or _extract_response_text(data)
        parsed = _parse_json_object(text)
        parsed["model"] = model
        parsed["used_llm"] = True
        return parsed
    except Exception as exc:  # pragma: no cover - external service path
        return {
            "model": model,
            "used_llm": False,
            "plan_id": "largest_town_road",
            "rationale": f"Fell back to built-in plan because the OpenAI plan call failed: {exc}",
        }


def _extract_response_text(data: dict[str, Any]) -> str:
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _parse_json_object(text: str) -> dict[str, Any]:
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        raise ValueError(f"Model did not return JSON: {text}")
    return json.loads(text[start : end + 1])


def _watch_config(seed: int) -> str:
    return f"""[misc]
graphicsset = OpenGFX
soundsset = NoSound
musicset = NoMusic
display_opt = SHOW_TOWN_NAMES|SHOW_STATION_NAMES|SHOW_SIGNS|FULL_ANIMATION|FULL_DETAIL|WAYPOINTS|SHOW_COMPETITOR_SIGNS

[gui]
pause_on_newgame = false

[difficulty]
competitor_start_time = 0
competitors_interval = 0
max_no_competitors = 1
number_towns = 2
number_industries = 1
terrain_type = 0
quantity_sea_lakes = 0
vehicle_breakdowns = 0

[network]
server_name = TycoonLE OpenTTD Watch
server_advertise = false
max_companies = 15

[game_creation]
generation_seed = {seed}
map_x = 7
map_y = 7
landscape = temperate
starting_year = 1950

[ai]
ai_in_multiplayer = true
ai_disable_veh_roadveh = false
ai_disable_veh_train = false
ai_disable_veh_aircraft = true
ai_disable_veh_ship = true

[script]
script_max_opcode_till_suspend = 100000

[ai_players]
{BRIDGE_NAME} = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
none = start_date=1
"""


def _new_run_dir(output_root: Path) -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root.resolve() / f"{timestamp}_watch_gpt"
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def _project_root() -> Path:
    return Path(__file__).resolve().parents[3]


def _openttd_user_dir() -> Path:
    override = os.environ.get("OPENTTD_USER_DIR")
    if override:
        path = Path(override).expanduser()
        path.mkdir(parents=True, exist_ok=True)
        return path
    documents = _documents_dir()
    path = documents / "OpenTTD"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _documents_dir() -> Path:
    if os.name != "nt":
        return Path.home() / "Documents"
    try:
        import winreg

        with winreg.OpenKey(
            winreg.HKEY_CURRENT_USER,
            r"Software\Microsoft\Windows\CurrentVersion\Explorer\User Shell Folders",
        ) as key:
            value, _ = winreg.QueryValueEx(key, "Personal")
            if value:
                return Path(os.path.expandvars(value))
    except Exception:
        pass
    return Path(os.environ.get("USERPROFILE", str(Path.home()))) / "Documents"
