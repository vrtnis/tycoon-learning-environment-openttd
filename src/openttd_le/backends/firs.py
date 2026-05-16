from __future__ import annotations

import os
import tarfile
import tomllib
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from openttd_le.backends.visual import _openttd_user_dir
from openttd_le.core.types import EnvError


FIRS_ECONOMY_PARAMETERS = {
    "basic_temperate": 0,
    "temperate_basic": 0,
    "basic_arctic": 1,
    "arctic_basic": 1,
    "basic_tropic": 2,
    "tropic_basic": 2,
    "steeltown": 3,
    "in_a_hot_country": 4,
}


@dataclass(frozen=True)
class FIRSInstall:
    user_dir: Path
    newgrf_path: Path
    cfg_entry: str


@dataclass(frozen=True)
class FIRSRunConfig:
    economy: str = "basic_temperate"
    seed: int = 1
    map_x: int = 8
    map_y: int = 8
    landscape: str = "temperate"
    starting_year: int = 1950
    towns: int = 3
    industries: int = 4
    budget: int = 500_000
    years: int = 3
    allowed_modes: tuple[str, ...] = ("road",)
    vehicles_per_route: int = 5
    target_chain: tuple[dict[str, Any], ...] = (
        {
            "step": 1,
            "source_type": "Coal Mine",
            "destination_type": "Steel Mill",
            "cargo": "COAL",
            "deadline_year": 1,
            "required_delivered": 1,
        },
        {
            "step": 2,
            "source_type": "Steel Mill",
            "destination_type": "Metal Works",
            "cargo": "STEL",
            "deadline_year": 3,
            "required_delivered": 1,
        },
    )

    @property
    def economy_parameter(self) -> int:
        key = self.economy.strip().lower().replace("-", "_").replace(" ", "_")
        if key not in FIRS_ECONOMY_PARAMETERS:
            known = ", ".join(sorted(set(FIRS_ECONOMY_PARAMETERS)))
            raise EnvError(f"Unknown FIRS economy '{self.economy}'. Known economies: {known}.")
        return FIRS_ECONOMY_PARAMETERS[key]


def default_config_path(name: str = "firs_basic") -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / f"{name}.toml"


def load_firs_config(path: str | Path | None = None) -> FIRSRunConfig:
    config_path = Path(path) if path else default_config_path()
    if not config_path.exists():
        raise EnvError(f"FIRS config not found: {config_path}")
    with config_path.open("rb") as handle:
        payload = tomllib.load(handle)
    scenario = payload.get("scenario", {})
    objective = payload.get("objective", {})
    defaults = FIRSRunConfig()
    return FIRSRunConfig(
        economy=str(scenario.get("economy", defaults.economy)),
        seed=int(scenario.get("seed", 1)),
        map_x=int(scenario.get("map_x", 8)),
        map_y=int(scenario.get("map_y", 8)),
        landscape=str(scenario.get("landscape", defaults.landscape)),
        starting_year=int(scenario.get("starting_year", 1950)),
        towns=int(scenario.get("towns", 3)),
        industries=int(scenario.get("industries", 4)),
        budget=int(scenario.get("budget", 500_000)),
        years=int(scenario.get("years", 3)),
        allowed_modes=tuple(str(item) for item in scenario.get("allowed_modes", ["road"])),
        vehicles_per_route=int(scenario.get("vehicles_per_route", 5)),
        target_chain=tuple(dict(item) for item in objective.get("target_chain", []))
        or FIRSRunConfig().target_chain,
    )


def config_from_workbook_fields(fields: dict[str, Any], objectives: list[dict[str, Any]]) -> FIRSRunConfig:
    defaults = FIRSRunConfig()
    allowed_modes = _split_modes(str(fields.get("allowed_modes", "road")))
    target_chain: list[dict[str, Any]] = []
    for row in objectives:
        if not row.get("source_type") or not row.get("destination_type") or not row.get("cargo"):
            continue
        target_chain.append(
            {
                "step": int(row.get("step") or len(target_chain) + 1),
                "source_type": str(row["source_type"]),
                "destination_type": str(row["destination_type"]),
                "cargo": str(row["cargo"]).upper(),
                "deadline_year": int(row.get("deadline_year") or fields.get("years") or 3),
                "required_delivered": int(row.get("required_delivered") or 1),
            }
        )
    return FIRSRunConfig(
        economy=str(fields.get("economy") or defaults.economy),
        seed=int(fields.get("seed") or 1),
        map_x=int(fields.get("map_x") or 8),
        map_y=int(fields.get("map_y") or 8),
        landscape=str(fields.get("landscape") or defaults.landscape),
        starting_year=int(fields.get("starting_year") or 1950),
        towns=int(fields.get("towns") or 3),
        industries=int(fields.get("industries") or 4),
        budget=int(fields.get("budget") or 500_000),
        years=int(fields.get("years") or 3),
        allowed_modes=allowed_modes,
        vehicles_per_route=int(fields.get("vehicles_per_route") or 5),
        target_chain=tuple(target_chain) or FIRSRunConfig().target_chain,
    )


def verify_firs_installed(user_dir: str | Path | None = None) -> FIRSInstall:
    base = Path(user_dir).expanduser() if user_dir else _openttd_user_dir()
    candidates = find_firs_newgrfs(base)
    if not candidates:
        searched = [
            base / "newgrf",
            base / "content_download" / "newgrf",
            base,
        ]
        lines = "\n".join(f"  - {path}" for path in searched)
        raise EnvError(
            "FIRS NewGRF is not installed or was not found. Install 'FIRS Industry "
            "Replacement Set' from OpenTTD Online Content, then rerun.\n"
            f"Searched:\n{lines}"
        )
    newgrf = candidates[0]
    cfg_entry = cfg_entry_for_newgrf(base, newgrf)
    return FIRSInstall(user_dir=base, newgrf_path=newgrf, cfg_entry=cfg_entry)


def cfg_entry_for_newgrf(base: Path, newgrf: Path) -> str:
    if newgrf.suffix.lower() == ".tar":
        with tarfile.open(newgrf, "r") as archive:
            for member in archive.getmembers():
                name = member.name.replace("\\", "/")
                if member.isfile() and name.lower().endswith(".grf") and "firs" in name.lower():
                    return name
    if newgrf.suffix.lower() == ".zip":
        with zipfile.ZipFile(newgrf) as archive:
            for name in archive.namelist():
                normalized = name.replace("\\", "/")
                if normalized.lower().endswith(".grf") and "firs" in normalized.lower():
                    return normalized
    try:
        return newgrf.relative_to(base).as_posix()
    except ValueError:
        return newgrf.as_posix()


def find_firs_newgrfs(user_dir: Path) -> list[Path]:
    search_roots = [
        user_dir / "newgrf",
        user_dir / "content_download" / "newgrf",
        user_dir,
    ]
    matches: list[Path] = []
    for root in search_roots:
        if not root.exists():
            continue
        for path in root.rglob("*"):
            if not path.is_file():
                continue
            suffix = path.suffix.lower()
            name = path.name.lower()
            if suffix in {".grf", ".tar", ".zip"} and ("firs" in name or "industry_replacement" in name):
                matches.append(path)
    return sorted(set(matches), key=lambda item: (0 if item.suffix.lower() == ".grf" else 1, str(item)))


def render_newgrf_section(install: FIRSInstall, config: FIRSRunConfig) -> str:
    return f"[newgrf]\n{install.cfg_entry} = {config.economy_parameter}\n"


def render_firs_live_config(
    *,
    run_config: FIRSRunConfig,
    install: FIRSInstall,
    game_port: int,
    admin_port: int,
    admin_password: str,
) -> str:
    newgrf_section = render_newgrf_section(install, run_config)
    allowed_years = max(1, run_config.years)
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
number_towns = {run_config.towns}
number_industries = {run_config.industries}
terrain_type = 0
quantity_sea_lakes = 0
vehicle_breakdowns = 0

[network]
server_name = OpenTTD-LE FIRS GPT
client_name = OpenTTD-LE FIRS Server
server_port = {game_port}
server_admin_port = {admin_port}
admin_password = {admin_password}
allow_insecure_admin_login = true
server_game_type = local
server_advertise = false
server_password =
default_company_pass =
max_clients = 4
max_companies = 15
max_spectators = 4
max_init_time = 32000
max_join_time = 32000
max_download_time = 32000
max_lag_time = 32000
pause_on_join = false

[game_creation]
generation_seed = {run_config.seed}
map_x = {run_config.map_x}
map_y = {run_config.map_y}
landscape = {run_config.landscape}
starting_year = {run_config.starting_year}
ending_year = {run_config.starting_year + allowed_years}

[ai]
ai_in_multiplayer = true
ai_disable_veh_roadveh = false
ai_disable_veh_train = false
ai_disable_veh_aircraft = true
ai_disable_veh_ship = true

[script]
script_max_opcode_till_suspend = 100000

{newgrf_section}
[game_scripts]
OpenTTDLEGameScript =

[ai_players]
OpenTTDLECompany = start_date=1
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


def _split_modes(value: str) -> tuple[str, ...]:
    modes = tuple(part.strip().lower() for part in value.replace(";", ",").split(",") if part.strip())
    return modes or ("road",)
