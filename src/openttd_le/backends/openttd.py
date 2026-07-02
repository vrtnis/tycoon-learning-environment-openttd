from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
import time
from typing import Any

from openttd_le.backends.base import Backend
from openttd_le.core.types import EnvError, GameState, Metrics, Scenario


class OpenTTDBackend(Backend):
    """Real OpenTTD process backend.

    This backend currently performs process-level integration: it resolves an
    OpenTTD executable, creates an isolated run directory, and starts a dedicated
    OpenTTD process. Macro-action construction still requires the planned
    GameScript/NoAI command bridge, so `apply()` intentionally raises instead of
    pretending that real gameplay has been automated.
    """

    def __init__(
        self,
        executable: str | None = None,
        keep_run_dir: bool = False,
        extra_args: list[str] | None = None,
    ) -> None:
        self.executable = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
        self.keep_run_dir = keep_run_dir
        self.extra_args = extra_args or []
        self.run_dir: tempfile.TemporaryDirectory[str] | None = None
        self.process: subprocess.Popen[str] | None = None
        self.scenario: Scenario | None = None
        self.state: GameState | None = None

    def reset(self, scenario: Scenario, seed: int | None = None) -> GameState:
        if not self.executable:
            raise EnvError(
                "OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE."
            )
        self.close()
        self.scenario = scenario
        self.run_dir = tempfile.TemporaryDirectory(prefix=f"tycoonle-openttd-{scenario.id}-")
        cfg_path = os.path.join(self.run_dir.name, "openttd.cfg")
        with open(cfg_path, "w", encoding="utf-8") as handle:
            handle.write(_config_text(seed or 0))

        cmd = [
            self.executable,
            "-D",
            "-c",
            cfg_path,
            "-x",
            "-X",
            "-d",
            "misc=1",
            *self.extra_args,
        ]
        self.process = subprocess.Popen(
            cmd,
            cwd=self.run_dir.name,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        self.state = GameState(
            scenario_id=scenario.id,
            month=0,
            step=0,
            cash=scenario.budget.starting_cash,
            loan=0.0,
            metrics=Metrics(cash=scenario.budget.starting_cash),
            last_event=(
                "OpenTTD dedicated process started. Gameplay macro-actions need "
                "the GameScript/NoAI bridge."
            ),
        )
        return self.state

    def apply(self, action: dict[str, Any]) -> GameState:
        raise NotImplementedError(
            "OpenTTD process backend is installed, but macro-action execution is "
            "not implemented until the GameScript/NoAI bridge is added."
        )

    def close(self) -> None:
        if self.process and self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait(timeout=10)
        self.process = None
        if self.run_dir and not self.keep_run_dir:
            self.run_dir.cleanup()
        self.run_dir = None

    def artifact_state(self) -> dict[str, Any]:
        return {
            "backend": "openttd",
            "executable": self.executable,
            "run_dir": self.run_dir.name if self.run_dir else None,
            "process_running": bool(self.process and self.process.poll() is None),
            "scenario_id": self.scenario.id if self.scenario else None,
            "last_event": self.state.last_event if self.state else None,
        }

    def smoke(self) -> dict[str, Any]:
        if not self.executable:
            raise EnvError("OpenTTD executable not found.")
        return {
            "executable": self.executable,
            "exists": os.path.exists(self.executable),
            "note": "Use --launch for a dedicated-server process smoke test.",
        }

    def smoke_launch(self, scenario: Scenario, seconds: float = 3.0) -> dict[str, Any]:
        state = self.reset(scenario, seed=1)
        time.sleep(seconds)
        poll = self.process.poll() if self.process else None
        stdout_head: list[str] = []
        stderr_head: list[str] = []
        if poll is not None and self.process:
            stdout, stderr = self.process.communicate(timeout=5)
            stdout_head = stdout.splitlines()[:12]
            stderr_head = stderr.splitlines()[:12]
        artifact = self.artifact_state()
        artifact.update(
            {
                "state_event": state.last_event,
                "process_returncode": poll,
                "stdout_head": stdout_head,
                "stderr_head": stderr_head,
            }
        )
        self.close()
        return artifact


def _find_openttd() -> str | None:
    from_path = shutil.which("openttd")
    if from_path:
        return from_path
    candidates = [
        os.path.join(os.environ.get("ProgramFiles", ""), "OpenTTD", "openttd.exe"),
        os.path.join(os.environ.get("ProgramFiles(x86)", ""), "OpenTTD", "openttd.exe"),
        os.path.join(os.environ.get("LOCALAPPDATA", ""), "Programs", "OpenTTD", "openttd.exe"),
    ]
    for candidate in candidates:
        if candidate and os.path.exists(candidate):
            return candidate
    return None


def _config_text(seed: int) -> str:
    return f"""[misc]
display_opt = SHOW_TOWN_NAMES|SHOW_STATION_NAMES|SHOW_SIGNS|FULL_ANIMATION|FULL_DETAIL|WAYPOINTS

[network]
server_name = TycoonLE OpenTTD
server_port = 3979
server_advertise = false
max_clients = 1

[game_creation]
generation_seed = {seed}
map_x = 7
map_y = 7
landscape = temperate
starting_year = 1950

[difficulty]
max_no_competitors = 0
"""
