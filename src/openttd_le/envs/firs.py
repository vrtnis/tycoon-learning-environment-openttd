from __future__ import annotations

import os
import socket
import subprocess
import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any

from openttd_le import __version__
from openttd_le.backends.firs import render_firs_live_config, verify_firs_installed
from openttd_le.backends.live import (
    ADMIN_PASSWORD,
    AdminClient,
    FIRSReplSession,
    _benchmark_success,
    _firs_objective_done,
    _firs_reward_snapshot,
    _firs_step_reward,
    _new_run_dir,
    _next_observation,
    _popen_hidden,
    _set_openttd_user_dir,
    _terminate_process,
)
from openttd_le.backends.openttd import _find_openttd
from openttd_le.backends.visual import ensure_opengfx, install_live_bridge
from openttd_le.core.types import EnvError
from openttd_le.research.benchmarks import BenchmarkTask, select_task, task_to_workbook_meta
from openttd_le.research.scoring import score_snapshot
from openttd_le.workbooks.template import read_firs_ops_workbook


class OpenTTDFIRSEnv:
    """Farama-style environment core for real OpenTTD/FIRS.

    This class owns only the simulation lifecycle and step contract. It does not
    prompt an LLM or choose actions for the caller.
    """

    metadata = {"render_modes": ["external"], "render_fps": 15}

    def __init__(
        self,
        *,
        workbook: Path | str,
        task_id: str | None = None,
        benchmark_file: Path | str | None = None,
        executable: str | None = None,
        openttd_user_dir: Path | str | None = None,
        output_root: Path | str = "runs_openttd",
        seed: int | None = None,
        max_steps: int | None = None,
        step_timeout: float = 90.0,
        deterministic: bool = False,
        candidate_route_consider_limit: int = 5,
        candidate_plan_attempt_limit: int = 1,
        candidate_stop_after_feasible: int = 1,
        candidate_plan_timeout: float = 30.0,
    ) -> None:
        self.workbook_path = Path(workbook)
        self.task_id = task_id
        self.benchmark_file = benchmark_file
        self.executable = executable
        self.executable_path: str | None = None
        self.openttd_user_dir = openttd_user_dir
        self.output_root = Path(output_root)
        self.seed = seed
        self.max_steps_override = max_steps
        self.step_timeout = step_timeout
        self.deterministic = bool(deterministic)
        self.candidate_route_consider_limit = max(0, int(candidate_route_consider_limit))
        self.candidate_plan_attempt_limit = max(0, int(candidate_plan_attempt_limit))
        self.candidate_stop_after_feasible = max(0, int(candidate_stop_after_feasible))
        self.candidate_plan_timeout = max(0.1, float(candidate_plan_timeout))

        self.run_config: Any | None = None
        self.workbook_meta: dict[str, Any] = {}
        self.task: BenchmarkTask | None = None
        self.local_user_dir: Path | None = None
        self.install: Any | None = None
        self.installed: dict[str, str] = {}
        self.opengfx_path: Path | None = None
        self.run_dir: Path | None = None
        self.cfg_path: Path | None = None
        self.server_cmd: list[str] = []
        self.process_cwd: str | None = None
        self.game_port: int | None = None
        self.admin_port: int | None = None
        self.server: subprocess.Popen[bytes] | None = None
        self.admin: AdminClient | None = None
        self.session: FIRSReplSession | None = None
        self.observation: dict[str, Any] | None = None
        self.previous_snapshot: dict[str, Any] | None = None
        self.total_reward = 0.0
        self.executed_steps = 0
        self.completed = False
        self.failed = True
        self.max_steps = max_steps or 32
        self.runtime_lock: dict[str, Any] = {}
        self._candidate_cache_key: tuple[Any, ...] | None = None
        self._candidate_cache: list[dict[str, Any]] | None = None
        self.candidate_planning_summary: dict[str, Any] = {}

    def reset(self, *, seed: int | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
        self.close()
        self.local_user_dir = _set_openttd_user_dir(self.openttd_user_dir)
        run_config, workbook_meta = read_firs_ops_workbook(self.workbook_path)
        task = select_task(self.task_id, self.benchmark_file) if self.task_id else None
        if task is not None:
            workbook_meta = task_to_workbook_meta(task, workbook_meta)
            run_config = replace(
                run_config,
                seed=task.seed,
                economy=task.economy,
                target_chain=tuple(task.objectives) or run_config.target_chain,
            )
        resolved_seed = seed if seed is not None else self.seed
        if resolved_seed is not None:
            run_config = replace(run_config, seed=int(resolved_seed))

        exe = self.executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
        if not exe or not Path(exe).exists():
            raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")
        exe_path = Path(exe).expanduser().resolve()
        self.executable_path = str(exe_path)

        self.install = verify_firs_installed(self.local_user_dir)
        self.run_dir = _new_run_dir(self.output_root, suffix="firs_env")
        self.opengfx_path = ensure_opengfx()
        self.installed = install_live_bridge()
        game_port, admin_port = _allocate_distinct_ports()
        self.game_port = game_port
        self.admin_port = admin_port
        cfg_text = render_firs_live_config(
            run_config=run_config,
            install=self.install,
            game_port=game_port,
            admin_port=admin_port,
            admin_password=ADMIN_PASSWORD,
        )
        artifact_cfg_path = self.run_dir / "openttd.cfg"
        cfg_path = (
            self.local_user_dir / f"tycoonle-openttd-{self.run_dir.name}.cfg"
            if self.local_user_dir
            else artifact_cfg_path
        )
        cfg_path.write_text(cfg_text, encoding="ascii")
        if cfg_path != artifact_cfg_path:
            artifact_cfg_path.write_text(cfg_text, encoding="ascii")

        self.server_cmd = [
            str(exe_path),
            "-D",
            f"0.0.0.0:{game_port}",
            "-g",
            "-G",
            str(run_config.seed),
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
        ]
        self.process_cwd = str(self.local_user_dir or exe_path.parent)
        self.server = _popen_hidden(
            self.server_cmd,
            cwd=self.process_cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        self.admin = AdminClient("127.0.0.1", admin_port)
        self.admin.connect()
        self.admin.send_gamescript({"type": "observe"})
        observation = _next_observation(self.admin, timeout=90.0, reasons=("requested",))
        _assert_fresh_observation(observation)
        task_meta = workbook_meta.get("benchmark_task", {})
        self.session = FIRSReplSession(
            self.admin,
            observation,
            workbook_meta,
            run_config.vehicles_per_route,
            task_meta=task_meta,
        )
        self.run_config = run_config
        self.workbook_meta = workbook_meta
        self.task = task
        self.cfg_path = cfg_path
        self.observation = observation
        self.previous_snapshot = _firs_reward_snapshot(observation, workbook_meta)
        self.total_reward = 0.0
        self.executed_steps = 0
        self.completed = False
        self.failed = False
        self.max_steps = self.max_steps_override or (task.steps if task is not None else 32)
        self.runtime_lock = self._build_runtime_lock()
        if self.run_dir is not None:
            (self.run_dir / "runtime_lock.json").write_text(json.dumps(self.runtime_lock, indent=2), encoding="utf-8")
        self._candidate_cache_key = None
        self._candidate_cache = None
        self.candidate_planning_summary = {}
        return self._observation_with_candidates(observation), self._info(result=None, reward_details=None)

    def step(self, action: dict[str, Any]) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        if self.session is None or self.observation is None:
            raise RuntimeError("Environment must be reset before step().")
        before = self.previous_snapshot or _firs_reward_snapshot(self.observation, self.workbook_meta)
        self.session.observation = self.observation
        result = self.session._send_action(dict(action))
        self.observation = self.session.observation
        current = _firs_reward_snapshot(self.observation, self.workbook_meta)
        reward_details = _firs_step_reward(before, current, self.session.last_actions)
        reward = float(reward_details["reward"])
        self.total_reward += reward
        self.previous_snapshot = current
        self.executed_steps += 1
        self._candidate_cache_key = None
        self._candidate_cache = None
        self.completed = _firs_objective_done(self.observation, self.workbook_meta)
        if self.task is not None and _benchmark_success(current, self.task.success):
            self.completed = True
        action_error = str(result.get("error") or "") if isinstance(result, dict) else ""
        timed_out = action_error == "action_timeout_or_error"
        if timed_out:
            self.failed = True
        truncated = (timed_out or self.executed_steps >= self.max_steps) and not self.completed
        return (
            self._observation_with_candidates(self.observation),
            reward,
            self.completed,
            truncated,
            self._info(result=result, reward_details=reward_details, snapshot=current),
        )

    def candidate_actions(self) -> list[dict[str, Any]]:
        if self.session is None or self.observation is None:
            return []
        cache_key = self._candidate_cache_fingerprint()
        if self._candidate_cache_key == cache_key and self._candidate_cache is not None:
            return [dict(item) for item in self._candidate_cache]
        candidates: list[dict[str, Any]] = []
        routes = self.observation.get("routes", []) if self.observation else []
        if routes:
            candidates.append(
                {
                    "id": "wait_1_month",
                    "kind": "wait_months",
                    "feasible": True,
                    "action": {"type": "wait_months", "months": 1, "label": "wait for route progress"},
                }
            )
            for route in routes[:3]:
                route_id = route.get("route_id") or route.get("id")
                if route_id:
                    candidates.append(
                        {
                            "id": f"add_vehicle_{route_id}",
                            "kind": "add_vehicles",
                            "feasible": True,
                            "action": {"type": "add_vehicles", "route_id": route_id, "count": 1},
                        }
                    )
            self.candidate_planning_summary = {
                "mode": "route_management",
                "settings": self._candidate_planning_settings(),
                "candidate_routes_available": 0,
                "candidate_routes_considered": 0,
                "plan_attempts": 0,
                "proven_feasible": len(candidates),
                "proven_infeasible": 0,
                "unknown": 0,
                "stop_reason": "existing_routes",
            }
            self._candidate_cache_key = cache_key
            self._candidate_cache = [dict(item) for item in candidates]
            return candidates

        candidate_routes = self.session.candidate_routes()
        considered_routes = candidate_routes[: self.candidate_route_consider_limit]
        planned_count = 0
        feasible_count = 0
        proven_infeasible_count = 0
        unknown_count = 0
        stop_reason = "candidate_routes_exhausted"
        for index, route in enumerate(considered_routes):
            action = {
                "type": "build_cargo_route",
                "source_id": route["source_id"],
                "destination_id": route["destination_id"],
                "cargo_id": route["cargo_id"],
                "vehicles": int(getattr(self.run_config, "vehicles_per_route", 5)),
                "physical": True,
                "max_path_tiles": 256,
                "station_candidate_limit": 8,
                "allow_virtual": False,
                "preview_roads": False,
                "label": f"candidate_route_{index + 1}",
            }
            if self.candidate_stop_after_feasible and feasible_count >= self.candidate_stop_after_feasible:
                plan = self._unknown_candidate_plan(action, "sufficient_feasible_candidates")
            elif planned_count >= self.candidate_plan_attempt_limit:
                plan = self._unknown_candidate_plan(action, "plan_attempt_limit")
            else:
                plan = self._plan_candidate_action(action)
                planned_count += 1
            feasibility = self._candidate_feasibility(plan)
            feasible = feasibility == "proven_feasible"
            if feasible:
                feasible_count += 1
            elif feasibility == "proven_infeasible":
                proven_infeasible_count += 1
            else:
                unknown_count += 1
            candidates.append(
                {
                    "id": f"build_route_{index + 1}",
                    "kind": "build_cargo_route",
                    "route": route,
                    "action": action,
                    "feasible": feasible,
                    "feasibility": feasibility,
                    "plan": plan,
                    "diagnostics": [] if feasible else [str(plan.get("error") or feasibility)],
                }
            )
        if self.candidate_stop_after_feasible and feasible_count >= self.candidate_stop_after_feasible:
            stop_reason = "sufficient_feasible_candidates"
        elif planned_count >= self.candidate_plan_attempt_limit and len(considered_routes) < len(candidate_routes):
            stop_reason = "plan_attempt_limit"
        candidates.sort(
            key=lambda item: (
                0 if item.get("feasible", True) else 1,
                0 if item.get("feasibility") == "proven_infeasible" else 1,
                int((item.get("plan") or {}).get("path_tiles", 999999) or 999999),
                -float((item.get("route") or {}).get("production", 0) or 0),
                int((item.get("route") or {}).get("distance", 999999) or 999999),
                int((item.get("route") or {}).get("source_id", 999999) or 999999),
                int((item.get("route") or {}).get("destination_id", 999999) or 999999),
            )
        )
        self.candidate_planning_summary = {
            "mode": "route_building",
            "settings": self._candidate_planning_settings(),
            "candidate_routes_available": len(candidate_routes),
            "candidate_routes_considered": len(considered_routes),
            "plan_attempts": planned_count,
            "proven_feasible": feasible_count,
            "proven_infeasible": proven_infeasible_count,
            "unknown": unknown_count,
            "stop_reason": stop_reason,
        }
        self._candidate_cache_key = cache_key
        self._candidate_cache = [dict(item) for item in candidates]
        return candidates

    def _candidate_cache_fingerprint(self) -> tuple[Any, ...]:
        routes = self.observation.get("routes", []) if self.observation else []
        route_state = tuple(
            (
                route.get("route_id"),
                route.get("source_id"),
                route.get("destination_id"),
                route.get("cargo_id"),
                route.get("delivered"),
                route.get("vehicles"),
            )
            for route in routes
        )
        return (self.observation.get("tick") if self.observation else None, route_state)

    def _plan_candidate_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if self.session is None:
            return {"feasible": False, "feasibility": "unknown_environment_not_ready", "error": "environment_not_ready"}
        try:
            plan = self.session.plan_cargo_route_action(action, timeout=self.candidate_plan_timeout)
        except Exception as exc:
            return {
                "feasible": False,
                "feasibility": "unknown_plan_timeout_or_error",
                "error": "route_plan_timeout_or_error",
                "detail": str(exc),
            }
        plan = dict(plan)
        plan["feasibility"] = self._candidate_feasibility(plan)
        return plan

    def _unknown_candidate_plan(self, action: dict[str, Any], reason: str) -> dict[str, Any]:
        return {
            "action_type": "plan_cargo_route",
            "cargo_id": action.get("cargo_id"),
            "destination_id": action.get("destination_id"),
            "source_id": action.get("source_id"),
            "feasible": False,
            "feasibility": f"unknown_{reason}",
            "error": f"candidate_planning_{reason}",
        }

    def _candidate_feasibility(self, plan: dict[str, Any]) -> str:
        explicit = str(plan.get("feasibility") or "")
        if explicit:
            return explicit
        if bool(plan.get("feasible")) and not plan.get("error"):
            return "proven_feasible"
        return "proven_infeasible"

    def _candidate_planning_settings(self) -> dict[str, Any]:
        return {
            "route_consider_limit": self.candidate_route_consider_limit,
            "plan_attempt_limit": self.candidate_plan_attempt_limit,
            "stop_after_feasible": self.candidate_stop_after_feasible,
            "plan_timeout_seconds": self.candidate_plan_timeout,
        }

    def close(self) -> None:
        if self.admin is not None:
            self.admin.close()
        self.admin = None
        _terminate_process(self.server)
        self.server = None

    def summary(self, *, agent: str, model: str | None) -> dict[str, Any]:
        final_observation = self.observation or {}
        routes = final_observation.get("routes", []) or []
        route_profit = sum(float(route.get("profit", 0) or route.get("vehicle_profit", 0) or 0) for route in routes)
        final_snapshot = _firs_reward_snapshot(final_observation, self.workbook_meta)
        completed = self.completed
        if self.task is not None and _benchmark_success(final_snapshot, self.task.success):
            completed = True
        return {
            "objective": "openttd_firs_env",
            "completed": completed,
            "failed": self.failed,
            "executed_steps": self.executed_steps,
            "requested_steps": self.max_steps,
            "agent": agent,
            "model": model,
            "backend": "openttd",
            "seed": getattr(self.run_config, "seed", None),
            "economy": getattr(self.run_config, "economy", None),
            "firs_newgrf": str(self.install.newgrf_path) if self.install else None,
            "firs_version": _firs_version_from_path(self.install.newgrf_path) if self.install else None,
            "openttd_executable": self.executable_path,
            "openttd_le_version": __version__,
            "deterministic": self.deterministic,
            "runtime": self.runtime_lock,
            "openttd_user_dir": str(self.local_user_dir) if self.local_user_dir else None,
            "game_port": self.game_port,
            "admin_port": self.admin_port,
            "final_tick": final_observation.get("tick"),
            "final_bank_balance": final_observation.get("bank_balance"),
            "route_profit": route_profit,
            "total_reward": round(self.total_reward, 3),
            "milestones": final_snapshot.get("milestones", {}),
            "final_score": score_snapshot(final_observation, self.workbook_meta.get("objectives", [])),
            "benchmark_task": self.task.id if self.task else self.task_id,
            "benchmark_mode": self.task.mode if self.task else None,
            "success_criteria": self.task.success if self.task else None,
            "routes": routes,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "workbook": str(self.workbook_path),
            "note": "Farama-style OpenTTD/FIRS env run: agent and environment are separated.",
        }

    def launch_info(self, *, summary_path: Path | None = None) -> dict[str, Any]:
        return {
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "server_pid": self.server.pid if self.server is not None else None,
            "model": None,
            "steps": self.max_steps,
            "summary": str(summary_path) if summary_path else None,
            "installed": self.installed,
            "firs_newgrf": str(self.install.newgrf_path) if self.install else None,
            "firs_version": _firs_version_from_path(self.install.newgrf_path) if self.install else None,
            "openttd_executable": self.executable_path,
            "openttd_le_version": __version__,
            "deterministic": self.deterministic,
            "runtime": self.runtime_lock,
            "openttd_user_dir": str(self.local_user_dir) if self.local_user_dir else None,
            "game_port": self.game_port,
            "admin_port": self.admin_port,
            "server_command": self.server_cmd,
        }

    def _observation_with_candidates(self, observation: dict[str, Any]) -> dict[str, Any]:
        enriched = dict(observation)
        enriched["candidate_actions"] = self.candidate_actions()
        enriched["candidate_planning"] = dict(self.candidate_planning_summary)
        enriched["task"] = self.workbook_meta.get("benchmark_task", {})
        enriched["workbook"] = {
            "scenario": self.workbook_meta.get("fields", {}),
            "objectives": self.workbook_meta.get("objectives", []),
        }
        return enriched

    def _info(
        self,
        *,
        result: dict[str, Any] | None,
        reward_details: dict[str, Any] | None,
        snapshot: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        return {
            "result": result,
            "actions": list(self.session.last_actions) if self.session is not None else [],
            "candidate_actions": self.candidate_actions(),
            "reward_details": reward_details,
            "snapshot": snapshot,
            "run_dir": str(self.run_dir) if self.run_dir else None,
            "deterministic": self.deterministic,
            "candidate_planning": dict(self.candidate_planning_summary),
            "task": self.workbook_meta.get("benchmark_task", {}),
        }

    def _build_runtime_lock(self) -> dict[str, Any]:
        company_ai_dir = self.installed.get("company_ai_dir") if self.installed else None
        gamescript_dir = self.installed.get("gamescript_dir") if self.installed else None
        bridge_root = Path(__file__).resolve().parents[3] / "openttd_bridge"
        benchmark_file = Path(self.benchmark_file) if self.benchmark_file else _default_benchmark_file()
        return {
            "schema": "openttd-le-runtime-lock-v2",
            "openttd_le_version": __version__,
            "python_version": sys.version.split()[0],
            "openttd_executable": self.executable_path,
            "openttd_executable_sha256": _file_sha256(Path(self.executable_path)) if self.executable_path else None,
            "opengfx_baseset": str(self.opengfx_path) if self.opengfx_path else None,
            "opengfx_baseset_sha256": _file_sha256(self.opengfx_path),
            "firs_newgrf": str(self.install.newgrf_path) if self.install else None,
            "firs_newgrf_sha256": _file_sha256(self.install.newgrf_path) if self.install else None,
            "firs_version": _firs_version_from_path(self.install.newgrf_path) if self.install else None,
            "workbook": str(self.workbook_path),
            "workbook_sha256": _file_sha256(self.workbook_path),
            "benchmark_file": str(benchmark_file) if benchmark_file else None,
            "benchmark_file_sha256": _file_sha256(benchmark_file) if benchmark_file else None,
            "cfg": str(self.cfg_path) if self.cfg_path else None,
            "cfg_sha256": _file_sha256(self.cfg_path),
            "cfg_effective_sha256": _normalized_cfg_sha256(self.cfg_path),
            "server_command": list(self.server_cmd),
            "server_command_effective": _normalized_server_command(self.server_cmd),
            "process_cwd": self.process_cwd,
            "company_ai_sha256": _directory_sha256(Path(company_ai_dir)) if company_ai_dir else None,
            "gamescript_sha256": _directory_sha256(Path(gamescript_dir)) if gamescript_dir else None,
            "company_ai_source_sha256": _directory_sha256(bridge_root / "OpenTTDLECompany"),
            "gamescript_source_sha256": _directory_sha256(bridge_root / "OpenTTDLEGameScript"),
            "seed": getattr(self.run_config, "seed", None),
            "economy": getattr(self.run_config, "economy", None),
            "map_x": getattr(self.run_config, "map_x", None),
            "map_y": getattr(self.run_config, "map_y", None),
            "landscape": getattr(self.run_config, "landscape", None),
            "starting_year": getattr(self.run_config, "starting_year", None),
            "target_chain": list(getattr(self.run_config, "target_chain", []) or []),
            "task_id": self.task.id if self.task else self.task_id,
            "gym_env_id": "OpenTTD-FIRS-Deterministic-v0" if self.deterministic else "OpenTTD-FIRS-Lab-v0",
            "deterministic": self.deterministic,
            "candidate_planning_settings": self._candidate_planning_settings(),
        }


def _allocate_distinct_ports() -> tuple[int, int]:
    sockets: list[socket.socket] = []
    ports: list[int] = []
    try:
        for _ in range(2):
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.bind(("0.0.0.0", 0))
            sockets.append(sock)
            ports.append(int(sock.getsockname()[1]))
        return ports[0], ports[1]
    finally:
        for sock in sockets:
            sock.close()


def _assert_fresh_observation(observation: dict[str, Any]) -> None:
    routes = observation.get("routes", []) or []
    tick = int(observation.get("tick", 0) or 0)
    if routes or tick > 50_000:
        raise EnvError(
            "OpenTTD/FIRS reset received a stale world state. "
            "The new environment expected a fresh game with no routes; "
            f"got routes={len(routes)} tick={tick}."
        )


def _firs_version_from_path(path: Path) -> str | None:
    for part in path.parts:
        if part.lower().startswith("firs_industries_"):
            return part.removeprefix("FIRS_Industries_")
    return None


def _file_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _directory_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_dir():
        return None
    digest = hashlib.sha256()
    for file_path in sorted(item for item in path.rglob("*") if item.is_file()):
        digest.update(file_path.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        file_digest = _file_sha256(file_path)
        if file_digest:
            digest.update(bytes.fromhex(file_digest))
    return digest.hexdigest()


def _normalized_cfg_sha256(path: Path | None) -> str | None:
    if path is None or not path.exists() or not path.is_file():
        return None
    lines = []
    for line in path.read_text(encoding="ascii").splitlines():
        if line.startswith("server_port = "):
            lines.append("server_port = <ephemeral>")
        elif line.startswith("server_admin_port = "):
            lines.append("server_admin_port = <ephemeral>")
        else:
            lines.append(line)
    return hashlib.sha256(("\n".join(lines) + "\n").encode("ascii")).hexdigest()


def _normalized_server_command(command: list[str]) -> list[str]:
    normalized: list[str] = []
    skip_cfg_path = False
    for index, item in enumerate(command):
        if index == 0:
            normalized.append("<openttd>")
            continue
        if skip_cfg_path:
            normalized.append("<cfg>")
            skip_cfg_path = False
            continue
        if item == "-c":
            normalized.append(item)
            skip_cfg_path = True
            continue
        if item.startswith("0.0.0.0:"):
            normalized.append("0.0.0.0:<ephemeral>")
            continue
        normalized.append(item)
    return normalized


def _default_benchmark_file() -> Path | None:
    candidate = Path(__file__).resolve().parents[3] / "scenarios" / "firs_benchmarks.json"
    return candidate if candidate.exists() else None
