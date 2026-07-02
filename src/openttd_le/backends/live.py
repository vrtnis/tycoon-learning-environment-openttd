from __future__ import annotations

import builtins
import contextlib
import io
import json
import os
import re
import shutil
import socket
import struct
import subprocess
import time
import traceback
import urllib.request
from dataclasses import asdict, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from openttd_le.backends.firs import render_firs_live_config, verify_firs_installed
from openttd_le.backends.openttd import _find_openttd
from openttd_le.backends.visual import ensure_opengfx, install_live_bridge
from openttd_le.core.types import EnvError
from openttd_le.research.api import Prototype, api_from_observation, get_cargo_chains, get_finance, get_industries, get_routes
from openttd_le.research.benchmarks import (
    ROUTE_BUILDER_INFEASIBLE_REASONS,
    aggregate_route_builder_attempts,
    aggregate_runs,
    select_task,
    task_to_workbook_meta,
)
from openttd_le.research.scoring import CARGO_VALUE, score_snapshot
from openttd_le.replay import export_replay, load_replay, replay_actions
from openttd_le.workbooks.export import export_run_to_xlsx
from openttd_le.workbooks.template import read_firs_ops_workbook


ADMIN_PASSWORD = "openttdle"


class AdminClient:
    def __init__(self, host: str, port: int, password: str = ADMIN_PASSWORD) -> None:
        self.host = host
        self.port = port
        self.password = password
        self.sock: socket.socket | None = None

    def connect(self, timeout: float = 30.0) -> None:
        deadline = time.time() + timeout
        last_error: OSError | None = None
        while time.time() < deadline:
            try:
                self.sock = socket.create_connection((self.host, self.port), timeout=3.0)
                self.sock.settimeout(1.0)
                self._send(0, _string(self.password) + _string("tycoonle-openttd") + _string("0.1"))
                self._wait_for_types({103, 104}, timeout=10.0)
                self._send(2, struct.pack("<HH", 9, 1 << 6))
                return
            except OSError as exc:
                last_error = exc
                time.sleep(0.5)
        raise EnvError(f"Could not connect to OpenTTD admin port: {last_error}")

    def send_gamescript(self, payload: dict[str, Any]) -> None:
        self._send(6, _string(json.dumps(payload, separators=(",", ":"))))

    def read_gamescript(self, timeout: float = 30.0) -> dict[str, Any]:
        deadline = time.time() + timeout
        while time.time() < deadline:
            packet_type, payload = self._read_packet(deadline - time.time())
            if packet_type == 124:
                text = _read_string(payload)
                return json.loads(text)
            if packet_type == 102:
                raise EnvError(f"OpenTTD admin error packet: {payload!r}")
        raise EnvError("Timed out waiting for GameScript message.")

    def close(self) -> None:
        if self.sock is not None:
            try:
                self._send(1, b"")
            except OSError:
                pass
            self.sock.close()
        self.sock = None

    def _wait_for_types(self, types: set[int], timeout: float) -> None:
        deadline = time.time() + timeout
        seen: set[int] = set()
        while time.time() < deadline:
            packet_type, _ = self._read_packet(deadline - time.time())
            seen.add(packet_type)
            if types.issubset(seen):
                return
        raise EnvError(f"Timed out during admin login. Saw packets: {sorted(seen)}")

    def _send(self, packet_type: int, payload: bytes) -> None:
        if self.sock is None:
            raise EnvError("Admin client is not connected.")
        size = 3 + len(payload)
        self.sock.sendall(struct.pack("<HB", size, packet_type) + payload)

    def _read_packet(self, timeout: float) -> tuple[int, bytes]:
        if self.sock is None:
            raise EnvError("Admin client is not connected.")
        self.sock.settimeout(max(0.1, timeout))
        header = _recv_exact(self.sock, 3)
        size, packet_type = struct.unpack("<HB", header)
        if size < 3:
            raise EnvError(f"Invalid admin packet size: {size}")
        return packet_type, _recv_exact(self.sock, size - 3)


SAFE_REPL_BUILTINS = {
    "__build_class__": builtins.__build_class__,
    "print": builtins.print,
    "len": builtins.len,
    "range": builtins.range,
    "enumerate": builtins.enumerate,
    "zip": builtins.zip,
    "min": builtins.min,
    "max": builtins.max,
    "sum": builtins.sum,
    "sorted": builtins.sorted,
    "list": builtins.list,
    "dict": builtins.dict,
    "set": builtins.set,
    "tuple": builtins.tuple,
    "str": builtins.str,
    "int": builtins.int,
    "float": builtins.float,
    "bool": builtins.bool,
    "abs": builtins.abs,
    "any": builtins.any,
    "all": builtins.all,
    "isinstance": builtins.isinstance,
    "object": builtins.object,
    "Exception": builtins.Exception,
    "ValueError": builtins.ValueError,
}


class FIRSReplSession:
    def __init__(
        self,
        admin: AdminClient,
        observation: dict[str, Any],
        workbook_meta: dict[str, Any],
        vehicles_per_route: int,
        task_meta: dict[str, Any] | None = None,
    ) -> None:
        self.admin = admin
        self.observation = observation
        self.workbook_meta = workbook_meta
        self.task_meta = task_meta or {}
        self.vehicles_per_route = vehicles_per_route
        self.last_stdout = ""
        self.last_stderr = ""
        self.last_actions: list[dict[str, Any]] = []
        self.failed_route_keys: set[tuple[Any, Any, Any]] = set()
        self.virtual_route_counter = 1
        self.env: dict[str, Any] = {
            "__builtins__": SAFE_REPL_BUILTINS,
            "__name__": "openttd_le_firs_repl",
        }
        self._install_helpers()
        self._refresh_env()

    def _install_helpers(self) -> None:
        self.env.update(
            {
                "observe": self.observe,
                "build_cargo_route": self.build_cargo_route,
                "add_vehicles": self.add_vehicles,
                "wait_months": self.wait_months,
                "inspect_bottlenecks": self.inspect_bottlenecks,
                "borrow_or_repay": self.borrow_or_repay,
                "get_industries": self.get_industries,
                "get_cargo_chains": self.get_cargo_chains,
                "get_routes": self.get_routes,
                "get_finance": self.get_finance,
                "short_routes": self.short_routes,
            }
        )

    def candidate_routes(self) -> list[dict[str, Any]]:
        graph = self.observation.get("industry_graph", [])
        candidates = _candidate_firs_pairs(graph, self.workbook_meta.get("objectives", []), limit=12)
        if not candidates:
            candidates = _candidate_firs_pairs_from_io(
                self.observation,
                self.workbook_meta.get("objectives", []),
                limit=12,
            )
        if not candidates:
            candidates = _candidate_open_play_pairs(self.observation, limit=12)
        routes = self.observation.get("routes", [])
        if routes:
            candidates = [pair for pair in candidates if not _route_already_registered(pair, routes)]
        candidates = [pair for pair in candidates if _route_key(pair) not in self.failed_route_keys]
        if not candidates:
            candidates = [
                pair
                for pair in _candidate_open_play_pairs(self.observation, limit=12)
                if _route_key(pair) not in self.failed_route_keys
            ]
        return candidates

    def _refresh_env(self) -> None:
        self.env.update(
            {
                "obs": self.observation,
                "routes": self.observation.get("routes", []),
                "candidate_routes": self.candidate_routes(),
                "workbook": {
                    "scenario": self.workbook_meta.get("fields", {}),
                    "objectives": self.workbook_meta.get("objectives", []),
                },
                "task": self.task_meta,
                "Prototype": Prototype,
                "api": api_from_observation(self.observation, CARGO_VALUE),
                "industries": get_industries(self.observation),
                "cargo_chains": get_cargo_chains(self.observation, CARGO_VALUE),
                "finance": get_finance(self.observation),
                "last_stdout": self.last_stdout,
                "last_stderr": self.last_stderr,
            }
        )

    def execute(self, code: str) -> dict[str, Any]:
        self.last_actions = []
        self._refresh_env()
        stdout = io.StringIO()
        stderr = io.StringIO()
        with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
            try:
                exec(code, self.env, self.env)
            except Exception:
                traceback.print_exc()
        self.last_stdout = stdout.getvalue()
        self.last_stderr = stderr.getvalue()
        self._refresh_env()
        return {
            "stdout": self.last_stdout,
            "stderr": self.last_stderr,
            "actions": list(self.last_actions),
            "observation": self.observation,
        }

    def _send_action(self, action: dict[str, Any]) -> dict[str, Any]:
        if action.get("type") == "build_cargo_route" and action.get("allow_virtual") and not action.get("preview_roads"):
            result = self._create_python_virtual_route(action)
            self.last_actions.append({"action": action, "result": result, "observation": self.observation})
            return result
        if action.get("type") == "wait_months" and self._has_python_virtual_routes():
            result = self._advance_python_virtual_routes(action)
            self.last_actions.append({"action": action, "result": result, "observation": self.observation})
            return result
        self.admin.send_gamescript({"type": "action", "action": action})
        try:
            result = _next_result(self.admin, timeout=90.0)
            observation = _next_observation(self.admin, timeout=90.0, reasons=("after_action",))
        except Exception as exc:
            result = {
                "type": "result",
                "step": self.observation.get("step"),
                "action_type": action.get("type"),
                "error": "action_timeout_or_error",
                "detail": str(exc),
            }
            if action.get("type") == "build_cargo_route":
                self.failed_route_keys.add(_route_key(action))
            self.last_actions.append({"action": action, "result": result, "observation": self.observation})
            return result
        if action.get("type") == "build_cargo_route" and result.get("error"):
            self.failed_route_keys.add(_route_key(action))
        self.observation = observation
        self.last_actions.append({"action": action, "result": result, "observation": observation})
        return result

    def observe(self) -> dict[str, Any]:
        self.admin.send_gamescript({"type": "observe"})
        self.observation = _next_observation(self.admin, timeout=90.0, reasons=("requested",))
        self._refresh_env()
        return self.observation

    def build_cargo_route(
        self,
        source_id: int,
        destination_id: int,
        cargo_id: int,
        vehicles: int | None = None,
        physical: bool = True,
        max_path_tiles: int = 256,
        allow_virtual: bool = False,
        preview_roads: bool = False,
        label: str = "",
    ) -> dict[str, Any]:
        action = {
            "type": "build_cargo_route",
            "source_id": int(source_id),
            "destination_id": int(destination_id),
            "cargo_id": int(cargo_id),
            "vehicles": int(vehicles if vehicles is not None else self.vehicles_per_route),
            "physical": bool(physical),
            "max_path_tiles": int(max_path_tiles),
            "allow_virtual": bool(allow_virtual),
            "preview_roads": bool(preview_roads),
            "label": label or "repl physical cargo route",
        }
        return self._send_action(action)

    def add_vehicles(self, route_id: str, count: int) -> dict[str, Any]:
        return self._send_action({"type": "add_vehicles", "route_id": str(route_id), "count": int(count)})

    def wait_months(self, months: int, label: str = "") -> dict[str, Any]:
        return self._send_action({"type": "wait_months", "months": int(months), "label": label or "repl wait"})

    def inspect_bottlenecks(self) -> dict[str, Any]:
        return self._send_action({"type": "inspect_bottlenecks"})

    def borrow_or_repay(self, amount: int) -> dict[str, Any]:
        return self._send_action({"type": "borrow_or_repay", "amount": int(amount)})

    def _create_python_virtual_route(self, action: dict[str, Any]) -> dict[str, Any]:
        source_id = int(action.get("source_id", -1))
        destination_id = int(action.get("destination_id", -1))
        cargo_id = int(action.get("cargo_id", -1))
        source = self._industry_lookup(source_id)
        destination = self._industry_lookup(destination_id)
        cargo = self._cargo_lookup(cargo_id)
        route_id = f"py_route_{self.virtual_route_counter:03d}"
        self.virtual_route_counter += 1
        route = {
            "route_id": route_id,
            "cargo_id": cargo_id,
            "cargo_label": cargo["cargo_label"],
            "cargo_name": cargo["cargo_name"],
            "source_id": source_id,
            "source_name": source,
            "destination_id": destination_id,
            "destination_name": destination,
            "source_station": -1,
            "destination_station": -1,
            "vehicles": int(action.get("vehicles", self.vehicles_per_route) or self.vehicles_per_route),
            "delivered": 0,
            "profit": 0,
            "source_waiting": 0,
            "source_rating": 100,
            "is_virtual": True,
            "virtual_delivery_rate": 12,
            "virtual_profit_rate": 1800,
            "vehicle_details": [],
        }
        routes = list(self.observation.get("routes", []) or [])
        routes.append(route)
        self.observation = {**self.observation, "routes": routes}
        return {
            "type": "result",
            "step": self.observation.get("step"),
            "action_type": action.get("type"),
            "route_id": route_id,
            "mode": "python_virtual_operational_route",
            "warning": "research_virtual_fallback",
            "cargo_label": route["cargo_label"],
            "cargo_name": route["cargo_name"],
            "source_id": source_id,
            "source_name": source,
            "destination_id": destination_id,
            "destination_name": destination,
            "vehicles": route["vehicles"],
            "virtual_delivery_rate": route["virtual_delivery_rate"],
            "virtual_profit_rate": route["virtual_profit_rate"],
        }

    def _advance_python_virtual_routes(self, action: dict[str, Any]) -> dict[str, Any]:
        months = max(1, int(action.get("months", 1) or 1))
        routes = []
        delivered = 0
        profit = 0
        for route in self.observation.get("routes", []) or []:
            route = dict(route)
            if route.get("is_virtual"):
                add_delivery = int(route.get("virtual_delivery_rate", 0) or 0) * months
                add_profit = int(route.get("virtual_profit_rate", 0) or 0) * months
                route["delivered"] = int(route.get("delivered", 0) or 0) + add_delivery
                route["profit"] = float(route.get("profit", 0) or 0) + add_profit
                delivered += add_delivery
                profit += add_profit
            routes.append(route)
        self.observation = {
            **self.observation,
            "routes": routes,
            "tick": int(self.observation.get("tick", 0) or 0) + months * 2220,
        }
        return {
            "type": "result",
            "step": self.observation.get("step"),
            "action_type": "wait_months",
            "mode": "python_virtual_wait",
            "months": months,
            "delivered": delivered,
            "profit": profit,
            "routes": routes,
        }

    def _has_python_virtual_routes(self) -> bool:
        return any(route.get("is_virtual") for route in self.observation.get("routes", []) or [])

    def _industry_lookup(self, industry_id: int) -> str:
        for field in ("industry_graph", "industry_inputs", "industry_outputs"):
            for item in self.observation.get(field, []) or []:
                if item.get("source_id") == industry_id:
                    return str(item.get("source_name", item.get("source_type", industry_id)))
                if item.get("destination_id") == industry_id:
                    return str(item.get("destination_name", item.get("destination_type", industry_id)))
                if item.get("industry_id") == industry_id:
                    return str(item.get("industry_name", industry_id))
        return str(industry_id)

    def _cargo_lookup(self, cargo_id: int) -> dict[str, str]:
        for field in ("industry_graph", "industry_inputs", "industry_outputs"):
            for item in self.observation.get(field, []) or []:
                if item.get("cargo_id") == cargo_id:
                    label = str(item.get("cargo_label", item.get("cargo", cargo_id))).upper()
                    return {"cargo_label": label, "cargo_name": str(item.get("cargo_name", label))}
        label = str(cargo_id)
        return {"cargo_label": label, "cargo_name": label}

    def get_industries(self) -> list[Any]:
        return get_industries(self.observation)

    def get_cargo_chains(self) -> list[Any]:
        return get_cargo_chains(self.observation, CARGO_VALUE)

    def get_routes(self) -> list[Any]:
        return get_routes(self.observation, CARGO_VALUE)

    def get_finance(self) -> Any:
        return get_finance(self.observation)

    def short_routes(self, max_distance: int = 40, cargo: str | None = None) -> list[dict[str, Any]]:
        pairs = self.candidate_routes()
        if cargo:
            pairs = [pair for pair in pairs if str(pair.get("cargo_label", pair.get("cargo", ""))).upper() == cargo.upper()]
        return sorted(
            [pair for pair in pairs if int(pair.get("distance", max_distance + 1) or max_distance + 1) <= max_distance],
            key=lambda pair: int(pair.get("distance", 999999) or 999999),
        )


def launch_gpt_live(
    *,
    executable: str | None = None,
    output_root: Path | str = "runs_live",
    model: str = "gpt-5.5",
    seed: int = 1,
    steps: int = 4,
    resolution: str = "1280x800",
    allow_heuristic: bool = False,
    focus_town_id: int | None = None,
    start_delay: float = 8.0,
    step_delay: float = 4.0,
) -> dict[str, Any]:
    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")

    if not os.environ.get("OPENAI_API_KEY") and not allow_heuristic:
        raise EnvError("OPENAI_API_KEY is required for live GPT play. Use --allow-heuristic only for bridge testing.")

    run_dir = _new_run_dir(Path(output_root))
    ensure_opengfx()
    installed = install_live_bridge()
    game_port, admin_port = _find_distinct_free_ports(3979, 3977)
    cfg_path = run_dir / "openttd.cfg"
    cfg_text = _live_config(seed, game_port, admin_port)
    cfg_path.write_text(cfg_text, encoding="ascii")
    client_cfg_path = run_dir / "openttd-viewer.cfg"
    client_cfg_path.write_text(_with_client_name(cfg_text, f"TycoonLE OpenTTD Viewer {game_port}"), encoding="ascii")

    server_cmd = [
        str(exe),
        "-D",
        f"0.0.0.0:{game_port}",
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
    ]
    server = _popen_hidden(
        server_cmd,
        cwd=str(Path(exe).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    client_cmd = [
        str(exe),
        "-c",
        str(client_cfg_path),
        "-x",
        "-X",
        "-n",
        f"127.0.0.1:{game_port}#0",
        "-I",
        "OpenGFX",
        "-S",
        "NoSound",
        "-M",
        "NoMusic",
        "-r",
        resolution,
    ]
    client: subprocess.Popen[bytes] | None = None

    trace_path = run_dir / "live_trace.jsonl"
    admin = AdminClient("127.0.0.1", admin_port)
    try:
        admin.connect()
        admin.send_gamescript({"type": "observe"})
        observation = _next_observation(admin, timeout=60.0)
        client = subprocess.Popen(
            client_cmd,
            cwd=str(Path(exe).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        time.sleep(start_delay)
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"event": "initial_observation", "data": observation}) + "\n")
            for index in range(steps):
                action = _choose_action(
                    observation,
                    model=model,
                    allow_heuristic=allow_heuristic,
                    focus_town_id=focus_town_id,
                )
                trace.write(json.dumps({"event": "action", "step": index + 1, "data": action}) + "\n")
                admin.send_gamescript({"type": "action", "action": action})
                result = _next_result(admin, timeout=60.0)
                observation = _next_observation(admin, timeout=60.0)
                trace.write(json.dumps({"event": "result", "step": index + 1, "data": result}) + "\n")
                trace.write(json.dumps({"event": "observation", "step": index + 1, "data": observation}) + "\n")
                time.sleep(step_delay)
    finally:
        admin.close()

    launch_info = {
        "run_dir": str(run_dir),
        "server_pid": server.pid,
        "client_pid": client.pid if client is not None else None,
        "game_port": game_port,
        "admin_port": admin_port,
        "model": model,
        "steps": steps,
        "focus_town_id": focus_town_id,
        "trace": str(trace_path),
        "installed": installed,
        "server_command": server_cmd,
        "client_command": client_cmd,
        "note": "The visible client remains open after the GPT action loop finishes.",
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
    return launch_info


def launch_coal_objective(
    *,
    executable: str | None = None,
    output_root: Path | str = "runs_coal",
    model: str = "gpt-5.5",
    seed: int = 1,
    steps: int = 6,
    resolution: str = "1280x800",
    allow_heuristic: bool = False,
    start_delay: float = 10.0,
    step_delay: float = 3.0,
) -> dict[str, Any]:
    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")

    if not os.environ.get("OPENAI_API_KEY") and not allow_heuristic:
        raise EnvError("OPENAI_API_KEY is required for GPT coal objective play. Use --allow-heuristic for bridge testing.")

    run_dir = _new_run_dir(Path(output_root), suffix="coal_objective")
    ensure_opengfx()
    installed = install_live_bridge()
    game_port, admin_port = _find_distinct_free_ports(3979, 3977)
    cfg_path = run_dir / "openttd.cfg"
    cfg_text = _live_config(seed, game_port, admin_port)
    cfg_path.write_text(cfg_text, encoding="ascii")
    client_cfg_path = run_dir / "openttd-viewer.cfg"
    client_cfg_path.write_text(_with_client_name(cfg_text, f"TycoonLE OpenTTD Viewer {game_port}"), encoding="ascii")

    server_cmd = [
        str(exe),
        "-D",
        f"0.0.0.0:{game_port}",
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
    ]
    server = _popen_hidden(
        server_cmd,
        cwd=str(Path(exe).parent),
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    client_cmd = [
        str(exe),
        "-c",
        str(client_cfg_path),
        "-x",
        "-X",
        "-n",
        f"127.0.0.1:{game_port}#0",
        "-I",
        "OpenGFX",
        "-S",
        "NoSound",
        "-M",
        "NoMusic",
        "-r",
        resolution,
    ]

    trace_path = run_dir / "coal_trace.jsonl"
    summary_path = run_dir / "summary.json"
    client: subprocess.Popen[bytes] | None = None
    admin = AdminClient("127.0.0.1", admin_port)
    executed_steps = 0
    completed = False
    final_observation: dict[str, Any] | None = None
    failed = True
    try:
        admin.connect()
        admin.send_gamescript({"type": "observe"})
        observation = _next_observation(admin, timeout=60.0)
        final_observation = observation
        client = subprocess.Popen(
            client_cmd,
            cwd=str(Path(exe).parent),
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        time.sleep(start_delay)
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"event": "initial_observation", "data": observation}) + "\n")
            for index in range(steps):
                action = _choose_coal_action(observation, model=model, allow_heuristic=allow_heuristic)
                trace.write(json.dumps({"event": "action", "step": index + 1, "data": action}) + "\n")
                admin.send_gamescript({"type": "action", "action": action})
                result = _next_result(admin, timeout=120.0)
                observation = _next_observation(admin, timeout=120.0)
                final_observation = observation
                executed_steps = index + 1
                trace.write(json.dumps({"event": "result", "step": index + 1, "data": result}) + "\n")
                trace.write(json.dumps({"event": "observation", "step": index + 1, "data": observation}) + "\n")
                completed = _coal_objective_done(observation)
                if completed:
                    break
                time.sleep(step_delay)
    finally:
        admin.close()

    final_objective = final_observation.get("active_objective") if final_observation else None
    summary = {
        "objective": "first_coal_delivery",
        "completed": completed,
        "executed_steps": executed_steps,
        "requested_steps": steps,
        "final_tick": final_observation.get("tick") if final_observation else None,
        "final_bank_balance": final_observation.get("bank_balance") if final_observation else None,
        "final_objective": final_objective,
        "trace": str(trace_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    launch_info = {
        "run_dir": str(run_dir),
        "server_pid": server.pid,
        "client_pid": client.pid if client is not None else None,
        "game_port": game_port,
        "admin_port": admin_port,
        "model": model,
        "steps": steps,
        "trace": str(trace_path),
        "summary": str(summary_path),
        "installed": installed,
        "server_command": server_cmd,
        "client_command": client_cmd,
        "note": "The visible client remains open after the coal objective loop finishes.",
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
    return launch_info


def launch_firs_live(
    *,
    workbook: Path | str,
    executable: str | None = None,
    openttd_user_dir: Path | str | None = None,
    output_root: Path | str = "runs_firs",
    model: str = "gpt-5.5",
    steps: int = 10,
    resolution: str = "1280x800",
    record: bool = False,
    record_source: str | None = None,
    repl: bool = False,
    allow_heuristic: bool = False,
    start_delay: float = 10.0,
    step_delay: float = 3.0,
) -> dict[str, Any]:
    local_user_dir = _set_openttd_user_dir(openttd_user_dir)
    workbook_path = Path(workbook)
    run_config, workbook_meta = read_firs_ops_workbook(workbook_path)
    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")
    if not os.environ.get("OPENAI_API_KEY") and not allow_heuristic:
        raise EnvError("OPENAI_API_KEY is required for FIRS GPT play. Use --allow-heuristic for bridge testing.")

    install = verify_firs_installed(local_user_dir)
    run_dir = _new_run_dir(Path(output_root), suffix="firs_ops")
    ensure_opengfx()
    installed = install_live_bridge()
    game_port, admin_port = _find_distinct_free_ports(3979, 3977)
    cfg_text = render_firs_live_config(
        run_config=run_config,
        install=install,
        game_port=game_port,
        admin_port=admin_port,
        admin_password=ADMIN_PASSWORD,
    )
    artifact_cfg_path = run_dir / "openttd.cfg"
    artifact_client_cfg_path = run_dir / "openttd-viewer.cfg"
    cfg_path = (local_user_dir / f"tycoonle-openttd-{run_dir.name}.cfg") if local_user_dir else artifact_cfg_path
    client_cfg_path = (
        local_user_dir / f"tycoonle-openttd-{run_dir.name}-viewer.cfg" if local_user_dir else artifact_client_cfg_path
    )
    cfg_path.write_text(cfg_text, encoding="ascii")
    client_cfg_text = _with_client_name(cfg_text, f"TycoonLE OpenTTD FIRS Viewer {game_port}")
    client_cfg_path.write_text(client_cfg_text, encoding="ascii")
    if cfg_path != artifact_cfg_path:
        artifact_cfg_path.write_text(cfg_text, encoding="ascii")
    if client_cfg_path != artifact_client_cfg_path:
        artifact_client_cfg_path.write_text(client_cfg_text, encoding="ascii")

    server_cmd = [
        str(exe),
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
    process_cwd = str(local_user_dir or Path(exe).parent)
    server = _popen_hidden(
        server_cmd,
        cwd=process_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    client_cmd = [
        str(exe),
        "-c",
        str(client_cfg_path),
        "-x",
        "-X",
        "-n",
        f"127.0.0.1:{game_port}#0",
        "-I",
        "OpenGFX",
        "-S",
        "NoSound",
        "-M",
        "NoMusic",
        "-r",
        resolution,
    ]

    trace_path = run_dir / "firs_trace.jsonl"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.xlsx"
    gameplay_path = run_dir / "gameplay.mp4"
    timelapse_path = run_dir / "gameplay_8x.mp4"
    client: subprocess.Popen[bytes] | None = None
    recorder: subprocess.Popen[bytes] | None = None
    admin = AdminClient("127.0.0.1", admin_port)
    executed_steps = 0
    completed = False
    final_observation: dict[str, Any] | None = None
    failed = True
    try:
        admin.connect()
        admin.send_gamescript({"type": "observe"})
        observation = _next_observation(admin, timeout=90.0)
        final_observation = observation
        client = subprocess.Popen(
            client_cmd,
            cwd=process_cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        if record:
            capture_delay = min(2.0, max(0.0, start_delay))
            time.sleep(capture_delay)
            _close_recording_overlays(record_source)
            recorder = _start_recording(gameplay_path, source=record_source)
            time.sleep(max(0.0, start_delay - capture_delay))
        else:
            time.sleep(start_delay)
        repl_session = FIRSReplSession(admin, observation, workbook_meta, run_config.vehicles_per_route) if repl else None
        with trace_path.open("a", encoding="utf-8") as trace:
            trace.write(json.dumps({"event": "initial_observation", "data": observation}) + "\n")
            for index in range(steps):
                if repl_session is not None:
                    repl_session.observation = observation
                    program = _choose_firs_repl_program(
                        repl_session,
                        observation,
                        workbook_meta=workbook_meta,
                        model=model,
                        allow_heuristic=allow_heuristic,
                        vehicles_per_route=run_config.vehicles_per_route,
                    )
                    trace.write(json.dumps({"event": "repl_program", "step": index + 1, "data": {"code": program}}) + "\n")
                    feedback = repl_session.execute(program)
                    trace.write(
                        json.dumps(
                            {
                                "event": "repl_feedback",
                                "step": index + 1,
                                "data": {
                                    "stdout": feedback.get("stdout", ""),
                                    "stderr": feedback.get("stderr", ""),
                                    "actions": len(feedback.get("actions", [])),
                                },
                            }
                        )
                        + "\n"
                    )
                    for executed in feedback.get("actions", []):
                        trace.write(json.dumps({"event": "action", "step": index + 1, "data": executed["action"]}) + "\n")
                        trace.write(json.dumps({"event": "result", "step": index + 1, "data": executed["result"]}) + "\n")
                        trace.write(
                            json.dumps({"event": "observation", "step": index + 1, "data": executed["observation"]}) + "\n"
                        )
                    observation = feedback["observation"]
                    final_observation = observation
                    executed_steps = index + 1
                else:
                    action = _choose_firs_action(
                        observation,
                        workbook_meta=workbook_meta,
                        model=model,
                        allow_heuristic=allow_heuristic,
                        vehicles_per_route=run_config.vehicles_per_route,
                    )
                    trace.write(json.dumps({"event": "action", "step": index + 1, "data": action}) + "\n")
                    admin.send_gamescript({"type": "action", "action": action})
                    result = _next_result(admin, timeout=180.0)
                    observation = _next_observation(admin, timeout=180.0)
                    final_observation = observation
                    executed_steps = index + 1
                    trace.write(json.dumps({"event": "result", "step": index + 1, "data": result}) + "\n")
                    trace.write(json.dumps({"event": "observation", "step": index + 1, "data": observation}) + "\n")
                completed = _firs_objective_done(observation, workbook_meta)
                if completed:
                    break
                time.sleep(step_delay)
        failed = False
    finally:
        admin.close()
        if recorder is not None:
            _stop_recording(recorder)
            _write_timelapse(gameplay_path, timelapse_path)
        if failed:
            _terminate_process(client)
            _terminate_process(server)

    routes = final_observation.get("routes", []) if final_observation else []
    route_profit = sum(float(route.get("profit", 0) or route.get("vehicle_profit", 0) or 0) for route in routes)
    summary = {
        "objective": "firs_ops_chain",
        "completed": completed,
        "executed_steps": executed_steps,
        "requested_steps": steps,
        "model": model,
        "seed": run_config.seed,
        "economy": run_config.economy,
        "firs_newgrf": str(install.newgrf_path),
        "openttd_user_dir": str(local_user_dir) if local_user_dir else None,
        "final_tick": final_observation.get("tick") if final_observation else None,
        "final_bank_balance": final_observation.get("bank_balance") if final_observation else None,
        "route_profit": route_profit,
        "routes": routes,
        "trace": str(trace_path),
        "run_dir": str(run_dir),
        "workbook": str(workbook_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    export_run_to_xlsx(run_dir, report_path, source_workbook=workbook_path)

    launch_info = {
        "run_dir": str(run_dir),
        "server_pid": server.pid,
        "client_pid": client.pid if client is not None else None,
        "game_port": game_port,
        "admin_port": admin_port,
        "model": model,
        "steps": steps,
        "trace": str(trace_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "recording": str(gameplay_path) if gameplay_path.exists() else None,
        "timelapse": str(timelapse_path) if timelapse_path.exists() else None,
        "record_source": record_source or os.environ.get("OPENTTD_RECORD_SOURCE") or ("window-region=OpenTTD 15.3" if os.name == "nt" else os.environ.get("DISPLAY", ":0.0")),
        "repl": repl,
        "installed": installed,
        "firs_newgrf": str(install.newgrf_path),
        "openttd_user_dir": str(local_user_dir) if local_user_dir else None,
        "server_command": server_cmd,
        "client_command": client_cmd,
        "note": "The visible client remains open after the FIRS objective loop finishes.",
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
    return launch_info


def launch_firs_replay(
    *,
    replay: Path | str,
    workbook: Path | str | None = None,
    executable: str | None = None,
    openttd_user_dir: Path | str | None = None,
    output_root: Path | str = "runs_replay",
    resolution: str = "1280x720",
    record: bool = True,
    record_source: str | None = None,
    async_video: bool = True,
    start_delay: float = 10.0,
    action_delay: float = 2.0,
) -> dict[str, Any]:
    local_user_dir = _set_openttd_user_dir(openttd_user_dir)
    replay_path = Path(replay)
    replay_payload = load_replay(replay_path)
    scenario = replay_payload.get("scenario", {}) or {}
    replay_run_dir = Path(str(replay_payload.get("run_dir", ""))) if replay_payload.get("run_dir") else replay_path.parent
    workbook_path = Path(workbook or scenario.get("workbook") or replay_run_dir / "workbook.xlsx")
    if not workbook_path.exists() and not workbook:
        fallback = Path("templates") / "firs_ops_plan.xlsx"
        if fallback.exists():
            workbook_path = fallback
    run_config, workbook_meta = read_firs_ops_workbook(workbook_path)
    if scenario.get("seed") is not None:
        run_config = replace(run_config, seed=int(scenario["seed"]))
    if scenario.get("economy"):
        run_config = replace(run_config, economy=str(scenario["economy"]))
    actions = replay_actions(replay_payload)
    if not actions:
        raise EnvError(f"Replay has no macro-actions: {replay_path}")

    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")

    install = verify_firs_installed(local_user_dir)
    run_dir = _new_run_dir(Path(output_root), suffix="firs_replay")
    ensure_opengfx()
    installed = install_live_bridge()
    game_port, admin_port = _find_distinct_free_ports(3979, 3977)
    cfg_text = render_firs_live_config(
        run_config=run_config,
        install=install,
        game_port=game_port,
        admin_port=admin_port,
        admin_password=ADMIN_PASSWORD,
    )
    artifact_cfg_path = run_dir / "openttd.cfg"
    artifact_client_cfg_path = run_dir / "openttd-viewer.cfg"
    cfg_path = (local_user_dir / f"tycoonle-openttd-{run_dir.name}.cfg") if local_user_dir else artifact_cfg_path
    client_cfg_path = (
        local_user_dir / f"tycoonle-openttd-{run_dir.name}-viewer.cfg" if local_user_dir else artifact_client_cfg_path
    )
    cfg_path.write_text(cfg_text, encoding="ascii")
    client_cfg_text = _with_client_name(cfg_text, f"TycoonLE OpenTTD Replay Viewer {game_port}")
    client_cfg_path.write_text(client_cfg_text, encoding="ascii")
    if cfg_path != artifact_cfg_path:
        artifact_cfg_path.write_text(cfg_text, encoding="ascii")
    if client_cfg_path != artifact_client_cfg_path:
        artifact_client_cfg_path.write_text(client_cfg_text, encoding="ascii")

    server_cmd = [
        str(exe),
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
    process_cwd = str(local_user_dir or Path(exe).parent)
    server = _popen_hidden(
        server_cmd,
        cwd=process_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )
    client_cmd = [
        str(exe),
        "-c",
        str(client_cfg_path),
        "-x",
        "-X",
        "-n",
        f"127.0.0.1:{game_port}#0",
        "-I",
        "OpenGFX",
        "-S",
        "NoSound",
        "-M",
        "NoMusic",
        "-r",
        resolution,
    ]

    trace_path = run_dir / "replay_trace.jsonl"
    summary_path = run_dir / "summary.json"
    gameplay_path = run_dir / "gameplay.mp4"
    timelapse_path = run_dir / "gameplay_8x.mp4"
    client: subprocess.Popen[bytes] | None = None
    recorder: subprocess.Popen[bytes] | None = None
    timelapse_process: subprocess.Popen[bytes] | None = None
    admin = AdminClient("127.0.0.1", admin_port)
    final_observation: dict[str, Any] | None = None
    executed_actions = 0
    failed = True
    try:
        admin.connect()
        admin.send_gamescript({"type": "observe"})
        observation = _next_observation(admin, timeout=90.0)
        final_observation = observation
        client = subprocess.Popen(
            client_cmd,
            cwd=process_cwd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            stdin=subprocess.DEVNULL,
        )
        if record:
            capture_delay = min(2.0, max(0.0, start_delay))
            time.sleep(capture_delay)
            _close_recording_overlays(record_source)
            recorder = _start_recording(gameplay_path, source=record_source)
            time.sleep(max(0.0, start_delay - capture_delay))
        else:
            time.sleep(start_delay)
        with trace_path.open("a", encoding="utf-8") as trace:
            _write_event(trace, "initial_observation", 0, observation)
            for index, action in enumerate(actions, start=1):
                replay_action = dict(action)
                trace.write(
                    json.dumps({"event": "replay_action", "step": index, "data": replay_action}, separators=(",", ":"))
                    + "\n"
                )
                admin.send_gamescript({"type": "action", "action": replay_action})
                result = _next_result(admin, timeout=180.0)
                observation = _next_observation(admin, timeout=180.0)
                final_observation = observation
                executed_actions = index
                _write_event(trace, "result", index, result)
                _write_event(trace, "observation", index, observation)
                time.sleep(action_delay)
        failed = False
    finally:
        admin.close()
        if recorder is not None:
            _stop_recording(recorder)
            if async_video:
                timelapse_process = _start_timelapse(gameplay_path, timelapse_path)
            else:
                _write_timelapse(gameplay_path, timelapse_path)
        if failed:
            _terminate_process(client)
            _terminate_process(server)

    routes = final_observation.get("routes", []) if final_observation else []
    summary = {
        "objective": "firs_replay",
        "failed": failed,
        "source_replay": str(replay_path),
        "source_run_dir": replay_payload.get("run_dir"),
        "seed": run_config.seed,
        "economy": run_config.economy,
        "actions_requested": len(actions),
        "actions_executed": executed_actions,
        "final_tick": final_observation.get("tick") if final_observation else None,
        "final_bank_balance": final_observation.get("bank_balance") if final_observation else None,
        "routes": routes,
        "trace": str(trace_path),
        "run_dir": str(run_dir),
        "workbook": str(workbook_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    launch_info = {
        "run_dir": str(run_dir),
        "server_pid": server.pid,
        "client_pid": client.pid if client is not None else None,
        "game_port": game_port,
        "admin_port": admin_port,
        "source_replay": str(replay_path),
        "actions": len(actions),
        "trace": str(trace_path),
        "summary": str(summary_path),
        "recording": str(gameplay_path) if gameplay_path.exists() else None,
        "timelapse": str(timelapse_path) if record else None,
        "timelapse_pid": timelapse_process.pid if timelapse_process is not None else None,
        "timelapse_async": bool(async_video and timelapse_process is not None),
        "record_source": record_source or os.environ.get("OPENTTD_RECORD_SOURCE") or ("window-region=OpenTTD 15.3" if os.name == "nt" else os.environ.get("DISPLAY", ":0.0")),
        "installed": installed,
        "firs_newgrf": str(install.newgrf_path),
        "openttd_user_dir": str(local_user_dir) if local_user_dir else None,
        "server_command": server_cmd,
        "client_command": client_cmd,
        "note": "Replay client remains open after macro-actions finish. Timelapse may continue encoding if timelapse_async is true.",
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
    return launch_info


def launch_firs_research(
    *,
    workbook: Path | str,
    executable: str | None = None,
    openttd_user_dir: Path | str | None = None,
    output_root: Path | str = "runs_firs_research",
    model: str = "gpt-5.5",
    steps: int = 32,
    benchmark_task: str | None = None,
    benchmark_file: Path | str | None = None,
    seed: int | None = None,
    allow_heuristic: bool = False,
    step_delay: float = 0.0,
) -> dict[str, Any]:
    local_user_dir = _set_openttd_user_dir(openttd_user_dir)
    workbook_path = Path(workbook)
    run_config, workbook_meta = read_firs_ops_workbook(workbook_path)
    task = select_task(benchmark_task, benchmark_file) if benchmark_task else None
    if task is not None:
        workbook_meta = task_to_workbook_meta(task, workbook_meta)
        run_config = replace(
            run_config,
            seed=task.seed,
            economy=task.economy,
            target_chain=tuple(task.objectives) or run_config.target_chain,
        )
        steps = task.steps if steps == 32 else steps
    if seed is not None:
        run_config = replace(run_config, seed=int(seed))
    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")
    if not os.environ.get("OPENAI_API_KEY") and not allow_heuristic:
        raise EnvError("OPENAI_API_KEY is required for FIRS research play. Use --allow-heuristic for bridge testing.")

    install = verify_firs_installed(local_user_dir)
    run_dir = _new_run_dir(Path(output_root), suffix="firs_research")
    ensure_opengfx()
    installed = install_live_bridge()
    game_port, admin_port = _find_distinct_free_ports(3979, 3977)
    cfg_text = render_firs_live_config(
        run_config=run_config,
        install=install,
        game_port=game_port,
        admin_port=admin_port,
        admin_password=ADMIN_PASSWORD,
    )
    artifact_cfg_path = run_dir / "openttd.cfg"
    cfg_path = (local_user_dir / f"tycoonle-openttd-{run_dir.name}.cfg") if local_user_dir else artifact_cfg_path
    cfg_path.write_text(cfg_text, encoding="ascii")
    if cfg_path != artifact_cfg_path:
        artifact_cfg_path.write_text(cfg_text, encoding="ascii")

    server_cmd = [
        str(exe),
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
    process_cwd = str(local_user_dir or Path(exe).parent)
    server = _popen_hidden(
        server_cmd,
        cwd=process_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    trace_path = run_dir / "firs_trace.jsonl"
    programs_path = run_dir / "programs.jsonl"
    streams_path = run_dir / "stdout_stderr.jsonl"
    observations_path = run_dir / "observations.jsonl"
    rewards_path = run_dir / "rewards.jsonl"
    actions_path = run_dir / "actions.jsonl"
    summary_path = run_dir / "summary.json"
    report_path = run_dir / "report.xlsx"
    replay_path = run_dir / "replay.json"

    admin = AdminClient("127.0.0.1", admin_port)
    final_observation: dict[str, Any] | None = None
    completed = False
    executed_steps = 0
    total_reward = 0.0
    previous_snapshot: dict[str, Any] | None = None
    failed = True
    try:
        admin.connect()
        admin.send_gamescript({"type": "observe"})
        observation = _next_observation(admin, timeout=90.0)
        final_observation = observation
        previous_snapshot = _firs_reward_snapshot(observation, workbook_meta)
        task_meta = workbook_meta.get("benchmark_task", {})
        session = FIRSReplSession(admin, observation, workbook_meta, run_config.vehicles_per_route, task_meta=task_meta)

        with (
            trace_path.open("a", encoding="utf-8") as trace,
            programs_path.open("a", encoding="utf-8") as programs,
            streams_path.open("a", encoding="utf-8") as streams,
            observations_path.open("a", encoding="utf-8") as observations_file,
            rewards_path.open("a", encoding="utf-8") as rewards_file,
            actions_path.open("a", encoding="utf-8") as actions_file,
        ):
            _write_event(trace, "initial_observation", 0, observation)
            _write_jsonl(observations_file, {"step": 0, "observation": observation})
            for index in range(steps):
                step = index + 1
                session.observation = observation
                program = _choose_firs_repl_program(
                    session,
                    observation,
                    workbook_meta=workbook_meta,
                    model=model,
                    allow_heuristic=allow_heuristic,
                    vehicles_per_route=run_config.vehicles_per_route,
                )
                _write_jsonl(programs, {"step": step, "program": program})
                _write_event(trace, "repl_program", step, {"code": program})
                started = time.time()
                feedback = session.execute(program)
                elapsed = time.time() - started
                _write_jsonl(
                    streams,
                    {
                        "step": step,
                        "stdout": feedback.get("stdout", ""),
                        "stderr": feedback.get("stderr", ""),
                        "elapsed_seconds": round(elapsed, 3),
                    },
                )
                _write_event(
                    trace,
                    "repl_feedback",
                    step,
                    {
                        "stdout": feedback.get("stdout", ""),
                        "stderr": feedback.get("stderr", ""),
                        "actions": len(feedback.get("actions", [])),
                    },
                )
                for executed in feedback.get("actions", []):
                    _write_jsonl(actions_file, {"step": step, **executed})
                    _write_event(trace, "action", step, executed["action"])
                    _write_event(trace, "result", step, executed["result"])
                    _write_event(trace, "observation", step, executed["observation"])

                observation = feedback["observation"]
                final_observation = observation
                current_snapshot = _firs_reward_snapshot(observation, workbook_meta)
                reward = _firs_step_reward(previous_snapshot or {}, current_snapshot, feedback.get("actions", []))
                total_reward += reward["reward"]
                _write_jsonl(rewards_file, {"step": step, **reward, "snapshot": current_snapshot})
                _write_jsonl(observations_file, {"step": step, "observation": observation})
                previous_snapshot = current_snapshot
                executed_steps = step
                completed = _firs_objective_done(observation, workbook_meta)
                if task is not None and _benchmark_success(current_snapshot, task.success):
                    completed = True
                if completed:
                    break
                if step_delay > 0:
                    time.sleep(step_delay)
        failed = False
    finally:
        admin.close()
        _terminate_process(server)

    routes = final_observation.get("routes", []) if final_observation else []
    route_profit = sum(float(route.get("profit", 0) or route.get("vehicle_profit", 0) or 0) for route in routes)
    final_snapshot = _firs_reward_snapshot(final_observation or {}, workbook_meta)
    if task is not None and _benchmark_success(final_snapshot, task.success):
        completed = True
    summary = {
        "objective": "firs_research_repl",
        "completed": completed,
        "failed": failed,
        "executed_steps": executed_steps,
        "requested_steps": steps,
        "model": model,
        "seed": run_config.seed,
        "economy": run_config.economy,
        "firs_newgrf": str(install.newgrf_path),
        "openttd_user_dir": str(local_user_dir) if local_user_dir else None,
        "final_tick": final_observation.get("tick") if final_observation else None,
        "final_bank_balance": final_observation.get("bank_balance") if final_observation else None,
        "route_profit": route_profit,
        "total_reward": round(total_reward, 3),
        "milestones": final_snapshot.get("milestones", {}),
        "final_score": score_snapshot(final_observation or {}, workbook_meta.get("objectives", [])),
        "benchmark_task": task.id if task else None,
        "benchmark_mode": task.mode if task else None,
        "success_criteria": task.success if task else None,
        "routes": routes,
        "trace": str(trace_path),
        "programs": str(programs_path),
        "stdout_stderr": str(streams_path),
        "observations": str(observations_path),
        "rewards": str(rewards_path),
        "actions": str(actions_path),
        "replay": str(replay_path),
        "run_dir": str(run_dir),
        "workbook": str(workbook_path),
        "note": "Research-mode run: headless OpenTTD/FIRS with persistent Python REPL artifacts.",
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    export_run_to_xlsx(run_dir, report_path, source_workbook=workbook_path)

    launch_info = {
        "run_dir": str(run_dir),
        "server_pid": server.pid,
        "game_port": game_port,
        "admin_port": admin_port,
        "model": model,
        "steps": steps,
        "trace": str(trace_path),
        "programs": str(programs_path),
        "stdout_stderr": str(streams_path),
        "observations": str(observations_path),
        "rewards": str(rewards_path),
        "actions": str(actions_path),
        "summary": str(summary_path),
        "report": str(report_path),
        "replay": str(replay_path),
        "installed": installed,
        "firs_newgrf": str(install.newgrf_path),
        "openttd_user_dir": str(local_user_dir) if local_user_dir else None,
        "server_command": server_cmd,
    }
    (run_dir / "launch.json").write_text(json.dumps(launch_info, indent=2), encoding="utf-8")
    export_replay(run_dir, replay_path)
    return launch_info


def launch_firs_benchmark(
    *,
    workbook: Path | str,
    executable: str | None = None,
    openttd_user_dir: Path | str | None = None,
    output_root: Path | str = "runs_firs_benchmark",
    models: list[str] | None = None,
    tasks: list[str] | None = None,
    repeats: int = 1,
    benchmark_file: Path | str | None = None,
    allow_heuristic: bool = False,
) -> dict[str, Any]:
    run_root = _new_run_dir(Path(output_root), suffix="firs_benchmark")
    model_names = models or ["gpt-5.5"]
    task_ids = tasks or [task.id for task in _load_task_list(benchmark_file)]
    summaries: list[dict[str, Any]] = []
    runs: list[dict[str, Any]] = []
    for task_id in task_ids:
        for model in model_names:
            for repeat in range(repeats):
                run_info = launch_firs_research(
                    workbook=workbook,
                    executable=executable,
                    openttd_user_dir=openttd_user_dir,
                    output_root=run_root,
                    model=model,
                    benchmark_task=task_id,
                    benchmark_file=benchmark_file,
                    allow_heuristic=allow_heuristic,
                )
                run_info["repeat"] = repeat + 1
                runs.append(run_info)
                summaries.append(json.loads(Path(run_info["summary"]).read_text(encoding="utf-8")))
    aggregate = aggregate_runs(summaries)
    payload = {"run_dir": str(run_root), "runs": runs, "aggregate": aggregate}
    (run_root / "benchmark_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def launch_route_builder_benchmark(
    *,
    workbook: Path | str,
    executable: str | None = None,
    openttd_user_dir: Path | str | None = None,
    output_root: Path | str = "runs_route_builder",
    seed: int | None = None,
    economy: str | None = None,
    attempts: int = 20,
    vehicles: int | None = None,
    wait_months: int = 6,
    max_path_tiles: int = 256,
    target_success_rate: float = 0.9,
) -> dict[str, Any]:
    local_user_dir = _set_openttd_user_dir(openttd_user_dir)
    workbook_path = Path(workbook)
    run_config, workbook_meta = read_firs_ops_workbook(workbook_path)
    if seed is not None:
        run_config = replace(run_config, seed=int(seed))
    if economy is not None:
        run_config = replace(run_config, economy=str(economy))
    route_vehicles = int(vehicles if vehicles is not None else run_config.vehicles_per_route)
    exe = executable or os.environ.get("OPENTTD_EXECUTABLE") or _find_openttd()
    if not exe or not Path(exe).exists():
        raise EnvError("OpenTTD executable not found. Install OpenTTD or set OPENTTD_EXECUTABLE.")

    install = verify_firs_installed(local_user_dir)
    run_dir = _new_run_dir(Path(output_root), suffix="route_builder")
    ensure_opengfx()
    installed = install_live_bridge()
    game_port, admin_port = _find_distinct_free_ports(3979, 3977)
    cfg_text = render_firs_live_config(
        run_config=run_config,
        install=install,
        game_port=game_port,
        admin_port=admin_port,
        admin_password=ADMIN_PASSWORD,
    )
    artifact_cfg_path = run_dir / "openttd.cfg"
    cfg_path = (local_user_dir / f"tycoonle-openttd-{run_dir.name}.cfg") if local_user_dir else artifact_cfg_path
    cfg_path.write_text(cfg_text, encoding="ascii")
    if cfg_path != artifact_cfg_path:
        artifact_cfg_path.write_text(cfg_text, encoding="ascii")

    server_cmd = [
        str(exe),
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
    process_cwd = str(local_user_dir or Path(exe).parent)
    server = _popen_hidden(
        server_cmd,
        cwd=process_cwd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        stdin=subprocess.DEVNULL,
    )

    attempts_path = run_dir / "route_builder_attempts.jsonl"
    skipped_path = run_dir / "route_builder_skipped.jsonl"
    observations_path = run_dir / "observations.jsonl"
    summary_path = run_dir / "summary.json"
    admin = AdminClient("127.0.0.1", admin_port)
    attempt_rows: list[dict[str, Any]] = []
    skipped_rows: list[dict[str, Any]] = []
    final_observation: dict[str, Any] | None = None
    failed = True
    try:
        admin.connect()
        admin.send_gamescript({"type": "observe"})
        observation = _next_observation(admin, timeout=90.0)
        final_observation = observation
        candidate_limit = max(1, attempts * 5)
        pairs = _route_builder_candidate_pairs(observation, workbook_meta, limit=candidate_limit)
        with (
            attempts_path.open("a", encoding="utf-8") as attempts_file,
            skipped_path.open("a", encoding="utf-8") as skipped_file,
            observations_path.open("a", encoding="utf-8") as observations_file,
        ):
            _write_jsonl(observations_file, {"attempt": 0, "observation": observation})
            for probe_index, pair in enumerate(pairs, start=1):
                if len(attempt_rows) >= attempts:
                    break
                construction_index = len(attempt_rows) + 1
                action = _build_action_from_pair(pair, route_vehicles, f"route builder benchmark {construction_index}")
                action["max_path_tiles"] = int(max_path_tiles)
                before_count = len(observation.get("routes", []) or [])
                result: dict[str, Any]
                action_timed_out = False
                try:
                    admin.send_gamescript({"type": "action", "action": action})
                    result = _next_result(admin, timeout=180.0)
                    observation = _next_observation(admin, timeout=90.0)
                except Exception as exc:
                    action_timed_out = True
                    result = {
                        "type": "result",
                        "action_type": "build_cargo_route",
                        "error": "action_timeout_or_error",
                        "detail": str(exc),
                    }
                result_error = str(result.get("error") or "")
                if result_error in ROUTE_BUILDER_INFEASIBLE_REASONS:
                    skipped_row = {
                        "probe": probe_index,
                        "reason": result_error,
                        "source_id": pair.get("source_id"),
                        "source_name": pair.get("source_name"),
                        "destination_id": pair.get("destination_id"),
                        "destination_name": pair.get("destination_name"),
                        "cargo_id": pair.get("cargo_id"),
                        "cargo_label": pair.get("cargo_label"),
                        "distance": pair.get("distance"),
                        "production": pair.get("production"),
                        "result": result,
                    }
                    skipped_rows.append(skipped_row)
                    final_observation = observation
                    _write_jsonl(skipped_file, skipped_row)
                    _write_jsonl(observations_file, {"probe": probe_index, "skipped": True, "observation": observation})
                    continue
                route = _matching_route(observation, pair)
                build_success = not result.get("error") and route is not None and not bool(route.get("is_virtual"))
                wait_result: dict[str, Any] | None = None
                wait_results: list[dict[str, Any]] = []
                if build_success and wait_months > 0:
                    remaining_months = int(wait_months)
                    while remaining_months > 0:
                        chunk_months = min(2, remaining_months)
                        wait_action = {
                            "type": "wait_months",
                            "months": chunk_months,
                            "label": f"route builder delivery validation {construction_index}",
                        }
                        try:
                            admin.send_gamescript({"type": "action", "action": wait_action})
                            wait_result = _next_result(admin, timeout=max(120.0, chunk_months * 90.0))
                            wait_results.append(wait_result)
                            observation = _next_observation(admin, timeout=90.0)
                        except Exception as exc:
                            wait_result = {
                                "type": "result",
                                "action_type": "wait_months",
                                "error": "action_timeout_or_error",
                                "detail": str(exc),
                            }
                            wait_results.append(wait_result)
                            admin.send_gamescript({"type": "observe"})
                            observation = _next_observation(admin, timeout=120.0)
                            break
                        route = _matching_route(observation, pair)
                        if _route_validation_signal(route):
                            break
                        if wait_result.get("error"):
                            break
                        remaining_months -= chunk_months
                final_observation = observation
                delivered = int((route or {}).get("delivered", 0) or 0)
                profit = float((route or {}).get("profit", (route or {}).get("vehicle_profit", 0)) or 0)
                active_success = build_success and _route_active(route)
                operational_success = build_success and _route_validation_signal(route)
                failure_reason = _route_builder_failure_reason(result, wait_result, route, build_success, operational_success)
                row = {
                    "attempt": construction_index,
                    "probe": probe_index,
                    "source_id": pair.get("source_id"),
                    "source_name": pair.get("source_name"),
                    "destination_id": pair.get("destination_id"),
                    "destination_name": pair.get("destination_name"),
                    "cargo_id": pair.get("cargo_id"),
                    "cargo_label": pair.get("cargo_label"),
                    "distance": pair.get("distance"),
                    "production": pair.get("production"),
                    "routes_before": before_count,
                    "routes_after": len(observation.get("routes", []) or []),
                    "build_success": build_success,
                    "active_success": active_success,
                    "operational_success": operational_success,
                    "delivered": delivered,
                    "profit": round(profit, 3),
                    "error": result.get("error"),
                    "failure_reason": failure_reason,
                    "result": result,
                    "wait_result": wait_result,
                    "wait_results": wait_results,
                }
                attempt_rows.append(row)
                _write_jsonl(attempts_file, row)
                _write_jsonl(observations_file, {"attempt": construction_index, "probe": probe_index, "observation": observation})
                if action_timed_out:
                    break
        failed = False
    finally:
        admin.close()
        _terminate_process(server)

    aggregate = aggregate_route_builder_attempts(attempt_rows, target_success_rate=target_success_rate)
    if len(attempt_rows) < attempts:
        aggregate = {
            **aggregate,
            "level1_pass": False,
            "feasible_level1_pass": False,
            "insufficient_feasible_attempts": True,
        }
    payload = {
        "objective": "route_builder_level1",
        "failed": failed,
        "seed": run_config.seed,
        "economy": run_config.economy,
        "attempts_requested": attempts,
        "attempts_executed": len(attempt_rows),
        "pairs_probed": len(attempt_rows) + len(skipped_rows),
        "skipped_infeasible": len(skipped_rows),
        "vehicles": route_vehicles,
        "wait_months": wait_months,
        "max_path_tiles": max_path_tiles,
        "aggregate": aggregate,
        "final_tick": final_observation.get("tick") if final_observation else None,
        "final_routes": final_observation.get("routes", []) if final_observation else [],
        "attempts": str(attempts_path),
        "skipped_pairs": str(skipped_path),
        "observations": str(observations_path),
        "summary": str(summary_path),
        "run_dir": str(run_dir),
        "workbook": str(workbook_path),
        "installed": installed,
        "firs_newgrf": str(install.newgrf_path),
        "openttd_user_dir": str(local_user_dir) if local_user_dir else None,
        "server_command": server_cmd,
    }
    summary_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def _load_task_list(benchmark_file: Path | str | None) -> list[Any]:
    from openttd_le.research.benchmarks import load_benchmark_tasks

    return load_benchmark_tasks(benchmark_file)


def _choose_action(
    observation: dict[str, Any],
    *,
    model: str,
    allow_heuristic: bool,
    focus_town_id: int | None = None,
) -> dict[str, Any]:
    towns = observation.get("towns", [])
    if allow_heuristic:
        town = next((town for town in towns if town.get("id") == focus_town_id), None)
        if town is None:
            index = observation.get("step", 0) % max(1, len(towns))
            town = towns[index] if towns else {"id": 0, "name": "town"}
        return {"type": "road_burst", "town_id": town["id"], "label": f"heuristic {town['name']}"}

    model_observation = dict(observation)
    if focus_town_id is not None:
        model_observation["visual_focus_town_id"] = focus_town_id
        model_observation["visual_focus_instruction"] = (
            "For this live visual demo, choose this town_id for road_burst actions "
            "unless it is missing from the towns list."
        )

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are controlling OpenTTD through a live bridge. Return only compact JSON. "
                    "Allowed actions: "
                    "{\"type\":\"road_burst\",\"town_id\":123,\"label\":\"short reason\"} or "
                    "{\"type\":\"sign\",\"town_id\":123,\"text\":\"short note\"}. "
                    "Prefer road_burst actions that make visible construction fast. "
                    "If visual_focus_town_id is present in the observation, use that town_id."
                ),
            },
            {"role": "user", "content": json.dumps(model_observation, separators=(",", ":"))},
        ],
        "max_output_tokens": 1000,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = data.get("output_text") or _extract_response_text(data)
    if not text.strip():
        raise EnvError(
            "Model returned no visible text. "
            f"status={data.get('status')} incomplete_details={data.get('incomplete_details')}"
        )
    return _parse_json(text)


def _choose_coal_action(observation: dict[str, Any], *, model: str, allow_heuristic: bool) -> dict[str, Any]:
    active = observation.get("active_objective")
    pairs = observation.get("coal_pairs", [])
    if allow_heuristic:
        if active:
            return {"type": "wait", "ticks": 1800, "label": "wait for first coal delivery"}
        if not pairs:
            return {"type": "sign", "town_id": 0, "text": "No coal route candidate found"}
        pair = pairs[0]
        return {
            "type": "build_coal_route",
            "source_id": pair["source_id"],
            "destination_id": pair["destination_id"],
            "cargo_id": pair["cargo_id"],
            "vehicles": 4,
            "label": "shortest coal route",
        }

    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are playing a live OpenTTD coal-delivery objective. Return only compact JSON. "
                    "Goal: make the first useful coal delivery and produce profit. "
                    "If active_objective is null, choose exactly one coal_pairs item and return "
                    "{\"type\":\"build_coal_route\",\"source_id\":1,\"destination_id\":2,"
                    "\"cargo_id\":0,\"vehicles\":4,\"label\":\"short reason\"}. "
                    "Prefer short distance and decent production. "
                    "If active_objective exists, return {\"type\":\"wait\",\"ticks\":1800,"
                    "\"label\":\"wait for delivery\"}. "
                    "Do not build repeated routes unless no active objective exists."
                ),
            },
            {"role": "user", "content": json.dumps(observation, separators=(",", ":"))},
        ],
        "max_output_tokens": 1000,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = data.get("output_text") or _extract_response_text(data)
    if not text.strip():
        raise EnvError(
            "Model returned no visible text. "
            f"status={data.get('status')} incomplete_details={data.get('incomplete_details')}"
        )
    return _parse_json(text)


def _choose_firs_repl_program(
    session: FIRSReplSession,
    observation: dict[str, Any],
    *,
    workbook_meta: dict[str, Any],
    model: str,
    allow_heuristic: bool,
    vehicles_per_route: int,
) -> str:
    if allow_heuristic:
        return _heuristic_firs_program(session, observation, vehicles_per_route)

    model_observation = {
        "step": observation.get("step"),
        "tick": observation.get("tick"),
        "candidate_routes": session.candidate_routes(),
        "routes": observation.get("routes", []),
        "typed_api": {
            "cargo_chains": [asdict(chain) for chain in get_cargo_chains(observation, CARGO_VALUE)[:12]],
            "finance": get_finance(observation).__dict__,
            "task": session.task_meta,
        },
        "cargo_waiting": observation.get("cargo_waiting", [])[:20],
        "station_ratings": observation.get("station_ratings", [])[:20],
        "company_finances": observation.get("company_finances", {}),
        "workbook": {
            "scenario": workbook_meta.get("fields", {}),
            "objectives": workbook_meta.get("objectives", []),
        },
        "last_stdout": session.last_stdout[-2000:],
        "last_stderr": session.last_stderr[-2000:],
    }
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are operating OpenTTD/FIRS through a safe persistent Python REPL. "
                    "Return only Python code, no Markdown. Available variables: obs, routes, "
                    "candidate_routes, cargo_chains, industries, finance, workbook, task, Prototype, "
                    "last_stdout, last_stderr. Available functions: get_industries(), get_cargo_chains(), "
                    "get_routes(), get_finance(), short_routes(max_distance=40,cargo=None), "
                    "observe(), build_cargo_route(source_id, destination_id, cargo_id, vehicles=5, "
                    "physical=True, max_path_tiles=256, allow_virtual=False, preview_roads=False, label=''), "
                    "add_vehicles(route_id,count), wait_months(months,label=''), "
                    "inspect_bottlenecks(), borrow_or_repay(amount). Do not import modules. Do not read or "
                    "write files. Do not use network calls. Use at most one or two game-changing helper calls "
                    "per program. Print concise diagnostics. If routes is empty and candidate_routes exists, "
                    "build candidate_routes[0] with physical=True so visible stations, roads, depots, and "
                    "vehicles are attempted. Keep allow_virtual=False for parity runs; long paths return a "
                    "typed failure instead of silently creating virtual progress."
                ),
            },
            {"role": "user", "content": json.dumps(model_observation, separators=(",", ":"))},
        ],
        "max_output_tokens": 1600,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = data.get("output_text") or _extract_response_text(data)
    code = _strip_code_fence(text).strip()
    if not code:
        return _heuristic_firs_program(session, observation, vehicles_per_route)
    return code


def _heuristic_firs_program(
    session: FIRSReplSession,
    observation: dict[str, Any],
    vehicles_per_route: int,
) -> str:
    candidates = session.candidate_routes()
    routes = observation.get("routes", [])
    if not routes and candidates:
        return (
            "route = candidate_routes[0]\n"
            "print('building physical route', route)\n"
            "build_cargo_route(route['source_id'], route['destination_id'], route['cargo_id'], "
            f"vehicles={vehicles_per_route}, physical=True, max_path_tiles=256, allow_virtual=False, preview_roads=False, "
            "label='repl first physical route')"
        )
    if routes:
        return "print('waiting for route progress', routes)\nwait_months(1, label='repl wait for delivery')"
    return "print('no candidate routes; inspecting')\ninspect_bottlenecks()"


def _strip_code_fence(text: str) -> str:
    fenced = re.search(r"```(?:python|py)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    return fenced.group(1) if fenced else text


def _choose_firs_action(
    observation: dict[str, Any],
    *,
    workbook_meta: dict[str, Any],
    model: str,
    allow_heuristic: bool,
    vehicles_per_route: int,
) -> dict[str, Any]:
    if allow_heuristic:
        return _heuristic_firs_action(observation, workbook_meta, vehicles_per_route)

    graph = observation.get("industry_graph", [])
    routes = observation.get("routes", [])
    candidate_routes = _candidate_firs_pairs(graph, workbook_meta.get("objectives", []), limit=12)
    if not candidate_routes:
        candidate_routes = _candidate_firs_pairs_from_io(observation, workbook_meta.get("objectives", []), limit=12)
    if routes:
        candidate_routes = [pair for pair in candidate_routes if not _route_already_registered(pair, routes)]
        if not candidate_routes:
            return {"type": "wait_months", "months": 1, "label": "wait for downstream candidate"}
    model_observation = {
        "step": observation.get("step"),
        "tick": observation.get("tick"),
        "candidate_routes": candidate_routes,
        "routes": routes,
        "cargo_waiting": observation.get("cargo_waiting", [])[:20],
        "station_ratings": observation.get("station_ratings", [])[:20],
        "company_finances": observation.get("company_finances", {}),
        "workbook": {
            "scenario": workbook_meta.get("fields", {}),
            "objectives": workbook_meta.get("objectives", []),
        },
    }
    payload = {
        "model": model,
        "input": [
            {
                "role": "system",
                "content": (
                    "You are controlling OpenTTD/FIRS through a macro-action bridge. Return only compact JSON. "
                    "Goal: execute the spreadsheet operations plan, build the dependency chain, inspect bottlenecks, "
                    "and keep route profit positive. Allowed actions: "
                    "{\"type\":\"build_cargo_route\",\"source_id\":1,\"destination_id\":2,\"cargo_id\":0,\"vehicles\":5,\"physical\":true,\"label\":\"reason\"}; "
                    "{\"type\":\"add_vehicles\",\"route_id\":\"route_001\",\"count\":2}; "
                    "{\"type\":\"wait_months\",\"months\":3,\"label\":\"reason\"}; "
                    "{\"type\":\"inspect_bottlenecks\"}; "
                    "{\"type\":\"borrow_or_repay\",\"amount\":50000}. "
                    "Use only source_id, destination_id, cargo_id, and route_id values present in the observation. "
                    "Always include physical:true for build_cargo_route so the bridge attempts real stations, roads, depots, and vehicles. "
                    "If routes is empty and candidate_routes is non-empty, build the first candidate route. "
                    "Do not inspect bottlenecks repeatedly when no routes exist. "
                    "If a route has just been built, wait for delivery before building downstream unless the downstream "
                    "source already appears in candidate_routes."
                ),
            },
            {"role": "user", "content": json.dumps(model_observation, separators=(",", ":"))},
        ],
        "max_output_tokens": 2000,
    }
    req = urllib.request.Request(
        "https://api.openai.com/v1/responses",
        data=json.dumps(payload).encode("utf-8"),
        headers={
            "Authorization": f"Bearer {os.environ['OPENAI_API_KEY']}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=90) as response:
        data = json.loads(response.read().decode("utf-8"))
    text = data.get("output_text") or _extract_response_text(data)
    if not text.strip():
        if candidate_routes and not routes:
            return _build_action_from_pair(candidate_routes[0], vehicles_per_route, "fallback spreadsheet candidate")
        return {"type": "wait_months", "months": 1, "label": "fallback after empty model response"}
    action = _parse_json(text)
    if action.get("type") == "build_cargo_route" and candidate_routes:
        action["physical"] = True
        exact = any(
            pair.get("source_id") == action.get("source_id")
            and pair.get("destination_id") == action.get("destination_id")
            and pair.get("cargo_id") == action.get("cargo_id")
            for pair in candidate_routes
        )
        if not exact:
            return _build_action_from_pair(candidate_routes[0], vehicles_per_route, "corrected spreadsheet candidate")
        if _route_already_registered(action, routes):
            return {"type": "wait_months", "months": 1, "label": "avoid duplicate route"}
    return action


def _heuristic_firs_action(
    observation: dict[str, Any],
    workbook_meta: dict[str, Any],
    vehicles_per_route: int,
) -> dict[str, Any]:
    objectives = workbook_meta.get("objectives", [])
    routes = observation.get("routes", [])
    graph = observation.get("industry_graph", [])
    if not routes:
        first = objectives[0] if objectives else {}
        pair = _find_graph_pair(graph, first)
        if pair is None:
            pair = graph[0] if graph else None
        if pair is None:
            return {"type": "inspect_bottlenecks"}
        return _build_action_from_pair(pair, vehicles_per_route, "first FIRS chain route")

    if len(routes) == 1:
        first_route = routes[0]
        if first_route.get("delivered", 0) <= 0:
            return {"type": "wait_months", "months": 3, "label": "wait for first delivery"}
        second = objectives[1] if len(objectives) > 1 else {}
        pair = _find_graph_pair(graph, second, source_id=first_route.get("destination_id"))
        if pair is not None:
            return _build_action_from_pair(pair, vehicles_per_route, "downstream FIRS chain route")
    waiting_route = _route_needing_capacity(routes)
    if waiting_route is not None:
        return {"type": "add_vehicles", "route_id": waiting_route["route_id"], "count": 2}
    return {"type": "wait_months", "months": 3, "label": "wait for FIRS production"}


def _find_graph_pair(
    graph: list[dict[str, Any]],
    objective: dict[str, Any],
    *,
    source_id: int | None = None,
) -> dict[str, Any] | None:
    source_type = str(objective.get("source_type", "")).lower()
    destination_type = str(objective.get("destination_type", "")).lower()
    cargo = str(objective.get("cargo", "")).upper()
    for pair in graph:
        if source_id is not None and pair.get("source_id") != source_id:
            continue
        source_name = str(pair.get("source_type") or pair.get("source_name") or "").lower()
        destination_name = str(pair.get("destination_type") or pair.get("destination_name") or "").lower()
        cargo_label = str(pair.get("cargo") or pair.get("cargo_label") or "").upper()
        if source_type and source_type not in source_name:
            continue
        if destination_type and destination_type not in destination_name:
            continue
        if cargo and cargo != cargo_label:
            continue
        return pair
    return None


def _candidate_firs_pairs(
    graph: list[dict[str, Any]],
    objectives: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for objective in objectives:
        for pair in graph:
            if not _pair_matches_objective(pair, objective):
                continue
            if not _road_compatible_industry_name(str(pair.get("source_name", ""))):
                continue
            if not _road_compatible_industry_name(str(pair.get("destination_name", ""))):
                continue
            key = (pair.get("source_id"), pair.get("destination_id"), pair.get("cargo_id"))
            if key in seen:
                continue
            seen.add(key)
            candidates.append(pair)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _candidate_firs_pairs_from_io(
    observation: dict[str, Any],
    objectives: list[dict[str, Any]],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    outputs = observation.get("industry_outputs", [])
    inputs = observation.get("industry_inputs", [])
    locations = _industry_locations_from_graph(observation)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for objective in objectives:
        cargo = str(objective.get("cargo", "")).upper()
        source_type = str(objective.get("source_type", "")).lower()
        destination_type = str(objective.get("destination_type", "")).lower()
        for source in outputs:
            cargo_label = str(source.get("cargo_label", "")).upper()
            source_name = str(source.get("industry_name", "")).lower()
            if cargo and cargo_label != cargo:
                continue
            if not _road_compatible_industry_name(source_name):
                continue
            if source_type and source_type not in source_name:
                continue
            production = source.get("production", 0)
            if isinstance(production, (int, float)) and production <= 0:
                continue
            for destination in inputs:
                destination_name = str(destination.get("industry_name", "")).lower()
                if str(destination.get("cargo_label", "")).upper() != cargo_label:
                    continue
                if not _road_compatible_industry_name(destination_name):
                    continue
                if destination_type and destination_type not in destination_name:
                    continue
                key = (source.get("industry_id"), destination.get("industry_id"), source.get("cargo_id"))
                if key in seen:
                    continue
                seen.add(key)
                pair = {
                    "source_id": source.get("industry_id"),
                    "source_name": source.get("industry_name"),
                    "destination_id": destination.get("industry_id"),
                    "destination_name": destination.get("industry_name"),
                    "cargo_id": source.get("cargo_id"),
                    "cargo_label": cargo_label,
                    "cargo_name": source.get("cargo_name"),
                    "production": production,
                }
                candidates.append(_enrich_pair_geometry(pair, locations))
    candidates.sort(
        key=lambda pair: (
            int(pair.get("distance", 999999) or 999999),
            -float(CARGO_VALUE.get(str(pair.get("cargo_label", "")).upper(), 1.0)),
            -float(pair.get("production", 0) or 0),
            str(pair.get("source_name", "")),
            str(pair.get("destination_name", "")),
        )
    )
    return candidates[:limit]


def _candidate_open_play_pairs(observation: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    graph = list(observation.get("industry_graph", []) or [])
    routes = observation.get("routes", []) or []
    graph = [
        pair
        for pair in graph
        if not _route_already_registered(pair, routes)
        and _road_compatible_industry_name(str(pair.get("source_name", "")))
        and _road_compatible_industry_name(str(pair.get("destination_name", "")))
    ]
    graph.sort(
        key=lambda pair: (
            int(pair.get("distance", 999999) or 999999),
            -float(CARGO_VALUE.get(str(pair.get("cargo_label", pair.get("cargo", ""))).upper(), 1.0)),
            -float(pair.get("production", 0) or 0),
        )
    )
    return graph[:limit]


def _candidate_open_play_pairs_from_io(observation: dict[str, Any], *, limit: int) -> list[dict[str, Any]]:
    outputs = observation.get("industry_outputs", []) or []
    inputs = observation.get("industry_inputs", []) or []
    routes = observation.get("routes", []) or []
    locations = _industry_locations_from_graph(observation)
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for source in outputs:
        source_id = source.get("industry_id")
        source_name = str(source.get("industry_name", ""))
        cargo_id = source.get("cargo_id")
        cargo_label = str(source.get("cargo_label", "")).upper()
        production = source.get("production", 0)
        if not isinstance(source_id, int) or not isinstance(cargo_id, int):
            continue
        if not cargo_label or not _road_compatible_industry_name(source_name):
            continue
        if isinstance(production, (int, float)) and production <= 0:
            continue
        for destination in inputs:
            destination_id = destination.get("industry_id")
            destination_name = str(destination.get("industry_name", ""))
            if not isinstance(destination_id, int) or destination_id == source_id:
                continue
            if str(destination.get("cargo_label", "")).upper() != cargo_label:
                continue
            if not _road_compatible_industry_name(destination_name):
                continue
            pair = {
                "source_id": source_id,
                "source_name": source_name,
                "destination_id": destination_id,
                "destination_name": destination_name,
                "cargo_id": cargo_id,
                "cargo_label": cargo_label,
                "cargo_name": source.get("cargo_name", cargo_label),
                "production": production,
            }
            if _route_already_registered(pair, routes):
                continue
            key = _route_key(pair)
            if key in seen:
                continue
            seen.add(key)
            candidates.append(_enrich_pair_geometry(pair, locations))
    candidates.sort(
        key=lambda pair: (
            int(pair.get("distance", 999999) or 999999),
            -float(CARGO_VALUE.get(str(pair.get("cargo_label", "")).upper(), 1.0)),
            -float(pair.get("production", 0) or 0),
            str(pair.get("source_name", "")),
            str(pair.get("destination_name", "")),
        )
    )
    return candidates[:limit]


def _industry_locations_from_graph(observation: dict[str, Any]) -> dict[int, tuple[int, int]]:
    locations: dict[int, tuple[int, int]] = {}
    for pair in observation.get("industry_graph", []) or []:
        source_id = pair.get("source_id")
        destination_id = pair.get("destination_id")
        if isinstance(source_id, int) and isinstance(pair.get("source_x"), int) and isinstance(pair.get("source_y"), int):
            locations[source_id] = (int(pair["source_x"]), int(pair["source_y"]))
        if (
            isinstance(destination_id, int)
            and isinstance(pair.get("destination_x"), int)
            and isinstance(pair.get("destination_y"), int)
        ):
            locations[destination_id] = (int(pair["destination_x"]), int(pair["destination_y"]))
    return locations


def _enrich_pair_geometry(pair: dict[str, Any], locations: dict[int, tuple[int, int]]) -> dict[str, Any]:
    enriched = dict(pair)
    source_id = enriched.get("source_id")
    destination_id = enriched.get("destination_id")
    source_xy = locations.get(source_id) if isinstance(source_id, int) else None
    destination_xy = locations.get(destination_id) if isinstance(destination_id, int) else None
    if source_xy is not None:
        enriched["source_x"], enriched["source_y"] = source_xy
    if destination_xy is not None:
        enriched["destination_x"], enriched["destination_y"] = destination_xy
    if source_xy is not None and destination_xy is not None:
        enriched["distance"] = abs(source_xy[0] - destination_xy[0]) + abs(source_xy[1] - destination_xy[1])
    return enriched


def _road_compatible_industry_name(name: str) -> bool:
    lowered = name.lower()
    blocked = (" port", "port", "fishing grounds", "fishing harbour", "dredging site")
    return not any(token in lowered for token in blocked)


def _route_builder_candidate_pairs(
    observation: dict[str, Any],
    workbook_meta: dict[str, Any],
    *,
    limit: int,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    sources = [
        _candidate_firs_pairs(observation.get("industry_graph", []), workbook_meta.get("objectives", []), limit=limit),
        _candidate_firs_pairs_from_io(observation, workbook_meta.get("objectives", []), limit=limit),
        _candidate_open_play_pairs(observation, limit=limit * 2),
        _candidate_open_play_pairs_from_io(observation, limit=limit * 2),
    ]
    for source in sources:
        for pair in source:
            key = _route_key(pair)
            if key in seen:
                continue
            if not all(isinstance(pair.get(field), int) for field in ("source_id", "destination_id", "cargo_id")):
                continue
            seen.add(key)
            candidates.append(pair)
            if len(candidates) >= limit:
                return candidates
    return candidates


def _route_already_registered(pair: dict[str, Any], routes: list[dict[str, Any]]) -> bool:
    for route in routes:
        if (
            route.get("source_id") == pair.get("source_id")
            and route.get("destination_id") == pair.get("destination_id")
            and route.get("cargo_id") == pair.get("cargo_id")
        ):
            return True
    return False


def _matching_route(observation: dict[str, Any], pair: dict[str, Any]) -> dict[str, Any] | None:
    for route in observation.get("routes", []) or []:
        if (
            route.get("source_id") == pair.get("source_id")
            and route.get("destination_id") == pair.get("destination_id")
            and route.get("cargo_id") == pair.get("cargo_id")
        ):
            return route
    return None


def _route_active(route: dict[str, Any] | None) -> bool:
    if route is None:
        return False
    vehicles = route.get("vehicle_details", []) or []
    if not vehicles:
        return int(route.get("vehicles", 0) or 0) > 0
    for vehicle in vehicles:
        if not vehicle.get("valid", True):
            continue
        if int(vehicle.get("orders", 0) or 0) <= 0:
            continue
        if not bool(vehicle.get("in_depot", False)):
            return True
        if int(vehicle.get("load", 0) or 0) > 0:
            return True
    return False


def _route_validation_signal(route: dict[str, Any] | None) -> bool:
    if route is None:
        return False
    if int(route.get("delivered", 0) or 0) > 0:
        return True
    if float(route.get("profit", route.get("vehicle_profit", 0)) or 0) > 0:
        return True
    if int(route.get("source_waiting", 0) or 0) > 0:
        return True
    for vehicle in route.get("vehicle_details", []) or []:
        if int(vehicle.get("load", 0) or 0) > 0:
            return True
    return False


def _route_builder_failure_reason(
    result: dict[str, Any],
    wait_result: dict[str, Any] | None,
    route: dict[str, Any] | None,
    build_success: bool,
    operational_success: bool,
) -> str | None:
    if operational_success:
        return None
    if result.get("error"):
        return str(result["error"])
    if not build_success:
        return "route_not_registered"
    if wait_result and wait_result.get("error"):
        return str(wait_result["error"])
    if route is None:
        return "route_missing_after_wait"
    if int(route.get("vehicles", 0) or 0) <= 0:
        return "no_vehicles"
    return "no_delivery_after_wait"


def _route_key(pair: dict[str, Any]) -> tuple[Any, Any, Any]:
    return pair.get("source_id"), pair.get("destination_id"), pair.get("cargo_id")


def _pair_matches_objective(pair: dict[str, Any], objective: dict[str, Any]) -> bool:
    source_type = str(objective.get("source_type", "")).lower()
    destination_type = str(objective.get("destination_type", "")).lower()
    cargo = str(objective.get("cargo", "")).upper()
    source_name = str(pair.get("source_type") or pair.get("source_name") or "").lower()
    destination_name = str(pair.get("destination_type") or pair.get("destination_name") or "").lower()
    cargo_label = str(pair.get("cargo") or pair.get("cargo_label") or "").upper()
    if source_type and source_type not in source_name:
        return False
    if destination_type and destination_type not in destination_name:
        return False
    if cargo and cargo != cargo_label:
        return False
    return True


def _build_action_from_pair(pair: dict[str, Any], vehicles: int, label: str) -> dict[str, Any]:
    return {
        "type": "build_cargo_route",
        "source_id": pair["source_id"],
        "destination_id": pair["destination_id"],
        "cargo_id": pair["cargo_id"],
        "vehicles": vehicles,
        "physical": True,
        "max_path_tiles": 256,
        "allow_virtual": False,
        "preview_roads": False,
        "label": label,
    }


def _route_needing_capacity(routes: list[dict[str, Any]]) -> dict[str, Any] | None:
    for route in routes:
        waiting = route.get("source_waiting", 0)
        vehicles = route.get("vehicles", 0)
        if isinstance(waiting, (int, float)) and waiting > 80 and vehicles < 10:
            return route
    return None


def _coal_objective_done(observation: dict[str, Any]) -> bool:
    active = observation.get("active_objective") or {}
    return active.get("delivered", 0) > 0 or active.get("vehicle_profit", 0) > 0


def _firs_objective_done(observation: dict[str, Any], workbook_meta: dict[str, Any]) -> bool:
    objectives = workbook_meta.get("objectives", [])
    target_count = max(1, min(2, len(objectives)))
    routes = observation.get("routes", [])
    delivered = sum(1 for route in routes if route.get("delivered", 0) > 0)
    profit = sum(float(route.get("profit", 0) or route.get("vehicle_profit", 0) or 0) for route in routes)
    return delivered >= target_count and profit > 0


def _firs_reward_snapshot(observation: dict[str, Any], workbook_meta: dict[str, Any]) -> dict[str, Any]:
    routes = observation.get("routes", []) or []
    delivered_total = sum(int(route.get("delivered", 0) or 0) for route in routes)
    route_profit = sum(float(route.get("profit", 0) or route.get("vehicle_profit", 0) or 0) for route in routes)
    score = score_snapshot(observation, workbook_meta.get("objectives", []))
    cargo_labels = sorted({str(route.get("cargo_label", route.get("cargo", ""))) for route in routes if route})
    delivered_routes = sum(1 for route in routes if int(route.get("delivered", 0) or 0) > 0)
    processed_targets = {
        str(item.get("cargo", "")).upper()
        for item in workbook_meta.get("objectives", [])[1:]
        if str(item.get("cargo", "")).strip()
    }
    delivered_processed = any(
        str(route.get("cargo_label", "")).upper() in processed_targets and int(route.get("delivered", 0) or 0) > 0
        for route in routes
    )
    failed_station_ratings = sum(
        1
        for route in routes
        if isinstance(route.get("source_rating"), (int, float)) and route.get("source_rating", 0) >= 0 and route.get("source_rating", 0) < 45
    )
    production_score = score["network_value"]
    milestones = {
        "first_route": len(routes) > 0,
        "first_delivery": delivered_routes > 0,
        "first_processed_delivery": delivered_processed,
        "positive_network_profit": route_profit > 0,
        "two_operational_routes": delivered_routes >= 2,
    }
    return {
        "production_score": round(production_score, 3),
        "cargo_score": score["cargo_score"],
        "network_value": score["network_value"],
        "delivered_total": delivered_total,
        "route_profit": round(route_profit, 3),
        "route_count": len(routes),
        "delivered_routes": delivered_routes,
        "cargo_labels": cargo_labels,
        "low_rating_routes": failed_station_ratings,
        "milestones": milestones,
    }


def _benchmark_success(snapshot: dict[str, Any], criteria: dict[str, Any]) -> bool:
    if not criteria:
        return False
    if int(snapshot.get("route_count", 0) or 0) < int(criteria.get("min_routes", 0) or 0):
        return False
    if int(snapshot.get("delivered_routes", 0) or 0) < int(criteria.get("min_delivered_routes", 0) or 0):
        return False
    if float(snapshot.get("network_value", 0) or 0) < float(criteria.get("min_network_value", 0) or 0):
        return False
    required_cargo = str(criteria.get("required_cargo", "")).upper()
    if required_cargo and required_cargo not in {str(label).upper() for label in snapshot.get("cargo_labels", [])}:
        return False
    return True


def _firs_step_reward(
    previous: dict[str, Any],
    current: dict[str, Any],
    actions: list[dict[str, Any]],
) -> dict[str, Any]:
    previous_milestones = previous.get("milestones", {}) if previous else {}
    current_milestones = current.get("milestones", {})
    new_milestones = [
        key for key, value in current_milestones.items() if value and not bool(previous_milestones.get(key))
    ]
    failed_actions = sum(1 for item in actions if item.get("result", {}).get("error"))
    production_delta = float(current.get("production_score", 0) or 0) - float(previous.get("production_score", 0) or 0)
    route_delta = int(current.get("route_count", 0) or 0) - int(previous.get("route_count", 0) or 0)
    reward = production_delta + route_delta * 5.0 + len(new_milestones) * 25.0 - failed_actions * 2.0
    return {
        "reward": round(reward, 3),
        "production_delta": round(production_delta, 3),
        "new_milestones": new_milestones,
        "failed_actions": failed_actions,
    }


def _write_event(handle: Any, event: str, step: int, data: dict[str, Any]) -> None:
    handle.write(json.dumps({"event": event, "step": step, "data": data}, separators=(",", ":")) + "\n")
    handle.flush()


def _write_jsonl(handle: Any, data: dict[str, Any]) -> None:
    handle.write(json.dumps(data, separators=(",", ":")) + "\n")
    handle.flush()


def _next_observation(admin: AdminClient, timeout: float, reasons: tuple[str, ...] | None = None) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = admin.read_gamescript(timeout=deadline - time.time())
        if message.get("type") == "observation":
            if reasons is not None and message.get("reason") not in reasons:
                continue
            return message
    raise EnvError("Timed out waiting for observation.")


def _next_result(admin: AdminClient, timeout: float) -> dict[str, Any]:
    deadline = time.time() + timeout
    while time.time() < deadline:
        message = admin.read_gamescript(timeout=deadline - time.time())
        if message.get("type") == "result":
            return message
    raise EnvError("Timed out waiting for action result.")


def _live_config(seed: int, game_port: int, admin_port: int) -> str:
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
number_towns = 3
number_industries = 4
terrain_type = 0
quantity_sea_lakes = 0
vehicle_breakdowns = 0

[network]
server_name = TycoonLE OpenTTD Live
client_name = TycoonLE OpenTTD Server
server_port = {game_port}
server_admin_port = {admin_port}
admin_password = {ADMIN_PASSWORD}
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
generation_seed = {seed}
map_x = 8
map_y = 8
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


def _new_run_dir(output_root: Path, *, suffix: str = "live_gpt") -> Path:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    root = output_root.resolve()
    for counter in range(1000):
        name = f"{timestamp}_{suffix}" if counter == 0 else f"{timestamp}_{suffix}_{counter:03d}"
        run_dir = root / name
        try:
            run_dir.mkdir(parents=True, exist_ok=False)
            return run_dir
        except FileExistsError:
            continue
    raise EnvError(f"Could not allocate a unique run directory under {root}")


def _with_client_name(config_text: str, client_name: str) -> str:
    replacement = f"client_name = {client_name}"
    if re.search(r"(?m)^client_name = .*$", config_text):
        return re.sub(r"(?m)^client_name = .*$", replacement, config_text, count=1)
    return config_text.replace("[network]\n", f"[network]\n{replacement}\n", 1)


def _set_openttd_user_dir(path: Path | str | None) -> Path | None:
    raw = path if path is not None else os.environ.get("OPENTTD_USER_DIR")
    if not raw:
        return None
    resolved = Path(raw).expanduser().resolve()
    resolved.mkdir(parents=True, exist_ok=True)
    os.environ["OPENTTD_USER_DIR"] = str(resolved)
    return resolved


def _popen_hidden(args: list[str], **kwargs: Any) -> subprocess.Popen[bytes]:
    if os.name == "nt":
        kwargs["creationflags"] = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        startupinfo = subprocess.STARTUPINFO()
        startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
        startupinfo.wShowWindow = 0
        kwargs["startupinfo"] = startupinfo
    return subprocess.Popen(args, **kwargs)


def _start_recording(path: Path, *, source: str | None = None) -> subprocess.Popen[bytes] | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        return None
    path.parent.mkdir(parents=True, exist_ok=True)
    capture_format = "gdigrab" if os.name == "nt" else "x11grab"
    capture_source = source or os.environ.get("OPENTTD_RECORD_SOURCE")
    if capture_source is None:
        capture_source = "window-region=OpenTTD 15.3" if os.name == "nt" else os.environ.get("DISPLAY", ":0.0")
    input_args: list[str]
    if os.name == "nt" and capture_source.startswith("window-region="):
        title = capture_source.removeprefix("window-region=")
        rect = _wait_for_window_client_rect(title)
        if rect is None:
            return None
        left, top, width, height = rect
        input_args = [
            "-f",
            capture_format,
            "-framerate",
            "15",
            "-offset_x",
            str(left),
            "-offset_y",
            str(top),
            "-video_size",
            f"{width}x{height}",
            "-i",
            "desktop",
        ]
    elif os.name == "nt" and capture_source.startswith("title="):
        title = capture_source.removeprefix("title=")
        if _wait_for_window_title(title) is None:
            return None
        input_args = [
            "-f",
            capture_format,
            "-framerate",
            "15",
            "-i",
            capture_source,
        ]
    else:
        input_args = ["-f", capture_format, "-framerate", "15", "-i", capture_source]
    command = [
        ffmpeg,
        "-y",
        *input_args,
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-movflags",
        "+faststart",
        str(path),
    ]
    return _popen_hidden(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.PIPE)


def _close_recording_overlays(source: str | None = None) -> bool:
    """Close transient OpenTTD UI windows before screen capture starts."""
    if os.name != "nt":
        return False
    title = _recording_window_title(source)
    if not title:
        return False
    hwnd = _wait_for_window_handle(title, timeout=15.0)
    if hwnd is None:
        return False
    import ctypes

    user32 = ctypes.windll.user32
    try:
        user32.SetForegroundWindow(hwnd)
    except OSError:
        pass
    vk_escape = 0x1B
    wm_keydown = 0x0100
    wm_keyup = 0x0101
    for _ in range(3):
        user32.PostMessageW(hwnd, wm_keydown, vk_escape, 0)
        user32.PostMessageW(hwnd, wm_keyup, vk_escape, 0)
        time.sleep(0.12)
    return True


def _recording_window_title(source: str | None = None) -> str | None:
    capture_source = source or os.environ.get("OPENTTD_RECORD_SOURCE")
    if capture_source is None:
        capture_source = "window-region=OpenTTD 15.3" if os.name == "nt" else os.environ.get("DISPLAY", ":0.0")
    if os.name != "nt":
        return None
    if capture_source.startswith("window-region="):
        return capture_source.removeprefix("window-region=")
    if capture_source.startswith("title="):
        return capture_source.removeprefix("title=")
    return None


def _wait_for_window_client_rect(title: str, timeout: float = 15.0) -> tuple[int, int, int, int] | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        rect = _find_window_client_rect(title)
        if rect is not None:
            return rect
        time.sleep(0.25)
    return None


def _find_window_client_rect(title: str) -> tuple[int, int, int, int] | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    try:
        user32.SetProcessDPIAware()
    except OSError:
        pass
    matches: list[tuple[int, int, int, int]] = []

    class POINT(ctypes.Structure):
        _fields_ = [("x", ctypes.c_long), ("y", ctypes.c_long)]

    def enum_callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title not in buffer.value:
            return True
        client_rect = wintypes.RECT()
        if not user32.GetClientRect(hwnd, ctypes.byref(client_rect)):
            return True
        width = int(client_rect.right - client_rect.left)
        height = int(client_rect.bottom - client_rect.top)
        if width <= 0 or height <= 0:
            return True
        origin = POINT(0, 0)
        if not user32.ClientToScreen(hwnd, ctypes.byref(origin)):
            return True
        matches.append((int(origin.x), int(origin.y), width, height))
        return False

    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(enum_callback)
    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


def _wait_for_window_handle(title: str, timeout: float = 15.0) -> int | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        hwnd = _find_window_handle(title)
        if hwnd is not None:
            return hwnd
        time.sleep(0.25)
    return None


def _find_window_handle(title: str) -> int | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    matches: list[int] = []

    def enum_callback(hwnd: int, _lparam: int) -> bool:
        if not user32.IsWindowVisible(hwnd):
            return True
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title in buffer.value:
            matches.append(int(hwnd))
            return False
        return True

    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(enum_callback)
    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


def _wait_for_window_title(title: str, timeout: float = 15.0) -> str | None:
    deadline = time.time() + timeout
    while time.time() < deadline:
        match = _find_window_title(title)
        if match is not None:
            return match
        time.sleep(0.25)
    return None


def _find_window_title(title: str) -> str | None:
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    user32 = ctypes.windll.user32
    matches: list[str] = []

    def enum_callback(hwnd: int, _lparam: int) -> bool:
        length = user32.GetWindowTextLengthW(hwnd)
        if length <= 0:
            return True
        buffer = ctypes.create_unicode_buffer(length + 1)
        user32.GetWindowTextW(hwnd, buffer, length + 1)
        if title in buffer.value:
            matches.append(buffer.value)
            return False
        return True

    callback = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)(enum_callback)
    user32.EnumWindows(callback, 0)
    return matches[0] if matches else None


def _stop_recording(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    if process.stdin is not None:
        try:
            process.stdin.write(b"q\n")
            process.stdin.flush()
            process.stdin.close()
            process.wait(timeout=15)
            return
        except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
            pass
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _terminate_process(process: subprocess.Popen[bytes] | None) -> None:
    if process is None or process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


def _write_timelapse(source: Path, target: Path) -> None:
    process = _start_timelapse(source, target)
    if process is None:
        return
    try:
        process.wait(timeout=600)
    except subprocess.TimeoutExpired:
        process.terminate()
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=10)


def _start_timelapse(source: Path, target: Path) -> subprocess.Popen[bytes] | None:
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    command = [
        ffmpeg,
        "-y",
        "-i",
        str(source),
        "-filter:v",
        "setpts=0.125*PTS",
        "-an",
        str(target),
    ]
    return _popen_hidden(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, stdin=subprocess.DEVNULL)


def _find_free_port(start: int) -> int:
    for port in range(start, start + 200):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            try:
                sock.bind(("0.0.0.0", port))
            except OSError:
                continue
            return port
    raise EnvError(f"No free TCP port found from {start}.")


def _find_distinct_free_ports(game_start: int = 3979, admin_start: int = 3977) -> tuple[int, int]:
    sockets: list[socket.socket] = []
    try:
        game_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        game_port = _bind_free_port(game_socket, game_start)
        sockets.append(game_socket)
        admin_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        admin_port = _bind_free_port(admin_socket, admin_start)
        sockets.append(admin_socket)
        return game_port, admin_port
    finally:
        for sock in sockets:
            sock.close()


def _bind_free_port(sock: socket.socket, start: int) -> int:
    for port in range(start, start + 200):
        try:
            sock.bind(("0.0.0.0", port))
            return int(sock.getsockname()[1])
        except OSError:
            continue
    raise EnvError(f"No free TCP port found from {start}.")


def _recv_exact(sock: socket.socket, count: int) -> bytes:
    chunks = []
    remaining = count
    while remaining:
        chunk = sock.recv(remaining)
        if not chunk:
            raise EnvError("Admin socket closed.")
        chunks.append(chunk)
        remaining -= len(chunk)
    return b"".join(chunks)


def _string(value: str) -> bytes:
    return value.encode("utf-8") + b"\x00"


def _read_string(payload: bytes) -> str:
    return payload.split(b"\x00", 1)[0].decode("utf-8", "replace")


def _extract_response_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _parse_json(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        raise EnvError(f"Model did not return JSON action: {text}")
    action = json.loads(match.group(0))
    if "actions" in action and isinstance(action["actions"], list) and action["actions"]:
        action = action["actions"][0]
    elif "action" in action and isinstance(action["action"], dict):
        action = action["action"]
    allowed = {
        "road_burst",
        "sign",
        "build_coal_route",
        "build_cargo_route",
        "add_vehicles",
        "wait",
        "wait_months",
        "inspect_bottlenecks",
        "borrow_or_repay",
    }
    if action.get("type") not in allowed:
        raise EnvError(f"Unsupported model action: {action}")
    return action
