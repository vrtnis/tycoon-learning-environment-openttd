from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from openttd_le.agents.base import Agent


class HeuristicFIRSAgent(Agent):
    name = "heuristic_firs"

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        candidates = observation.get("candidate_actions", []) or []
        routes = observation.get("routes", []) or []
        if routes:
            for candidate in candidates:
                if candidate.get("kind") == "wait_months":
                    return dict(candidate["action"])
            return {"type": "wait_months", "months": 1, "label": "heuristic wait for delivery"}
        if candidates:
            return dict(candidates[0]["action"])
        return {"type": "inspect_bottlenecks"}


class OpenAIFIRSMacroAgent(Agent):
    name = "openai_firs_macro"

    def __init__(self, model: str = "gpt-5.5", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI FIRS agent.")

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {
                    "role": "system",
                    "content": (
                        "You are a baseline agent for OpenTTD-LE's real OpenTTD/FIRS environment. "
                        "Return exactly one compact JSON macro-action and no prose. The environment, "
                        "not you, executes the action and returns reward/termination. Prefer actions "
                        "listed in candidate_actions. Use only IDs present in the observation. "
                        "Valid actions: "
                        "{\"type\":\"build_cargo_route\",\"source_id\":1,\"destination_id\":2,\"cargo_id\":2,\"vehicles\":5,\"physical\":true,\"allow_virtual\":false,\"label\":\"reason\"}; "
                        "{\"type\":\"wait_months\",\"months\":1,\"label\":\"reason\"}; "
                        "{\"type\":\"add_vehicles\",\"route_id\":\"route_1\",\"count\":1}; "
                        "{\"type\":\"inspect_bottlenecks\"}; "
                        "{\"type\":\"borrow_or_repay\",\"amount\":50000}. "
                        "If no route exists and a feasible candidate exists, build a physical candidate route. "
                        "If a route exists, wait for delivery unless observations show a clear bottleneck."
                    ),
                },
                {"role": "user", "content": json.dumps(_compact_firs_observation(observation), separators=(",", ":"))},
            ],
            "max_output_tokens": 500,
        }
        req = urllib.request.Request(
            "https://api.openai.com/v1/responses",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _parse_action(_extract_openai_text(data))


def make_firs_agent(name: str, model: str | None = None) -> Agent:
    if name == "heuristic":
        return HeuristicFIRSAgent()
    if name == "openai":
        return OpenAIFIRSMacroAgent(model=model or "gpt-5.5")
    raise ValueError(f"Unknown OpenTTD/FIRS agent: {name}")


def _compact_firs_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "step": observation.get("step"),
        "tick": observation.get("tick"),
        "task": observation.get("task", {}),
        "workbook": observation.get("workbook", {}),
        "candidate_actions": observation.get("candidate_actions", [])[:12],
        "routes": observation.get("routes", []),
        "cargo_waiting": observation.get("cargo_waiting", [])[:20],
        "station_ratings": observation.get("station_ratings", [])[:20],
        "company_finances": observation.get("company_finances", {}),
        "bank_balance": observation.get("bank_balance"),
        "industry_graph": observation.get("industry_graph", [])[:20],
    }


def _extract_openai_text(data: dict[str, Any]) -> str:
    if isinstance(data.get("output_text"), str):
        return data["output_text"]
    chunks = []
    for item in data.get("output", []):
        for content in item.get("content", []):
            if isinstance(content.get("text"), str):
                chunks.append(content["text"])
    return "\n".join(chunks)


def _parse_action(text: str) -> dict[str, Any]:
    fenced = re.search(r"```(?:json)?\s*([\s\S]*?)```", text, re.IGNORECASE)
    candidate = fenced.group(1) if fenced else text
    match = re.search(r"\{[\s\S]*\}", candidate)
    if not match:
        raise RuntimeError(f"Model did not return JSON action: {text}")
    return json.loads(match.group(0))
