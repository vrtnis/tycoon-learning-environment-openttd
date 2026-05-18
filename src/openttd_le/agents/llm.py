from __future__ import annotations

import json
import os
import re
import urllib.request
from typing import Any

from .base import Agent


SYSTEM_PROMPT = """You are playing OpenTTD-LE, a transport logistics benchmark.
Choose exactly one legal macro-action as compact JSON and no prose.

Allowed actions:
{"type":"build_route","source_id":"...","destination_id":"...","cargo":"...","mode":"road|rail"}
{"type":"add_vehicle","route_id":"...","count":1}
{"type":"wait","months":1}
{"type":"take_loan","amount":50000}
{"type":"repay_loan","amount":50000}

Prefer actions from candidate_actions when present. Prefer profitable cargo/passenger routes, keep debt controlled, add vehicles to routes before waiting, and submit only valid IDs from the observation."""


class OpenAIAgent(Agent):
    name = "openai"

    def __init__(self, model: str = "gpt-5.5", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENAI_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENAI_API_KEY is required for the OpenAI agent.")

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "input": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(_compact_observation(observation), separators=(",", ":"))},
            ],
            "max_output_tokens": 300,
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


class OpenRouterAgent(Agent):
    name = "openrouter"

    def __init__(self, model: str = "openai/gpt-5.5", api_key: str | None = None) -> None:
        self.model = model
        self.api_key = api_key or os.environ.get("OPENROUTER_API_KEY")
        if not self.api_key:
            raise RuntimeError("OPENROUTER_API_KEY is required for the OpenRouter agent.")

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(_compact_observation(observation), separators=(",", ":"))},
            ],
            "max_tokens": 300,
        }
        req = urllib.request.Request(
            "https://openrouter.ai/api/v1/chat/completions",
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
                "HTTP-Referer": "https://github.com/openttd-le/openttd-le",
                "X-Title": "OpenTTD-LE",
            },
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=90) as response:
            data = json.loads(response.read().decode("utf-8"))
        return _parse_action(data["choices"][0]["message"]["content"])


def _compact_observation(observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "scenario": observation["scenario"],
        "time": observation["time"],
        "company": observation["company"],
        "nodes": observation["nodes"],
        "routes": observation["routes"],
        "candidate_actions": observation.get("candidate_actions", [])[:12],
        "metrics": observation["metrics"],
        "last_event": observation["last_event"],
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
