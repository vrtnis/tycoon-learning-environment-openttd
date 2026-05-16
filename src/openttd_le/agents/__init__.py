from __future__ import annotations

from openttd_le.agents.base import Agent
from openttd_le.agents.greedy import GreedyAgent
from openttd_le.agents.llm import OpenAIAgent, OpenRouterAgent
from openttd_le.agents.random_agent import RandomAgent


def make_agent(name: str, model: str | None = None, seed: int | None = None) -> Agent:
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent()
    if name == "openai":
        return OpenAIAgent(model=model or "gpt-5.5")
    if name == "openrouter":
        return OpenRouterAgent(model=model or "openai/gpt-5.5")
    raise ValueError(f"Unknown agent: {name}")


__all__ = ["Agent", "GreedyAgent", "OpenAIAgent", "OpenRouterAgent", "RandomAgent", "make_agent"]
