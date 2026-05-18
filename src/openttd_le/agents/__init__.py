from __future__ import annotations

from openttd_le.agents.base import Agent
from openttd_le.agents.firs import HeuristicFIRSAgent, OpenAIFIRSMacroAgent, make_firs_agent
from openttd_le.agents.frontier import CandidateRankAgent, PreviewRerankAgent
from openttd_le.agents.greedy import GreedyAgent
from openttd_le.agents.llm import OpenAIAgent, OpenRouterAgent
from openttd_le.agents.random_agent import RandomAgent


def make_agent(name: str, model: str | None = None, seed: int | None = None) -> Agent:
    if name == "random":
        return RandomAgent(seed=seed)
    if name == "greedy":
        return GreedyAgent()
    if name in {"candidate_rank", "rank"}:
        return CandidateRankAgent()
    if name in {"preview_rerank", "best_of_n"}:
        return PreviewRerankAgent()
    if name == "openai":
        return OpenAIAgent(model=model or "gpt-5.5")
    if name == "openrouter":
        return OpenRouterAgent(model=model or "openai/gpt-5.5")
    raise ValueError(f"Unknown agent: {name}")


__all__ = [
    "Agent",
    "CandidateRankAgent",
    "GreedyAgent",
    "HeuristicFIRSAgent",
    "OpenAIFIRSMacroAgent",
    "OpenAIAgent",
    "OpenRouterAgent",
    "PreviewRerankAgent",
    "RandomAgent",
    "make_firs_agent",
    "make_agent",
]
