from __future__ import annotations

from typing import Any

from openttd_le.core.research import preview_action

from .base import Agent


class CandidateRankAgent(Agent):
    name = "candidate_rank"

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        candidates = [
            candidate
            for candidate in observation.get("candidate_actions", [])
            if candidate.get("directly_executable") and candidate.get("feasible")
        ]
        if not candidates:
            candidates = [candidate for candidate in observation.get("candidate_actions", []) if candidate.get("feasible")]
        if not candidates:
            return {"type": "wait", "months": 1}
        return dict(max(candidates, key=lambda item: float(item.get("rank_score", 0) or 0))["action"])


class PreviewRerankAgent(Agent):
    name = "preview_rerank"

    def act(self, observation: dict[str, Any]) -> dict[str, Any]:
        best: tuple[float, dict[str, Any]] | None = None
        for candidate in observation.get("candidate_actions", []):
            if not candidate.get("directly_executable") or not candidate.get("feasible"):
                continue
            action = dict(candidate["action"])
            preview = preview_action(observation, action)
            score = _preview_score(candidate, preview)
            if best is None or score > best[0]:
                best = (score, action)
        if best is not None:
            return best[1]
        return CandidateRankAgent().act(observation)


def _preview_score(candidate: dict[str, Any], preview: dict[str, Any]) -> float:
    estimates = candidate.get("estimates", {})
    components = preview.get("components", {})
    return (
        float(candidate.get("rank_score", 0) or 0)
        + float(candidate.get("objective_relevance", 0) or 0)
        + max(0.0, float(estimates.get("monthly_profit", components.get("profit_delta", 0)) or 0)) / 1000.0
        - float(candidate.get("requires_loan", 0) or 0) / 100000.0
    )
