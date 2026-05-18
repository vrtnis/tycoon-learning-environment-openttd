from __future__ import annotations

from typing import Any


ENVIRONMENT_VERSION = "0.4.0"
OBSERVATION_SCHEMA = "openttd-le-observation-v1"
CANDIDATE_ACTION_SCHEMA = "openttd-le-candidate-action-v1"
ACTION_PREVIEW_SCHEMA = "openttd-le-action-preview-v1"
STEP_REWARD_SCHEMA = "openttd-le-step-reward-v1"
EPISODE_TRACE_SCHEMA = "openttd-le-episode-trace-v1"
REPLAY_SCHEMA = "openttd-le-core-replay-v1"
DATASET_SCHEMA = "openttd-le-core-dataset-v1"


def schema_manifest() -> dict[str, Any]:
    return {
        "environment_version": ENVIRONMENT_VERSION,
        "observation": OBSERVATION_SCHEMA,
        "candidate_action": CANDIDATE_ACTION_SCHEMA,
        "action_preview": ACTION_PREVIEW_SCHEMA,
        "step_reward": STEP_REWARD_SCHEMA,
        "episode_trace": EPISODE_TRACE_SCHEMA,
        "replay": REPLAY_SCHEMA,
        "dataset": DATASET_SCHEMA,
    }
