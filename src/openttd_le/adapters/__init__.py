"""Compatibility adapters for external research tooling."""

from openttd_le.adapters.gymnasium import (
    FIRS_DETERMINISTIC_GYM_ID,
    FIRS_GYM_ID,
    FIRS_OBSERVATION_SPEC,
    TOY_GYM_ID,
    OpenTTDFIRSGymEnv,
    OpenTTDLEGymEnv,
    make_firs,
    make_firs_vector,
    register_envs,
)

__all__ = [
    "FIRS_GYM_ID",
    "FIRS_DETERMINISTIC_GYM_ID",
    "FIRS_OBSERVATION_SPEC",
    "OpenTTDFIRSGymEnv",
    "OpenTTDLEGymEnv",
    "TOY_GYM_ID",
    "make_firs",
    "make_firs_vector",
    "register_envs",
]
