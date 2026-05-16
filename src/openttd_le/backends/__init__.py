from __future__ import annotations

from openttd_le.backends.base import Backend
from openttd_le.backends.openttd import OpenTTDBackend
from openttd_le.backends.toy import ToyLogisticsBackend


def make_backend(name: str) -> Backend:
    if name == "toy":
        return ToyLogisticsBackend()
    if name == "openttd":
        return OpenTTDBackend()
    raise ValueError(f"Unknown backend: {name}")


__all__ = ["Backend", "OpenTTDBackend", "ToyLogisticsBackend", "make_backend"]
