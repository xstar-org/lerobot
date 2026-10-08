"""Optional world capability for simulated robots."""

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass(frozen=True)
class EpisodeStatus:
    success: bool
    terminated: bool
    truncated: bool
    step: int
    reward: float


@runtime_checkable
class SimWorld(Protocol):
    def reset_world(self, task: str | None = None, seed: int | None = None) -> None: ...
    @property
    def task_description(self) -> str: ...
    @property
    def episode_status(self) -> EpisodeStatus: ...
    @property
    def sim_time(self) -> float: ...
    def render(self): ...
