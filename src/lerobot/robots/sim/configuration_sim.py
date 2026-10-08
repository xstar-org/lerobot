from dataclasses import dataclass

from lerobot.robots.config import RobotConfig


@RobotConfig.register_subclass("sim")
@dataclass
class SimRobotConfig(RobotConfig):
    endpoint: str = "tcp/127.0.0.1:7448"
    deployment: str = "default"
    task: str | None = None
    task_id: int = 0
    timeout_s: float = 120
    profile: str | None = None
