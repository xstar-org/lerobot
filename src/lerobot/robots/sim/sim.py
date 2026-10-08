"""Realtime Robot face of a simulator session."""

import numpy as np

from lerobot.robots.robot import Robot
from lerobot.sim_server.client import SimClient
from lerobot.sim_server.contracts import StepResult
from lerobot.sim_server.profiles import EvalProfile

from .configuration_sim import SimRobotConfig
from .world import EpisodeStatus


class SimRobot(Robot):
    config_class = SimRobotConfig
    name = "sim"

    def __init__(self, config: SimRobotConfig):
        super().__init__(config)
        self.config = config
        self.client = SimClient(config.endpoint, config.deployment, config.timeout_s)
        try:
            self.descriptor = self.client.describe()
        except BaseException:
            self.client.close()
            raise
        try:
            self.profile = EvalProfile.load(config.profile) if config.profile else None
            if self.profile:
                self.profile.validate_descriptor(self.descriptor)
            self.latest = None
            self._state = next((f for f in self.descriptor.features if f.name == "observation.state"), None)
            if self._state is None or not self._state.names:
                raise ValueError("Robot face requires explicit state component names")
            self._camera_keys = {}
            for feature in self.descriptor.features:
                if feature.kind == "rgb":
                    canonical = (
                        self.profile.feature_mapping.get(feature.name, feature.name)
                        if self.profile
                        else feature.name
                    )
                    camera = canonical.removeprefix("observation.images.").removeprefix("observation.")
                    self._camera_keys[feature.name] = camera
        except BaseException:
            self.client.close()
            raise

    @property
    def supports_position_hold(self):
        return self.descriptor.hold == "position"

    @property
    def supports_command_hold(self):
        return self.descriptor.hold == "command"

    @property
    def observation_features(self):
        return {
            **dict.fromkeys(self._state.names, float),
            **{self._camera_keys[f.name]: f.shape for f in self.descriptor.features if f.kind == "rgb"},
        }

    @property
    def action_features(self):
        return dict.fromkeys(self.policy_action_names, float)

    @property
    def policy_action_names(self):
        return (
            self.profile.policy_action_names(self.descriptor)
            if self.profile
            else self.descriptor.action_feature.names
        )

    @property
    def is_connected(self):
        return bool(self.client.session) and not self.client.failed

    @property
    def is_calibrated(self):
        return True

    def calibrate(self):
        pass

    def configure(self):
        pass

    def connect(self, calibrate=True):
        if self.client.session:
            raise RuntimeError("Simulator robot is already connected")
        self.latest = self.client.open(1, "realtime", self.config.task, self.config.task_id)

    def _snapshot(self):
        self.latest = StepResult.from_dict(self.client.request("snapshot")["result"])
        return self.latest

    def get_observation(self):
        result = self._snapshot()
        return {
            **{
                key: float(value)
                for key, value in zip(self._state.names, result.obs[self._state.name][0], strict=True)
            },
            **{camera: result.obs[name][0] for name, camera in self._camera_keys.items()},
        }

    def send_action(self, action):
        names = self.policy_action_names
        if set(action) != set(names):
            raise ValueError("Sim action must contain exactly the declared components")
        values = np.asarray([[action[name] for name in names]], dtype=np.float32)
        applied = self.client.request("apply", actions=values)["applied"][0]
        return {name: float(value) for name, value in zip(names, applied, strict=True)}

    def hold(self):
        self.client.request("hold")

    def reset_world(self, task=None, seed=None):
        if task is not None and task != (self.config.task or next(iter(self.descriptor.tasks))):
            raise ValueError("Changing task requires opening a new robot session")
        self.latest = StepResult.from_dict(self.client.request("reset", seeds=[seed])["result"])

    @property
    def task_description(self):
        return self.latest.task[0] if self.latest is not None else ""

    @property
    def episode_status(self):
        result = self.latest
        if result is None:
            raise RuntimeError("Connect before reading world status")
        return EpisodeStatus(
            bool(result.is_success[0]),
            bool(result.terminated[0]),
            bool(result.truncated[0]),
            int(result.step[0]),
            float(result.reward[0]),
        )

    @property
    def sim_time(self):
        return self.latest.sim_time if self.latest is not None else 0.0

    def render(self):
        return self.client.request("render")["frames"][0]

    def disconnect(self):
        self.client.close()
