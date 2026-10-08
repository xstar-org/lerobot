"""Gymnasium vector face of the simulator protocol; no native simulator imports."""

import gymnasium as gym
import numpy as np
from gymnasium.vector.utils import batch_space

from lerobot.sim_server.client import SimClient
from lerobot.sim_server.contracts import StepResult


class SimVectorEnv(gym.vector.VectorEnv):
    canonical_observations = True

    def __init__(
        self,
        endpoint,
        deployment="default",
        num_envs=1,
        task_group=None,
        task_id=0,
        timeout_s=120,
        descriptor=None,
        profile=None,
    ):
        self.client = SimClient(endpoint, deployment, timeout_s)
        self.descriptor = descriptor or self.client.describe()
        self.num_envs = num_envs
        self.task_group = task_group or next(iter(self.descriptor.tasks))
        self.task_id = task_id
        self.profile = profile
        if profile is not None:
            profile.validate_descriptor(self.descriptor)
        self.metadata = {"render_fps": self.descriptor.fps, "autoreset_mode": "Disabled"}
        self.render_mode = "rgb_array"
        self.single_observation_space = gym.spaces.Dict(
            {
                f.name: gym.spaces.Box(0, 255, f.shape, np.uint8)
                if f.kind == "rgb"
                else gym.spaces.Box(-np.inf, np.inf, f.shape, np.dtype(f.dtype))
                for f in self.descriptor.features
            }
        )
        self.single_action_space = gym.spaces.Box(
            -np.inf, np.inf, self.descriptor.action_feature.shape, np.float32
        )
        self.observation_space = batch_space(self.single_observation_space, num_envs)
        self.action_space = batch_space(self.single_action_space, num_envs)
        self.latest = None
        self.closed = False

    def _ensure(self, seeds=None):
        if not self.client.session:
            self.latest = self.client.open(self.num_envs, "lockstep", self.task_group, self.task_id, seeds)
            self.descriptor = self.client.descriptor
            return True
        return False

    def _result(self, body):
        self.latest = StepResult.from_dict(body["result"])
        return self.latest

    def reset(self, *, seed=None, options=None):
        if isinstance(seed, int):
            seed = [seed + i for i in range(self.num_envs)]
        result = self.latest if self._ensure(seed) else self._result(self.client.request("reset", seeds=seed))
        return result.obs, {"is_success": result.is_success}

    def step(self, actions):
        self._ensure()
        result = self._result(
            self.client.request("step", actions=np.asarray(actions, dtype=np.float32), n_ticks=1)
        )
        return (
            result.obs,
            result.reward,
            result.terminated,
            result.truncated,
            {"is_success": result.is_success},
        )

    def call(self, name, *args, **kwargs):
        if name == "_max_episode_steps":
            return (self.descriptor.max_episode_steps,) * self.num_envs
        if name == "task":
            return (self.task_group,) * self.num_envs
        self._ensure()
        if name == "task_description":
            return self.latest.task
        if name == "render":
            return tuple(self.client.request("render")["frames"])
        raise AttributeError(f"Simulator vector attribute is not exposed: {name}")

    def get_attr(self, name):
        return self.call(name)

    def render(self):
        return self.call("render")

    def close_extras(self, **kwargs):
        self.client.close()

    def close(self, **kwargs):
        self.client.close()
        self.closed = True


def prepare_canonical_observation(observations):
    """Generic dataset-value conversion, once, on the policy side."""
    import torch

    result = {}
    for name, value in observations.items():
        tensor = torch.from_numpy(value)
        if value.dtype == np.uint8 and value.ndim == 4:
            tensor = tensor.permute(0, 3, 1, 2).contiguous().float() / 255
        elif tensor.is_floating_point():
            tensor = tensor.float()
        result[name] = tensor
    return result
