# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""Opt-in text-only Paligemma tokenization used by OpenPI fine-tuned checkpoints."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
import numpy as np
from huggingface_hub import hf_hub_download
from PIL import Image

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package


@ProcessorStepRegistry.register(name="pi05_openpi_image_resize")
@dataclass
class OpenPIImageResizeProcessorStep(ProcessorStep):
    """Match OpenPI client's uint8 PIL bilinear resize before model preprocessing.

    Input is LeRobot's floating RGB tensor in [0, 1], BCHW (or BTCHW).
    Camera mapping and orientation remain the environment processor's responsibility.
    """

    image_keys: list[str]
    size: tuple[int, int] = (224, 224)

    def __post_init__(self) -> None:
        if len(self.size) != 2 or min(self.size) < 1:
            raise ValueError("size must contain two positive dimensions")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        result = transition.copy()
        observation = dict(transition.get(TransitionKey.OBSERVATION) or {})
        height, width = self.size
        for key in self.image_keys:
            if key not in observation:  # Empty cameras are masked by the policy.
                continue
            x = observation[key]
            if x.ndim not in (4, 5) or x.shape[-3] != 3 or not x.is_floating_point():
                raise ValueError(f"{key}: expected floating BCHW or BTCHW RGB in [0, 1]")
            if not torch.isfinite(x).all() or x.min() < 0 or x.max() > 1:
                raise ValueError(f"{key}: pixels must be finite and in [0, 1]")
            leading = x.shape[:-3]
            images = (x.detach().flatten(0, -4).cpu().permute(0, 2, 3, 1) * 255).round()
            images = images.to(torch.uint8).numpy()
            resized = []
            for pixels in images:
                image = Image.fromarray(pixels)
                ratio = max(image.width / width, image.height / height)
                new_size = (int(image.width / ratio), int(image.height / ratio))
                image = image.resize(new_size, resample=Image.Resampling.BILINEAR)
                canvas = Image.new("RGB", (width, height), 0)
                canvas.paste(image, ((width - new_size[0]) // 2, (height - new_size[1]) // 2))
                resized.append(np.asarray(canvas))
            output = torch.from_numpy(np.stack(resized)).permute(0, 3, 1, 2).float() / 255
            observation[key] = output.reshape(*leading, 3, height, width).to(x.device)
        result[TransitionKey.OBSERVATION] = observation
        return result

    def get_config(self) -> dict[str, Any]:
        return {"image_keys": self.image_keys, "size": list(self.size)}

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {key: value.copy() for key, value in features.items()}
        obs = features.setdefault(PipelineFeatureType.OBSERVATION, {})
        for key in self.image_keys:
            if key in obs:
                obs[key] = PolicyFeature(type=FeatureType.VISUAL, shape=(*obs[key].shape[:-2], *self.size))
        return features


@ProcessorStepRegistry.register(name="pi05_openpi_text_tokenizer")
@dataclass
class OpenPITextTokenizerProcessorStep(ProcessorStep):
    """BOS + cleaned task + separately encoded newline, with no Task/State/Action wrapper.

    ``tokenizer_name`` accepts a SentencePiece file, a local directory containing
    tokenizer.model, or a Hub repository. Pin ``revision`` for Hub reproducibility.
    The model is loaded lazily; simply importing PI05 does not require SentencePiece.
    """

    tokenizer_name: str = "google/paligemma-3b-pt-224"
    revision: str | None = None
    max_length: int = 200
    task_key: str = "task"
    _tokenizer: Any = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.max_length < 1:
            raise ValueError("max_length must be positive")

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        tasks = (transition.get(TransitionKey.COMPLEMENTARY_DATA) or {}).get(self.task_key)
        observation = transition.get(TransitionKey.OBSERVATION)
        if observation is None or not isinstance(tasks, list) or not all(isinstance(t, str) for t in tasks):
            raise ValueError("OpenPI tokenization requires an observation and a batched list of task strings")
        if self._tokenizer is None:
            require_package("sentencepiece", "pi")
            import sentencepiece

            path = Path(self.tokenizer_name)
            if path.is_dir():
                path = path / "tokenizer.model"
            elif not path.is_file():
                path = Path(hf_hub_download(self.tokenizer_name, "tokenizer.model", revision=self.revision))
            self._tokenizer = sentencepiece.SentencePieceProcessor(model_file=str(path))

        tokens = torch.zeros((len(tasks), self.max_length), dtype=torch.long)
        mask = torch.zeros_like(tokens, dtype=torch.bool)
        for i, task in enumerate(tasks):
            clean = task.strip().replace("_", " ").replace("\n", " ")
            ids = self._tokenizer.encode(clean, add_bos=True) + self._tokenizer.encode("\n")
            ids = ids[: self.max_length]
            tokens[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            mask[i, : len(ids)] = True
        result = transition.copy()
        result[TransitionKey.OBSERVATION] = {
            **observation,
            OBS_LANGUAGE_TOKENS: tokens,
            OBS_LANGUAGE_ATTENTION_MASK: mask,
        }
        return result

    def get_config(self) -> dict[str, Any]:
        return {
            "tokenizer_name": self.tokenizer_name,
            "revision": self.revision,
            "max_length": self.max_length,
            "task_key": self.task_key,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        features = {key: value.copy() for key, value in features.items()}
        obs = features.setdefault(PipelineFeatureType.OBSERVATION, {})
        for key in (OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
            obs[key] = PolicyFeature(type=FeatureType.LANGUAGE, shape=(self.max_length,))
        return features
