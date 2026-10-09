# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy at http://www.apache.org/licenses/LICENSE-2.0

"""OpenPI task prompt tokenization without state tokens; image inputs are unaffected."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from huggingface_hub import hf_hub_download

from lerobot.configs import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import ProcessorStep, ProcessorStepRegistry
from lerobot.utils.constants import OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS
from lerobot.utils.import_utils import require_package


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
