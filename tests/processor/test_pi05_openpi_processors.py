"""CPU-only tests: no checkpoint download, GPU, simulator, or gated tokenizer required."""

import io
import os
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
import torch

from lerobot.configs import FeatureType, NormalizationMode, PolicyFeature
from lerobot.lerobot_types import TransitionKey
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import (
    OpenPITextTokenizerProcessorStep,
    make_pi05_pre_post_processors,
)
from lerobot.processor.factory import load_pretrained_policy_processors
from lerobot.utils.constants import ACTION, OBS_LANGUAGE_ATTENTION_MASK, OBS_LANGUAGE_TOKENS, OBS_STATE


@pytest.fixture
def tokenizer_file(tmp_path):
    sp = pytest.importorskip("sentencepiece")
    buffer = io.BytesIO()
    sp.SentencePieceTrainer.train(
        sentence_iterator=iter(["pick up the cup", "put bowl on plate", "open drawer", "close drawer"]),
        model_writer=buffer,
        vocab_size=64,
        hard_vocab_limit=False,
        minloglevel=2,
    )
    path = tmp_path / "tokenizer.model"
    path.write_bytes(buffer.getvalue())
    return path


@pytest.mark.parametrize("length", [3, 20])
def test_native_text_tokens_and_mask(tokenizer_file, length):
    import sentencepiece as sp

    processor = OpenPITextTokenizerProcessorStep(str(tokenizer_file), max_length=length)
    tasks = ["  pick_up\nthe cup  ", ""]
    transition = {TransitionKey.OBSERVATION: {}, TransitionKey.COMPLEMENTARY_DATA: {"task": tasks}}
    output = processor(transition)[TransitionKey.OBSERVATION]
    reference = sp.SentencePieceProcessor(model_file=str(tokenizer_file))
    for i, text in enumerate(tasks):
        clean = text.strip().replace("_", " ").replace("\n", " ")
        ids = (reference.encode(clean, add_bos=True) + reference.encode("\n"))[:length]
        assert output[OBS_LANGUAGE_TOKENS][i].tolist() == ids + [0] * (length - len(ids))
        assert output[OBS_LANGUAGE_ATTENTION_MASK][i].tolist() == [True] * len(ids) + [False] * (
            length - len(ids)
        )
    assert transition[TransitionKey.OBSERVATION] == {}
    assert transition[TransitionKey.COMPLEMENTARY_DATA]["task"] == tasks


def test_saved_pipeline_round_trip(tokenizer_file, tmp_path):
    cfg = PI05Config(
        device="cpu",
        openpi_prompt_without_state=True,
        normalization_mapping={"STATE": NormalizationMode.QUANTILES, "ACTION": NormalizationMode.QUANTILES},
        text_tokenizer_name=str(tokenizer_file),
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (2,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (2,))},
        chunk_size=10,
        n_action_steps=10,
    )
    stats = {key: {"q01": torch.zeros(2), "q99": torch.ones(2)} for key in (OBS_STATE, ACTION)}
    pre, post = make_pi05_pre_post_processors(cfg, stats)
    folder = tmp_path / "converted"
    pre.save_pretrained(folder)
    post.save_pretrained(folder)
    loaded_pre, loaded_post = load_pretrained_policy_processors(folder)
    batch = {OBS_STATE: torch.tensor([[0.2, 0.7]]), "task": ["pick up the cup"]}
    expected, actual = pre(batch), loaded_pre(batch)
    for key in (OBS_STATE, OBS_LANGUAGE_TOKENS, OBS_LANGUAGE_ATTENTION_MASK):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    action = torch.tensor([[[3.0, -2.0]]])
    torch.testing.assert_close(post(action), loaded_post(action), rtol=0, atol=0)


def test_legacy_default_language_pipeline_unchanged():
    cfg = PI05Config(device="cpu", input_features={}, output_features={})
    with patch("lerobot.processor.tokenizer_processor.AutoTokenizer.from_pretrained"):
        pre, _ = make_pi05_pre_post_processors(cfg)
    names = [type(step).__name__ for step in pre.steps]
    assert "Pi05PrepareStateTokenizerProcessorStep" in names
    assert "TokenizerProcessorStep" in names
    assert "OpenPITextTokenizerProcessorStep" not in names


@pytest.mark.skipif(
    not os.environ.get("PI05_PARITY_DIR"), reason="Requires locally recorded OpenPI observations"
)
def test_recorded_openpi_tokenizer_parity(tmp_path):
    """Optional integration gate against real OpenPI tensors; fixtures are never downloaded.

    PI05_PARITY_DIR contains fixed-observations.npz and native-30000-float32/outputs.npz.
    PI05_PARITY_TOKENIZER points to the training tokenizer file.
    """
    root = Path(os.environ["PI05_PARITY_DIR"])
    raw = np.load(root / "fixed-observations.npz")
    native = np.load(root / "native-30000-float32/outputs.npz")
    cfg = PI05Config(
        device="cpu",
        openpi_prompt_without_state=True,
        normalization_mapping={"STATE": NormalizationMode.QUANTILES, "ACTION": NormalizationMode.QUANTILES},
        text_tokenizer_name=os.environ["PI05_PARITY_TOKENIZER"],
        input_features={OBS_STATE: PolicyFeature(FeatureType.STATE, (8,))},
        output_features={ACTION: PolicyFeature(FeatureType.ACTION, (7,))},
        chunk_size=10,
        n_action_steps=10,
    )
    stats = {
        key: {
            "q01": torch.zeros(8 if key == OBS_STATE else 7),
            "q99": torch.ones(8 if key == OBS_STATE else 7),
        }
        for key in (OBS_STATE, ACTION)
    }
    pre, post = make_pi05_pre_post_processors(cfg, stats)
    pre.save_pretrained(tmp_path)
    post.save_pretrained(tmp_path)
    pre, post = load_pretrained_policy_processors(tmp_path)
    batch = {OBS_STATE: torch.from_numpy(raw["state"]), "task": raw["prompt"].tolist()}
    actual = pre(batch)
    for key, name in ((OBS_LANGUAGE_TOKENS, "tokens"), (OBS_LANGUAGE_ATTENTION_MASK, "masks")):
        np.testing.assert_array_equal(actual[key].numpy(), native[name])
