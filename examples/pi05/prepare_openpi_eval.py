#!/usr/bin/env python
"""Prepare a NEW standard lerobot-eval directory for a text-only OpenPI checkpoint.

Only for PyTorch PI0.5 checkpoints trained with discrete_state_input=False and
quantile normalization. The template supplies architecture and feature/camera
mapping, NOT normalization statistics. Explicitly verify those against training.
No weights or existing processor files are overwritten; no training is performed.

Example:
    python examples/pi05/prepare_openpi_eval.py \
        --template /path/to/compatible/lerobot/config \
        --checkpoint /path/to/openpi/30000/model.safetensors \
        --stats /path/to/openpi/30000/assets/DATASET/norm_stats.json \
        --tokenizer /path/to/training/tokenizer.model \
        --action-horizon 10 --output /path/to/new/eval_checkpoint

Then use the normal CLI (no monkey patches):
    MUJOCO_GL=osmesa PYOPENGL_PLATFORM=osmesa lerobot-eval \
        --policy.path=/path/to/new/eval_checkpoint --policy.compile_model=false \
        --env.type=libero --env.task=libero_spatial --env.task_ids='[0]' \
        --env.episode_length=600 --env.observation_height=256 --env.observation_width=256 \
        --eval.n_episodes=8 --eval.batch_size=8 --eval.use_async_envs=true \
        --eval.max_episodes_rendered=0 --seed=7 --output_dir=/path/to/new/results

The tokenizer file is copied into the output and its absolute location serialized;
if moving this directory, regenerate its processors with the relocated tokenizer.
Image resize matches OpenPI PIL, but camera mapping/orientation are still supplied
by the environment processor. Validate those for each benchmark/checkpoint.
"""

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import torch

from lerobot.configs import NormalizationMode
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from lerobot.policies.pi05.processor_pi05 import make_pi05_pre_post_processors
from lerobot.utils.constants import ACTION, OBS_STATE


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--stats", type=Path, required=True)
    parser.add_argument("--tokenizer", type=Path, required=True)
    parser.add_argument("--action-horizon", type=int, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.action_horizon < 1:
        parser.error("output must not exist and action-horizon must be positive")
    for path in (args.checkpoint, args.stats, args.tokenizer):
        if not path.is_file():
            parser.error(f"Missing file: {path}")
    config = PI05Config.from_pretrained(args.template)
    config.openpi_text_only = True
    config.openpi_quantile_normalization = True
    config.openpi_image_resize = True
    config.use_relative_actions = False
    config.use_visual_memory = False
    config.use_proprioceptive_memory = False
    config.chunk_size = config.n_action_steps = args.action_horizon
    config.rtc_config = None
    config.rtc_training_max_delay = 0
    config.device = "cuda"
    config.dtype = torch.float32
    config.compile_model = False
    config.normalization_mapping.update(STATE=NormalizationMode.QUANTILES, ACTION=NormalizationMode.QUANTILES)
    config.text_tokenizer_name = str((args.output / "tokenizer.model").resolve())
    config.pretrained_path = args.output.resolve()
    config.validate_features()
    source = json.loads(args.stats.read_text())["norm_stats"]
    stats = {}
    for key, source_key, feature in (
        (OBS_STATE, "state", config.input_features[OBS_STATE]),
        (ACTION, "actions", config.output_features[ACTION]),
    ):
        stats[key] = {name: torch.tensor(source[source_key][name], dtype=torch.float32) for name in ("q01", "q99")}
        if any(tuple(value.shape) != feature.shape or not torch.isfinite(value).all() for value in stats[key].values()):
            raise ValueError(f"{key}: stats must match the unpadded feature dimensions and be finite")
        if torch.any(stats[key]["q99"] < stats[key]["q01"]):
            raise ValueError(f"{key}: q99 must not be below q01")
    args.output.mkdir(parents=True)
    shutil.copy2(args.tokenizer, args.output / "tokenizer.model")
    config.save_pretrained(args.output)
    pre, post = make_pi05_pre_post_processors(config, stats)
    pre.save_pretrained(args.output)
    post.save_pretrained(args.output)
    # Raw OpenPI keys are converted by PI05Policy.from_pretrained, which now fails closed.
    shutil.copy2(args.checkpoint, args.output / "model.safetensors")
    manifest = {"action_horizon": args.action_horizon, "discrete_state_input": False, "quantile": True}
    for name, path in (("checkpoint", args.checkpoint), ("stats", args.stats), ("tokenizer", args.tokenizer)):
        with path.open("rb") as stream:
            manifest[name] = {"source": str(path.resolve()), "sha256": hashlib.file_digest(stream, "sha256").hexdigest()}
    (args.output / "openpi_eval_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Prepared {args.output}; validate camera mapping and run a smoke test before full evaluation.")


if __name__ == "__main__":
    main()
