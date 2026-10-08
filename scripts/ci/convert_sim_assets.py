"""Convert trusted LIBERO initial-state files during image build, before removing torch."""

import argparse
from pathlib import Path

import numpy as np


def main():
    import torch

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    count = 0
    for path in args.root.rglob("*.pruned_init"):
        value = torch.load(path, weights_only=False)  # nosec B614: trusted image-build assets
        if isinstance(value, torch.Tensor):
            value = value.cpu().numpy()
        array = np.asarray(value)
        if array.dtype.kind not in "fiu" or not np.isfinite(array).all():
            raise ValueError(f"Invalid initial states: {path}")
        np.save(path.with_suffix(path.suffix + ".npy"), array, allow_pickle=False)
        count += 1
    if not count:
        raise ValueError(f"No LIBERO initial-state assets found: {args.root}")
    print(f"Converted {count} initial-state assets")


if __name__ == "__main__":
    main()
