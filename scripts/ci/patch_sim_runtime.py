"""Remove LIBERO's eager training imports and use converted initial-state assets.

This narrowly patches the pinned simulator distribution inside its image. The original
checkpoint/training environments are untouched. All initial-state loading becomes NumPy.
"""

import argparse
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    args = parser.parse_args()
    benchmarks = list(args.root.rglob("libero/benchmark/__init__.py"))
    if not benchmarks:
        raise ValueError("Pinned LIBERO benchmark module was not found")
    for path in benchmarks:
        source = path.read_text()
        if "import torch" not in source:
            raise ValueError(f"Expected pinned LIBERO torch import: {path}")
        source = source.replace("import torch", "import numpy as np")
        source = source.replace("torch.load(", "_load_numpy_states(")
        source += '\n\ndef _load_numpy_states(path, *args, **kwargs):\n    return np.load(str(path) + ".npy", allow_pickle=False)\n'
        path.write_text(source)
    for path in args.root.rglob("libero/utils/utils.py"):
        source = path.read_text()
        if "torch." in source:
            raise ValueError(f"Unexpected LIBERO training dependency in runtime utility: {path}")
        path.write_text(source.replace("import torch\n", ""))


if __name__ == "__main__":
    main()
