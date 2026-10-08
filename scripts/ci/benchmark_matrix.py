"""Generate the CI matrix and content identities from the benchmark manifest."""

import argparse
import hashlib
import json
from pathlib import Path

import yaml


def image_hash(root: Path, entry: dict) -> str:
    digest = hashlib.sha256()
    paths = [
        root / entry["dockerfile"],
        root / "docker/Dockerfile.sim-base",
        root / "docker/sims/requirements.txt",
        root / entry["sim_config"],
        root / "scripts/ci/convert_sim_assets.py",
        root / "scripts/ci/patch_sim_runtime.py",
        root / "scripts/ci/check_sim_parity.py",
        root / "pyproject.toml",
        root / "setup.py",
        root / "README.md",
        root / "MANIFEST.in",
        root / "src/lerobot/__init__.py",
        root / "src/lerobot/__version__.py",
    ]
    for directory in (
        "src/lerobot/sim_server",
        "src/lerobot/sims",
        "src/lerobot/transport",
        "src/lerobot/utils",
    ):
        paths.extend((root / directory).rglob("*.py"))
    paths.append(root / "src/lerobot/sims/native/metaworld_config.json")
    paths.extend(path for path in (root / "docker/sims").rglob("*") if path.is_file())
    for path in sorted(set(paths)):
        digest.update(str(path.relative_to(root)).encode())
        digest.update(path.read_bytes())
    digest.update(json.dumps(entry.get("build_args", {}), sort_keys=True).encode())
    return digest.hexdigest()[:24]


def load_entries(root: Path) -> list[dict]:
    entries = yaml.safe_load((root / "benchmarks/benchmarks.yaml").read_text())["benchmarks"]
    if len({e["name"] for e in entries}) != len(entries):
        raise ValueError("Duplicate benchmark name")
    return [{**entry, "sim_tag": image_hash(root, entry)} for entry in entries]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--name")
    args = parser.parse_args()
    entries = load_entries(args.root)
    if args.name:
        print(json.dumps(next(e for e in entries if e["name"] == args.name)))
    else:
        print(json.dumps({"include": entries}))


if __name__ == "__main__":
    main()
