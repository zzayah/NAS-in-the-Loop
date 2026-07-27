#!/usr/bin/env python3
"""split a lidar dataset for the supervised architecture search."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

#
# ---------- START INPUT ----------
#

INPUT_PATH = "accuracy-nas/datasets/combined_all.npz"
OUTPUT_DIR = "accuracy-nas/datasets"
TRAIN_RATIO = 0.8
SEED = 0

#
# ---------- END INPUT ----------
#

def _write_split(path: Path, arrays: dict[str, np.ndarray], indices: np.ndarray) -> None:
    """Write one subset of the dataset arrays."""
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **{key: values[indices] for key, values in arrays.items()})


def split_dataset(
    input_path: str | Path,
    output_dir: str | Path,
    train_ratio: float = 0.8,
    seed: int = 0,
) -> dict[str, Path]:
    """Split a lidar dataset into train.npz and test.npz."""
    if train_ratio <= 0 or train_ratio >= 1:
        raise ValueError("train ratio must leave non-empty train and test splits.")

    with np.load(input_path) as data:
        arrays = {key: data[key] for key in data.files}
    if not arrays:
        raise ValueError(f"{input_path} has no arrays.")

    rows = {values.shape[0] for values in arrays.values()}
    if len(rows) != 1:
        raise ValueError("all dataset arrays must have the same first dimension.")
    total = rows.pop()

    indices = np.random.default_rng(seed).permutation(total)
    train_end = int(total * train_ratio)
    split_indices = {
        "train": indices[:train_end],
        "test": indices[train_end:],
    }

    output_root = Path(output_dir)
    paths = {name: output_root / f"{name}.npz" for name in split_indices}
    for name, split in split_indices.items():
        _write_split(paths[name], arrays, split)
    return paths


def main() -> None:
    """Split the dataset and print row counts."""
    paths = split_dataset(
        INPUT_PATH,
        OUTPUT_DIR,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )
    for name, path in paths.items():
        with np.load(path) as data:
            count = data[data.files[0]].shape[0]
        print(f"[{name}] {count} rows -> {path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    paths = split_dataset(
        INPUT_PATH,
        args.output_dir,
        train_ratio=TRAIN_RATIO,
        seed=SEED,
    )
    for name, path in paths.items():
        with np.load(path) as data:
            count = data[data.files[0]].shape[0]
        print(f"[{name}] {count} rows -> {path}")
