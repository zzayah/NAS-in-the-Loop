#!/usr/bin/env python3
"""Run the Safety-NAS track comparison for an accuracy-nas checkpoint triplet."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SAFETY_NAS_COMPARE_PATH = REPO_ROOT / "safety-nas" / "compare-track.py"


def _load_safety_nas_compare():
    """Load the Safety-NAS comparison script."""
    spec = importlib.util.spec_from_file_location("safety_nas_compare_track", SAFETY_NAS_COMPARE_PATH)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load Safety-NAS comparison module from {SAFETY_NAS_COMPARE_PATH}.")

    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _checkpoint_triplet_id(
    left_model: str,
    track_model: str,
    heading_model: str,
) -> str:
    """Get the shared run ID for the three model paths."""
    parents = {
        Path(model_path).expanduser().parent.resolve()
        for model_path in (left_model, track_model, heading_model)
    }
    if len(parents) != 1:
        raise ValueError(
            "The three model paths must share one composite checkpoint directory."
        )

    composite_id = next(iter(parents)).name
    if composite_id in {"", ".", "dnn-output", "accuracy-nas", "test-best"}:
        raise ValueError(
            "Model paths must live under test-best/<composite-id>/ so compare-map "
            "can keep runs separate."
        )
    return composite_id


def main(
    left_model: str,
    track_model: str,
    heading_model: str,
    output_dir: str,
) -> None:
    """Compare the accuracy-NAS models against the baseline models."""
    compare = _load_safety_nas_compare()
    compare.ARGS.run = [
        *compare.BASELINE_RUNS,
        (
            "arch7",
            left_model,
            track_model,
            heading_model,
        )
    ]
    compare.ARGS.run_id = _checkpoint_triplet_id(
        left_model,
        track_model,
        heading_model,
    )
    compare.ARGS.output_dir = output_dir
    compare.main()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-model", required=True)
    parser.add_argument("--track-model", required=True)
    parser.add_argument("--heading-model", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(
        left_model=args.left_model,
        track_model=args.track_model,
        heading_model=args.heading_model,
        output_dir=args.output_dir,
    )
