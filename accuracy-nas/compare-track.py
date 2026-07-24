#!/usr/bin/env python3
"""Run the Safety-NAS track comparison for an accuracy-nas checkpoint triplet."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

#
# ---------- INPUT ----------
#

# The inputs here are the .pt files from the test-best run (NN architecture and weights).
LEFT_WALL_MODEL = "accuracy-nas/dnn-output/test-best-150-combinedall-8020/7307d4/left_wall_dist_arch7_trial70.pt"
TRACK_WIDTH_MODEL = "accuracy-nas/dnn-output/test-best-150-combinedall-8020/7307d4/track_width_arch7_trial108.pt"
HEADING_ERROR_MODEL = "accuracy-nas/dnn-output/test-best-150-combinedall-8020/7307d4/heading_error_arch7_trial29.pt"
OUTPUT_DIR = "accuracy-nas/compare-map-150-combinedall-8020"

#
# ---------- END INPUT ----------
#

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


def _checkpoint_triplet_id() -> str:
    """Get the shared run ID for the three model paths."""
    parents = {
        Path(model_path).expanduser().parent.resolve()
        for model_path in (LEFT_WALL_MODEL, TRACK_WIDTH_MODEL, HEADING_ERROR_MODEL)
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


def main() -> None:
    """Compare the accuracy-NAS models against the baseline models."""
    compare = _load_safety_nas_compare()
    compare.ARGS.run = [
        *compare.BASELINE_RUNS,
        (
            "arch7",
            LEFT_WALL_MODEL,
            TRACK_WIDTH_MODEL,
            HEADING_ERROR_MODEL,
        )
    ]
    compare.ARGS.run_id = _checkpoint_triplet_id()
    compare.ARGS.output_dir = OUTPUT_DIR
    compare.main()


if __name__ == "__main__":
    main()
