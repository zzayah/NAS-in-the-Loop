#!/usr/bin/env python3
"""retrain the best supervised model for each lidar target."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
SAFETY_NAS_DIR = REPO_ROOT / "safety-nas"
if str(SAFETY_NAS_DIR) not in sys.path:
    sys.path.insert(0, str(SAFETY_NAS_DIR))

from cnn import TRAIN_EVAL_TRACKS
from testing import test_cnn_arch
from training import export_trial_configs, train_from_configs

TRAINING_PROFILES = {
    0: {
        "label": "arch1-2",
        "max_epochs": 150,
        "lr": 1e-3,
        "weight_decay": 1e-5,
        "early_stopping_patience": 15,
        "lr_patience": 10,
        "optimizer": "adam",
    },
    1: {
        "label": "arch3-4",
        "max_epochs": 250,
        "lr": 5e-4,
        "weight_decay": 1e-5,
        "early_stopping_patience": 20,
        "lr_patience": 10,
        "optimizer": "adam",
    },
    2: {
        "label": "arch5",
        "max_epochs": 450,
        "lr": 1e-4,
        "weight_decay": 1e-5,
        "early_stopping_patience": 60,
        "lr_patience": 20,
        "optimizer": "adam",
    },
    3: {
        "label": "arch6",
        "max_epochs": 700,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "early_stopping_patience": 60,
        "lr_patience": 20,
        "optimizer": "adamw",
    },
}

#
# ---------- START INPUT ----------
#

# Input the paths to the .jsonl files outputted from the accuracy-NAS.
# Note that each of these files has the same ID; in this set, that ID is 7307d4.
TARGET_FILES = {
    "left_wall_dist": "",   # Ex. "accuracy-nas/dnn-output/nas-with-train80-test20/standard_trials_left_wall_dist_20260524T041713_62533_7307d4.jsonl",
    "track_width": "",      # Ex. "accuracy-nas/dnn-output/nas-with-train80-test20/standard_trials_track_width_20260524T041713_62533_7307d4.jsonl",
    "heading_error": ""     # Ex. "accuracy-nas/dnn-output/nas-with-train80-test20/standard_trials_heading_error_20260524T041713_62533_7307d4.jsonl",
}

# Adjust output_dir depending on training profile
TRAINING_PROFILE = 0    # 0: arch1-2, 1: arch3-4, 2: arch5, 3: arch6
TRAIN_PATH = "accuracy-nas/datasets/combined_all.npz"
OUTPUT_DIR = "data/accuracy-nas/dnn-output/test-best-150-tp0"
SKIP_EVAL = False

#
# ---------- END INPUT ----------
#

def _trials_file_id(target: str, trials_file: str | Path) -> str:
    """Get the run ID from an accuracy-NAS trials filename."""
    stem = Path(trials_file).stem
    prefix = f"standard_trials_{target}_"
    if not stem.startswith(prefix):
        raise ValueError(f"Trials file {trials_file} must be named {prefix}<run>.jsonl.")
    return stem[len(prefix):]


def _composite_id(target_files: dict[str, str]) -> str:
    """Build the output ID for the selected trials files."""
    source_ids = {
        target: _trials_file_id(target, trials_file)
        for target, trials_file in target_files.items()
    }
    if len(set(source_ids.values())) == 1:
        return next(iter(source_ids.values()))
    return "_".join(
        (
            f"left-{source_ids['left_wall_dist']}",
            f"track-{source_ids['track_width']}",
            f"heading-{source_ids['heading_error']}",
        )
    )


def _load_best(path: str | Path) -> dict[str, Any]:
    """Load the trial with the lowest validation objective."""
    with Path(path).open("r", encoding="utf-8") as handle:
        entries = [json.loads(line) for line in handle if line.strip()]
    if not entries:
        raise RuntimeError(f"no trials found in {path}")
    return min(entries, key=lambda entry: float(entry["objective"]["value"]))


def _selected_training_profile() -> dict[str, Any]:
    """Return the selected training profile."""
    try:
        return TRAINING_PROFILES[TRAINING_PROFILE]
    except KeyError as exc:
        valid = ", ".join(str(key) for key in sorted(TRAINING_PROFILES))
        raise ValueError(
            f"Unknown TRAINING_PROFILE={TRAINING_PROFILE}. Valid profiles: {valid}."
        ) from exc


def _stage_model(config_path: Path, model_path: Path) -> Path:
    """Copy a trained model next to its config file."""
    staged = config_path.with_suffix(".pt")
    if staged.resolve() != model_path.resolve():
        shutil.copy2(model_path, staged)
    return staged


def _target_name(path: Path) -> str:
    """Get the target name from a config or model path."""
    return path.stem.split("_arch", 1)[0]


def _evaluate_models(model_paths: list[Path]) -> float | None:
    """Evaluate a staged model triplet on the training tracks."""
    lookup = {_target_name(path): path for path in model_paths}
    required = {"left_wall_dist", "track_width", "heading_error"}
    missing = sorted(required - set(lookup))
    if missing:
        print(f"[warn] missing checkpoints for: {', '.join(missing)}")
        return None

    track_configs = [(track.map_path, track.waypoints_path) for track in TRAIN_EVAL_TRACKS]
    avg_rmse, _, _, _ = test_cnn_arch(
        left_wall_dist_filepath=str(lookup["left_wall_dist"]),
        track_width_filepath=str(lookup["track_width"]),
        heading_error_filepath=str(lookup["heading_error"]),
        track_configs=track_configs,
    )
    print(f"[rmse] average across {len(track_configs)} tracks: {avg_rmse:.4f}")
    return avg_rmse


def main(
    target_files: dict[str, str],
    output_dir: str,
) -> None:
    """Retrain the selected accuracy-NAS models."""
    global TARGET_FILES, OUTPUT_DIR
    TARGET_FILES = target_files
    OUTPUT_DIR = output_dir
    composite_id = _composite_id(TARGET_FILES)
    output_dir = Path(OUTPUT_DIR).expanduser().resolve() / composite_id
    training_profile = _selected_training_profile()
    print(f"[run] composite_id={composite_id}")
    print(f"[run] output_dir={output_dir}")
    print(f"[run] training_profile={TRAINING_PROFILE} ({training_profile['label']})")
    configs: list[Path] = []
    for target, trials_file in TARGET_FILES.items():
        best = _load_best(trials_file)
        target_configs = export_trial_configs(
            best,
            output_dir,
            dataset_path=TRAIN_PATH,
            max_epochs=training_profile["max_epochs"],
            lr=training_profile["lr"],
            weight_decay=training_profile["weight_decay"],
            early_stopping_patience=training_profile["early_stopping_patience"],
            lr_patience=training_profile["lr_patience"],
            optimizer=training_profile["optimizer"],
        )
        configs.extend(target_configs)
        print(
            f"[best] {target} trial #{best['trial_number']} "
            f"validation_rmse={best['objective']['value']:.6f}"
        )

    models = train_from_configs(configs)
    staged_models: list[Path] = []
    for config_path, model_path in zip(configs, models, strict=True):
        staged = _stage_model(config_path, model_path)
        staged_models.append(staged)
        print(f"[model] {_target_name(staged)} -> {staged}")
    if not SKIP_EVAL:
        _evaluate_models(staged_models)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--left-trials", required=True)
    parser.add_argument("--track-trials", required=True)
    parser.add_argument("--heading-trials", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    main(
        target_files={
            "left_wall_dist": args.left_trials,
            "track_width": args.track_trials,
            "heading_error": args.heading_trials,
        },
        output_dir=args.output_dir,
    )
