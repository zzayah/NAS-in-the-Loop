#!/usr/bin/env python3
"""
Retrain or evaluate the best NAS trial and stage the resulting checkpoints.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nas.cnn import TRAIN_EVAL_TRACKS
from nas.testing import test_cnn_arch
from nas.training import orchestrate_best_trial, train_from_configs

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
        "label": "arch6-7",
        "max_epochs": 700,
        "lr": 5e-5,
        "weight_decay": 1e-4,
        "early_stopping_patience": 60,
        "lr_patience": 20,
        "optimizer": "adamw",
    },
}

# Configuration (edit as needed)
TRIALS_FILES = [
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T172706_1828022_a8b20f.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T172706_1828023_3d2630.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T172706_1828025_f847ab.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T184238_1357002_017f7c.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T193527_1776610_4ec05d.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T193845_2985264_dc559d.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T203158_3307492_60de23.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T203833_1888159_1b391f.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T210412_3311456_e6278a.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T212320_1900172_3cd867.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T213142_3317205_782a1c.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T213240_1902398_08e72b.jsonl",
    "nas/dnn-output/all-nas-runs/nas_trials_20260508T220850_3400979_d24a66.jsonl",
]
TRAINING_PROFILE = 0  # 0: arch1-2, 1: arch3-4, 2: arch5, 3: arch6-7
DATASET_PATH = "nas/datasets/combined_all.npz"
OUTPUT_DIR: str | None = "nas/dnn-output/test-best-runs-150"

MODE = "train"  # "train" or "test"
SKIP_EVAL = False

def _resolve_trials_file(arg: str | None) -> str | None:
    if arg is None:
        return None
    path = Path(arg).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Trials file {path} does not exist.")
    return str(path)


def _resolve_trials_files(args: Iterable[str] | str | None) -> list[str]:
    if args is None:
        return []
    if isinstance(args, str):
        resolved = _resolve_trials_file(args)
        return [] if resolved is None else [resolved]
    resolved_files: list[str] = []
    for arg in args:
        resolved = _resolve_trials_file(arg)
        if resolved is not None:
            resolved_files.append(resolved)
    return resolved_files


def _selected_training_profile() -> dict:
    try:
        profile = TRAINING_PROFILES[TRAINING_PROFILE]
    except KeyError as exc:
        valid = ", ".join(str(key) for key in sorted(TRAINING_PROFILES))
        raise ValueError(
            f"Unknown TRAINING_PROFILE={TRAINING_PROFILE}. Valid profiles: {valid}."
        ) from exc
    return profile


def _default_output_dir(trials_file: str | None) -> Path:
    if trials_file is None:
        raise ValueError("trials_file must be resolved before deriving output_dir.")
    trial_id = Path(trials_file).stem.rsplit("_", 1)[-1]
    return (REPO_ROOT / "nas/dnn-output/test-best-runs" / trial_id).resolve()


def _target_name(config_path: Path) -> str:
    stem = config_path.stem
    return stem if "_arch" not in stem else stem.rsplit("_arch", 1)[0]


def _stage_model(src: Path, dest: Path) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.resolve() != dest.resolve():
        shutil.copy2(src, dest)
    return dest


def _stage_trained_models(
    config_paths: list[Path], trained_paths: Iterable[Path]
) -> list[Path]:
    staged: list[Path] = []
    for cfg, checkpoint in zip(config_paths, trained_paths, strict=False):
        source = checkpoint if checkpoint.is_absolute() else (REPO_ROOT / checkpoint)
        staged.append(_stage_model(source, cfg.with_suffix(".pt")))
    return staged


def _summary_for_config(best_trial: dict, config_path: Path) -> dict | None:
    target = _target_name(config_path)
    for entry in best_trial.get("targets", []):
        if entry.get("target_col") == target:
            return entry
    return None


def _locate_model_from_summary(summary: dict | None) -> Path | None:
    if not summary:
        return None
    candidate = Path(summary["model_path"])
    if candidate.exists():
        return candidate
    return None


def _collect_existing_models(
    config_paths: list[Path], best_trial: dict
) -> tuple[list[Path], list[str]]:
    staged: list[Path] = []
    missing: list[str] = []
    for cfg in config_paths:
        staged_path = cfg.with_suffix(".pt")
        if staged_path.exists():
            staged.append(staged_path)
            continue
        summary = _summary_for_config(best_trial, cfg)
        candidate = _locate_model_from_summary(summary)
        if candidate:
            staged.append(_stage_model(candidate, staged_path))
        else:
            info = str(staged_path)
            if summary:
                info += (
                    f" (expected {summary['target_col']}_arch{summary['arch_id']}.pt)"
                )
            missing.append(info)
    return staged, missing


def _evaluate_models(model_paths: list[Path]) -> float | None:
    lookup = {_target_name(path): path for path in model_paths}
    required = {"left_wall_dist", "track_width", "heading_error"}
    missing = sorted(required - set(lookup))
    if missing:
        print(f"[warn] missing checkpoints for: {', '.join(missing)}")
        return None
    track_configs = [(t.map_path, t.waypoints_path) for t in TRAIN_EVAL_TRACKS]
    avg_rmse, _, _, _ = test_cnn_arch(
        left_wall_dist_filepath=str(lookup["left_wall_dist"]),
        track_width_filepath=str(lookup["track_width"]),
        heading_error_filepath=str(lookup["heading_error"]),
        track_configs=track_configs,
    )
    print(f"[rmse] average across {len(track_configs)} tracks: {avg_rmse:.4f}")
    return avg_rmse


def _run_trials_file(trials_file: str) -> None:
    if MODE not in {"train", "test"}:
        raise ValueError("MODE must be 'train' or 'test'")
    output_dir = (
        (Path(OUTPUT_DIR).expanduser().resolve() / Path(trials_file).stem.rsplit("_", 1)[-1])
        if OUTPUT_DIR is not None
        else _default_output_dir(trials_file)
    )
    print(f"[run] trials_file={trials_file}")
    print(f"[run] output_dir={output_dir}")
    training_profile = _selected_training_profile()
    print(f"[run] training_profile={TRAINING_PROFILE} ({training_profile['label']})")
    best_trial, config_paths = orchestrate_best_trial(
        trials_path=trials_file,
        dataset_path=DATASET_PATH,
        output_dir=output_dir,
        max_epochs=training_profile["max_epochs"],
        lr=training_profile["lr"],
        weight_decay=training_profile["weight_decay"],
        early_stopping_patience=training_profile["early_stopping_patience"],
        lr_patience=training_profile["lr_patience"],
        optimizer=training_profile["optimizer"],
    )

    print(
        f"[trial] #{best_trial['trial_number']} "
        f"avg_rmse={best_trial['rmse'][0]['value']:.4f}"
    )

    config_paths = [Path(cfg) for cfg in config_paths]
    if MODE == "train":
        print("[mode] retraining configs...")
        trained = train_from_configs(config_paths)
        model_paths = _stage_trained_models(config_paths, trained)
    else:
        print("[mode] staging existing checkpoints...")
        model_paths, missing = _collect_existing_models(config_paths, best_trial)
        if missing:
            print("[warn] checkpoints missing:")
            for item in missing:
                print(f"    {item}")
        if not model_paths:
            raise RuntimeError("No checkpoints available to stage or evaluate.")

    for cfg, model_path in zip(config_paths, model_paths, strict=False):
        print(f"[config] {cfg}")
        print(f"[model]  {model_path}")

    if not SKIP_EVAL:
        _evaluate_models(model_paths)


def main() -> None:
    if MODE not in {"train", "test"}:
        raise ValueError("MODE must be 'train' or 'test'")
    trials_files = _resolve_trials_files(TRIALS_FILES)
    if not trials_files:
        raise ValueError("TRIALS_FILES must contain at least one trials file.")

    for trials_file in trials_files:
        _run_trials_file(trials_file)


if __name__ == "__main__":
    main()
