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

LEGACY_MODEL_DIRS = [
    (REPO_ROOT / "nas/models").resolve(),
    (REPO_ROOT / "data/models").resolve(),
]

# Configuration (edit as needed)
TRIALS_FILE = "nas/dnn-output/nas_trials_20260415T001852_385995_b175ea.jsonl" # e.g. "nas/dnn-output/nas_trials_20260411T184604.jsonl"
DATASET_PATH = "nas/datasets/combined_all.npz"
OUTPUT_DIR: str | None = None
MODE = "train"  # "train" or "test"
MAX_EPOCHS = 700
EARLY_STOPPING = 60
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
    fallback = f"{summary['target_col']}_arch{summary['arch_id']}.pt"
    for directory in LEGACY_MODEL_DIRS:
        alt = directory / fallback
        if alt.exists():
            return alt
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
    avg_rmse, _ = test_cnn_arch(
        left_wall_dist_filepath=str(lookup["left_wall_dist"]),
        track_width_filepath=str(lookup["track_width"]),
        heading_error_filepath=str(lookup["heading_error"]),
        track_configs=track_configs,
    )
    print(f"[rmse] average across {len(track_configs)} tracks: {avg_rmse:.4f}")
    return avg_rmse


def main() -> None:
    if MODE not in {"train", "test"}:
        raise ValueError("MODE must be 'train' or 'test'")
    trials_file = _resolve_trials_file(TRIALS_FILE)
    best_trial, config_paths = orchestrate_best_trial(
        trials_path=trials_file,
        dataset_path=DATASET_PATH,
        output_dir=OUTPUT_DIR,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING,
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


if __name__ == "__main__":
    main()
