#!/usr/bin/env python3
"""
Retrain or evaluate the best NAS trial and stage the resulting checkpoints.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path
from typing import Iterable

import yaml

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from nas.cnn import TRAIN_EVAL_TRACKS
from nas.testing import test_cnn_arch
from nas.training import orchestrate_best_trial, train_from_configs

# Configuration (edit as needed)
TRIALS_FILES = [
    "nas/dnn-output/nas_trials_20260504T024406_360276_f926b9.jsonl",
    # "nas/dnn-output/nas_trials_20260504T024406_360279_be986b.jsonl",
    "nas/dnn-output/nas_trials_20260504T024406_360280_193215.jsonl"
]
DATASET_PATH = "nas/datasets/combined_all.npz"
OUTPUT_DIR: str | None = None
MODE = "train"  # "train" or "test"
MAX_EPOCHS = 700
EARLY_STOPPING = 60
SKIP_EVAL = False
# LEFT_WALL_RESUME_CKPT: str | None = "nas/20260415T001238_47186_15d6f3_trial00113/left_wall_dist/checkpoints/left_wall_dist_arch8/best-epoch=19.ckpt"
# TRACK_WIDTH_RESUME_CKPT: str | None = "nas/20260415T001238_47186_15d6f3_trial00113/track_width/checkpoints/track_width_arch8/best-epoch=27.ckpt"
# HEADING_ERROR_RESUME_CKPT: str | None = "nas/20260415T001238_47186_15d6f3_trial00113/heading_error/checkpoints/heading_error_arch8/best-epoch=29.ckpt"
LEFT_WALL_RESUME_CKPT: str | None = None
TRACK_WIDTH_RESUME_CKPT: str | None = None
HEADING_ERROR_RESUME_CKPT: str | None = None


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


def _default_output_dir(trials_file: str | None) -> Path:
    if trials_file is None:
        raise ValueError("trials_file must be resolved before deriving output_dir.")
    trial_id = Path(trials_file).stem.rsplit("_", 1)[-1]
    return (REPO_ROOT / "nas/dnn-output/test-best-runs" / trial_id).resolve()


def _target_name(config_path: Path) -> str:
    stem = config_path.stem
    return stem if "_arch" not in stem else stem.rsplit("_arch", 1)[0]


def _resolve_optional_path(path_str: str | None) -> Path | None:
    if not path_str:
        return None
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (REPO_ROOT / path).resolve()
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint file {path} does not exist.")
    return path


def _load_yaml(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as fh:
        return yaml.safe_load(fh)


def _write_yaml(path: Path, payload: dict) -> None:
    with path.open("w", encoding="utf-8") as fh:
        yaml.safe_dump(payload, fh, sort_keys=False)


def _resume_checkpoint_overrides() -> dict[str, Path]:
    overrides: dict[str, Path] = {}
    left = _resolve_optional_path(LEFT_WALL_RESUME_CKPT)
    track = _resolve_optional_path(TRACK_WIDTH_RESUME_CKPT)
    heading = _resolve_optional_path(HEADING_ERROR_RESUME_CKPT)
    if left is not None:
        overrides["left_wall_dist"] = left
    if track is not None:
        overrides["track_width"] = track
    if heading is not None:
        overrides["heading_error"] = heading
    return overrides


def _apply_resume_checkpoint_overrides(config_paths: list[Path]) -> None:
    overrides = _resume_checkpoint_overrides()
    if not overrides:
        return

    staged_root = config_paths[0].parent / "_resume_checkpoints"
    staged_root.mkdir(parents=True, exist_ok=True)

    for cfg_path in config_paths:
        cfg = _load_yaml(cfg_path)
        target_col = cfg["data"]["target_col"]
        ckpt_src = overrides.get(target_col)
        if ckpt_src is None:
            continue

        arch_id = int(cfg["model"]["arch_id"])
        model_name = f"{target_col}_arch{arch_id}"
        ckpt_dest_dir = staged_root / model_name
        ckpt_dest_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dest = ckpt_dest_dir / "last.ckpt"
        shutil.copy2(ckpt_src, ckpt_dest)

        cfg.setdefault("training", {})
        cfg["training"]["resume"] = True
        cfg.setdefault("artifacts", {})
        cfg["artifacts"]["checkpoint_dir"] = str(staged_root)
        _write_yaml(cfg_path, cfg)
        print(f"[resume] {target_col}: {ckpt_src} -> {ckpt_dest}")


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
    best_trial, config_paths = orchestrate_best_trial(
        trials_path=trials_file,
        dataset_path=DATASET_PATH,
        output_dir=output_dir,
        max_epochs=MAX_EPOCHS,
        early_stopping_patience=EARLY_STOPPING,
    )

    print(
        f"[trial] #{best_trial['trial_number']} "
        f"avg_rmse={best_trial['rmse'][0]['value']:.4f}"
    )

    config_paths = [Path(cfg) for cfg in config_paths]
    _apply_resume_checkpoint_overrides(config_paths)
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
