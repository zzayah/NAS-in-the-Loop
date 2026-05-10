"""Helpers for generating training configs and launching Lightning trainers."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Sequence

import yaml

try:  # Allows both ``python -m nas.training`` and ``python nas/training.py``
    from nas.cnn import _build_training_config
except ModuleNotFoundError:  # pragma: no cover - fallback for script usage
    from nas.testing import test_cnn_arch  # ensure nested import works
    from nas.cnn import _build_training_config

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
TRAIN_SCRIPT = (REPO_ROOT / "packages/f110_scripts/src/f110_scripts/train/train_nn.py").resolve()
DEFAULT_DATASET = "nas/datasets/combined_all.npz"
TRIALS_DIR = BASE_DIR / "dnn-output"
DEFAULT_CONFIG_OUTPUT_ROOT = TRIALS_DIR / "best_configs"


def _model_block_from_summary(target_summary: dict[str, Any]) -> dict[str, Any]:
    """Recreate a ``model`` block from the NAS JSON summary."""
    return {
        "arch_id": target_summary["arch_id"],
        "dynamic": {
            "in_channels": target_summary.get("in_channels", 1),
            "input_length": target_summary.get("input_length", 1080),
            "activation": target_summary.get("activation", "elu"),
            "conv_layers": target_summary.get("conv_layers", []),
            "fc_layers": target_summary.get("fc_layers", []),
        },
    }


def export_trial_configs(
    trial_entry: dict[str, Any],
    output_dir: str | Path,
    dataset_path: str = DEFAULT_DATASET,
    max_epochs: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    lr_patience: int | None = None,
    optimizer: str | None = None,
) -> list[Path]:
    """
    Materialise Lightning YAML configs for each target in a NAS trial entry.
    """
    targets = trial_entry.get("targets")
    if not targets:
        raise ValueError("trial_entry must contain a non-empty 'targets' list.")

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    exported: list[Path] = []
    trial_id = trial_entry.get("trial_number", "best")

    for target in targets:
        target_col = target["target_col"]
        model_block = _model_block_from_summary(target)
        cfg = _build_training_config(
            model_block,
            target_col,
            dataset_path,
            artifact_root=output_dir / target_col,
        )
        if max_epochs is not None:
            cfg["training"]["max_epochs"] = int(max_epochs)
        if lr is not None:
            cfg["training"]["lr"] = float(lr)
        if weight_decay is not None:
            cfg["training"]["weight_decay"] = float(weight_decay)
        if early_stopping_patience is not None:
            cfg["training"]["early_stopping_patience"] = int(early_stopping_patience)
        if lr_patience is not None:
            cfg["training"]["lr_patience"] = int(lr_patience)
        if optimizer is not None:
            cfg["training"]["optimizer"] = optimizer

        cfg_name = f"{target_col}_arch{target['arch_id']}_trial{trial_id}.yaml"
        cfg_path = output_dir / cfg_name
        with cfg_path.open("w", encoding="utf-8") as fh:
            yaml.safe_dump(cfg, fh, sort_keys=False)
        exported.append(cfg_path)

    return exported


def _load_trials(trials_path: Path) -> list[dict[str, Any]]:
    """Load JSONL NAS output into a list of dicts."""
    records: list[dict[str, Any]] = []
    with trials_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if not records:
        raise RuntimeError(f"No trials found inside {trials_path}.")
    return records


def _pick_best_trial(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Return the trial dict with the lowest RMSE."""
    return min(records, key=_average_rmse)


def _average_rmse(record: dict[str, Any]) -> float:
    """Extract the averaged RMSE from a NAS record."""
    rmse_entries = record.get("rmse", [])
    if not rmse_entries:
        raise ValueError("NAS record is missing RMSE entries.")
    average_entry = rmse_entries[0]
    return average_entry.get("value", float("inf"))


def _resolve_trials_path(path_arg: str | Path | None) -> Path:
    """Resolve ``--trials-file``; default to the newest file in dnn-output."""
    if path_arg:
        trials_path = Path(path_arg).expanduser().resolve()
        if not trials_path.exists():
            raise FileNotFoundError(f"Specified trials file {trials_path} is missing.")
        return trials_path

    candidates = sorted(TRIALS_DIR.glob("nas_trials_*.jsonl"))
    if not candidates:
        raise FileNotFoundError(
            f"No nas_trials_*.jsonl files found under {TRIALS_DIR}. "
            "Provide --trials-file explicitly."
        )
    return candidates[-1]


def _metadata_from_config(config_path: Path) -> tuple[str, int, Path]:
    """Extract target column, arch ID, and configured model directory."""
    with config_path.open("r", encoding="utf-8") as fh:
        cfg = yaml.safe_load(fh)
    target_col = cfg["data"]["target_col"]
    arch_id = cfg["model"]["arch_id"]
    model_dir_raw = cfg.get("artifacts", {}).get("model_dir")
    if not model_dir_raw:
        raise ValueError(f"Config {config_path} is missing artifacts.model_dir.")
    model_dir = Path(model_dir_raw)
    return target_col, arch_id, model_dir


def _run_training_command(config_path: Path) -> Path:
    """Invoke the Lightning trainer for one YAML config and return .pt path."""
    cmd = [
        sys.executable,
        str(TRAIN_SCRIPT),
        "--config",
        str(config_path),
    ]
    print(f"[train] running {config_path}", flush=True)
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        print(
            f"Training command {exc.cmd} failed with code {exc.returncode}",
            file=sys.stderr,
        )
        raise

    target_col, arch_id, model_dir = _metadata_from_config(config_path)
    model_path = model_dir / f"{target_col}_arch{arch_id}.pt"
    print(f"[train] saved {model_path}", flush=True)
    return model_path


def train_from_configs(config_paths: Sequence[str | Path]) -> list[Path]:
    """
    Train one model per YAML config, returning the expected .pt checkpoints.
    """
    if not config_paths:
        raise ValueError("config_paths must contain at least one YAML file.")

    trained_models: list[Path] = []
    for config in config_paths:
        config_path = Path(config)
        trained_models.append(_run_training_command(config_path))
    return trained_models


def train_cnn_arch(config_paths: Sequence[str | Path]) -> list[Path]:
    """
    Convenience wrapper mirroring the original CLI.
    """
    return train_from_configs(config_paths)


def orchestrate_best_trial(
    trials_path: str | Path | None = None,
    dataset_path: str = DEFAULT_DATASET,
    output_dir: str | Path | None = None,
    max_epochs: int | None = None,
    lr: float | None = None,
    weight_decay: float | None = None,
    early_stopping_patience: int | None = None,
    lr_patience: int | None = None,
    optimizer: str | None = None,
) -> tuple[dict[str, Any], list[Path]]:
    """
    High-level helper to export configs for the top NAS trial.
    """
    resolved_trials = _resolve_trials_path(trials_path)
    records = _load_trials(resolved_trials)
    best_trial = _pick_best_trial(records)

    if output_dir is None:
        export_dir = DEFAULT_CONFIG_OUTPUT_ROOT / resolved_trials.stem
    else:
        export_dir = Path(output_dir).expanduser().resolve()

    generated_configs = export_trial_configs(
        best_trial,
        export_dir,
        dataset_path=dataset_path,
        max_epochs=max_epochs,
        lr=lr,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        lr_patience=lr_patience,
        optimizer=optimizer,
    )
    return best_trial, generated_configs


def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the best NAS trial end-to-end.")
    parser.add_argument(
        "--trials-file",
        type=str,
        help="Path to nas_trials_*.jsonl. Defaults to the newest file under nas/dnn-output.",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=DEFAULT_DATASET,
        help=f"Dataset path for training (default: {DEFAULT_DATASET}).",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        help="Directory to place generated YAML configs. "
        "Defaults to nas/dnn-output/best_configs/<trials stem>/",
    )
    parser.add_argument(
        "--max-epochs",
        type=int,
        help="Override training.max_epochs in the exported configs.",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        help="Override training.early_stopping_patience in the exported configs.",
    )
    parser.add_argument(
        "--train",
        action="store_true",
        help="Run training immediately after generating the configs.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    best_trial, config_paths = orchestrate_best_trial(
        trials_path=args.trials_file,
        dataset_path=args.dataset,
        output_dir=args.output_dir,
        max_epochs=args.max_epochs,
        early_stopping_patience=args.early_stopping_patience,
    )

    trained_models: list[Path] = []
    if args.train:
        trained_models = train_from_configs(config_paths)

    best_trial_rmse = _average_rmse(best_trial)
    print(
        f"[best] trial #{best_trial['trial_number']} "
        f"rmse={best_trial_rmse:.4f}"
    )
    for cfg in config_paths:
        print(f"[config] {cfg}")
    if args.max_epochs:
        print(f"[config] max_epochs={args.max_epochs}")
    if args.early_stopping_patience:
        print(f"[config] early_stopping_patience={args.early_stopping_patience}")
    if args.train:
        for model_path in trained_models:
            print(f"[checkpoint] {model_path}")
    else:
        print("[info] re-run with --train to start training.")


if __name__ == "__main__":
    main()
