"""per-target supervised Optuna search for lidar CNNs."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
import importlib.util
from datetime import datetime
from pathlib import Path
from typing import Any

import optuna
from f110_planning.utils.nn_models import get_architecture

BASE_DIR = Path(__file__).resolve().parent
REPO_ROOT = BASE_DIR.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

SAFETY_NAS_DIR = REPO_ROOT / "safety-nas"
if str(SAFETY_NAS_DIR) not in sys.path:
    sys.path.insert(0, str(SAFETY_NAS_DIR))

spec = importlib.util.spec_from_file_location("safety_nas_cnn", SAFETY_NAS_DIR / "cnn.py")
if spec is None or spec.loader is None:
    raise ImportError(f"Could not load Safety-NAS CNN module from {SAFETY_NAS_DIR / 'cnn.py'}.")
safety_nas_cnn = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = safety_nas_cnn
spec.loader.exec_module(safety_nas_cnn)

DynamicCNN = safety_nas_cnn.DynamicCNN
_build_training_config = safety_nas_cnn._build_training_config
_run_training = safety_nas_cnn._run_training
_summarize_layers = safety_nas_cnn._summarize_layers

from evaluation import evaluate_model

TARGETS = ("left_wall_dist", "track_width", "heading_error")
OUTPUT_DIR = BASE_DIR / "dnn-output"
SESSION_ID = os.getenv("F1_SESSION_ID") or (
    f"{datetime.utcnow():%Y%m%dT%H%M%S}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
)


def configure_run(output_dir: str | Path, session_id: str) -> None:
    """Set the artifact directory and identifier for one search process."""
    global OUTPUT_DIR, SESSION_ID
    OUTPUT_DIR = Path(output_dir)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    SESSION_ID = session_id


def objective(
    trial: optuna.trial.Trial,
    target_col: str,
    train_path: str = "accuracy-nas/datasets/train.npz",
    validation_path: str = "accuracy-nas/datasets/validation.npz",
) -> float:
    """Train one candidate architecture and return validation RMSE."""
    if target_col not in TARGETS:
        raise ValueError(f"unknown target: {target_col}")

    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    model_block = DynamicCNN(trial).to_model_block()
    # if any(
    #     previous.number != trial.number
    #     and previous.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.RUNNING)
    #     and previous.params == trial.params
    #     for previous in trial.study.get_trials(deepcopy=False)
    # ):
    #     raise optuna.TrialPruned("Duplicate parameters")

    trial_root = OUTPUT_DIR / "trial_artifacts" / f"{SESSION_ID}_{target_col}_trial{trial.number:05d}"
    cfg = _build_training_config(
        model_block,
        target_col,
        train_path,
        artifact_root=trial_root,
    )
    cfg["training"]["optimizer"] = optimizer

    try:
        model_path = _run_training(cfg)
    except subprocess.CalledProcessError:
        return float("inf")

    metrics = evaluate_model(model_path, validation_path, target_col)
    _log_trial(trial, cfg, model_path, metrics, validation_path)
    return float(metrics["rmse"])


def _log_trial(
    trial: optuna.trial.Trial,
    cfg: dict[str, Any],
    model_path: Path,
    metrics: dict[str, float | int],
    validation_path: str,
) -> None:
    """Write one trial record."""
    model = get_architecture(cfg["model"]["arch_id"], cfg["model"])
    target = cfg["data"]["target_col"]
    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "trial_number": trial.number,
        "target_col": target,
        "objective": {
            "name": "validation_rmse",
            "value": metrics["rmse"],
            "dataset": validation_path,
        },
        "validation": metrics,
        "params": trial.params,
        "targets": [
            {
                "target_col": target,
                "arch_id": cfg["model"]["arch_id"],
                "conv_layers": cfg["model"]["dynamic"]["conv_layers"],
                "fc_layers": cfg["model"]["dynamic"]["fc_layers"],
                "activation": cfg["model"]["dynamic"]["activation"],
                "model_path": str(model_path),
                "architecture": repr(model),
                "layers": _summarize_layers(model),
            }
        ],
    }
    path = OUTPUT_DIR / f"standard_trials_{target}_{SESSION_ID}.jsonl"
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(entry))
        handle.write("\n")
