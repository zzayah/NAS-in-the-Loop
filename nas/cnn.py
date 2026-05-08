import json
import os
import subprocess
import sys
import tempfile
import uuid
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Sequence

import optuna
import yaml
from torch import nn

from f110_planning.utils.nn_models import get_architecture
from testing import test_cnn_arch

# sending output to ./dnn-output
BASE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = BASE_DIR / "dnn-output"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
SESSION_ID = os.getenv("F1_SESSION_ID") or f"{datetime.utcnow():%Y%m%dT%H%M%S}_{os.getpid()}_{uuid.uuid4().hex[:6]}"
LOG_PATH = OUTPUT_DIR / f"nas_trials_{SESSION_ID}.jsonl"
DEFAULT_TARGET_COLS = ("left_wall_dist", "track_width", "heading_error")
LATEST_MODEL_PATHS = {target: None for target in DEFAULT_TARGET_COLS}


class EvaluationTrack(Enum):
    SEPANG = (
        "data/maps/F1/Sepang/Sepang_map",
        "data/maps/F1/Sepang/Sepang_centerline.tsv",
    )
    YAS_MARINA = (
        "data/maps/F1/YasMarina/YasMarina_map",
        "data/maps/F1/YasMarina/YasMarina_centerline.tsv",
    )
    AUSTIN = (
        "data/maps/F1/Austin/Austin_map",
        "data/maps/F1/Austin/Austin_centerline.tsv",
    )
    SAKHIR = (
        "data/maps/F1/Sakhir/Sakhir_map",
        "data/maps/F1/Sakhir/Sakhir_centerline.tsv",
    )
    MELBOURNE = (
        "data/maps/F1/Melbourne/Melbourne_map",
        "data/maps/F1/Melbourne/Melbourne_centerline.tsv",
    )
    CATALUNYA = (
        "data/maps/F1/Catalunya/Catalunya_map",
        "data/maps/F1/Catalunya/Catalunya_centerline.tsv",
    )
    HOCKENHEIM = (
        "data/maps/F1/Hockenheim/Hockenheim_map",
        "data/maps/F1/Hockenheim/Hockenheim_centerline.tsv",
    )
    BUDAPEST = (
        "data/maps/F1/Budapest/Budapest_map",
        "data/maps/F1/Budapest/Budapest_centerline.tsv",
    )
    MONTREAL = (
        "data/maps/F1/Montreal/Montreal_map",
        "data/maps/F1/Montreal/Montreal_centerline.tsv",
    )
    SPIELBERG = (
        "data/maps/F1/Spielberg/Spielberg_map",
        "data/maps/F1/Spielberg/Spielberg_centerline.tsv",
    )
    ZANDVOORT = (
        "data/maps/F1/Zandvoort/Zandvoort_map",
        "data/maps/F1/Zandvoort/Zandvoort_centerline.tsv",
    )

    @property
    def map_path(self) -> str:
        return self.value[0]

    @property
    def waypoints_path(self) -> str:
        return self.value[1]

# Default evaluation tracks if none are specified.
TRAIN_EVAL_TRACKS = [
    EvaluationTrack.MELBOURNE,
]


def _select_evaluation_tracks(selected: Sequence[object] | None) -> list[EvaluationTrack]:
    """
    Normalise optional user-provided selectors (enum instances or strings) into a track list.
    """
    if not selected:
        return list(TRAIN_EVAL_TRACKS)

    resolved: list[EvaluationTrack] = []
    for entry in selected:
        if isinstance(entry, EvaluationTrack):
            resolved.append(entry)
            continue
        normalized = str(entry).strip().upper()
        try:
            resolved.append(EvaluationTrack[normalized])
        except KeyError as exc:
            valid = ", ".join(track.name for track in EvaluationTrack)
            raise ValueError(
                f"Unknown evaluation track '{entry}'. Valid options: {valid}"
            ) from exc
    return resolved


class DynamicCNN:
    """
    Generates a dynamic architecture specification for f110's 1-D LiDAR CNNs.
    """

    def __init__(self, trial: optuna.trial.Trial, prefix: str = "") -> None:
        def _key(name: str) -> str:
            return f"{prefix}_{name}" if prefix else name

        def _conv1d_output_length(length: int, kernel: int, stride_value: int, pad: int) -> int:
            return max(1, (length + 2 * pad - kernel) // stride_value + 1)

        def _pool1d_output_length(length: int, pool: int) -> int:
            if pool <= 1:
                return length
            return max(1, (length - pool) // pool + 1)

        self.num_layers = trial.suggest_categorical(_key("num_layers"), [1, 2, 3])
        self.activation = trial.suggest_categorical(_key("activation"), ["elu", "relu"])
        first_layer_channels = [1, 4, 8, 16]
        later_layer_channels = [4, 8, 16, 24, 32]
        fc_layer_options = {
            "fc16": [16],
            "fc32": [32],
            "fc64": [64],
            "fc128": [128],
            "fc64_32": [64, 32],
            "fc128_64": [128, 64],
        }
        self.fc_layers = fc_layer_options[
            trial.suggest_categorical(_key("fc_layers"), list(fc_layer_options))
        ]
        self.conv_layers: list[dict[str, int]] = []

        in_channels = 1  # ensures 1×1080 LiDAR input is accepted
        feature_length = 1080
        curr_channels = in_channels
        kernel_size = 3  # intentionally set
        stride = 1  # intentionally set
        padding = 0  # intentionally set

        for idx in range(self.num_layers):
            pool_key = _key(f"pool_size_l{idx}")
            pool_size = trial.suggest_categorical(pool_key, [2, 3, 4])
            if pool_size > feature_length:
                self.num_layers = len(self.conv_layers)
                break
            next_feature_length = _conv1d_output_length(
                feature_length, kernel_size, stride, padding
            )
            next_feature_length = _pool1d_output_length(next_feature_length, pool_size)
            if idx == 0:
                out_channels = trial.suggest_categorical(_key(f"out_channels_l{idx}"), first_layer_channels)
            else:
                out_channels = trial.suggest_categorical(_key(f"out_channels_l{idx}"), later_layer_channels)
                if out_channels < curr_channels:
                    raise optuna.TrialPruned("Conv channel counts must be non-decreasing.")

            self.conv_layers.append(
                {
                    "out_channels": out_channels,
                    "kernel_size": kernel_size,
                    "stride": stride,
                    "padding": padding,
                    "pool_size": pool_size,
                }
            )

            # Simple check to ensure that feature_length is not less than the Kernal size
            feature_length = next_feature_length
            if feature_length < 3:
                self.num_layers = len(self.conv_layers)
                break
            curr_channels = out_channels

        if feature_length * curr_channels < 256:
            raise optuna.TrialPruned("Flattened representation too small.")

        self.model_block = {
            "arch_id": 8,
            "dynamic": {
                "in_channels": 1,
                "input_length": 1080,
                "activation": self.activation,
                "conv_layers": self.conv_layers,
                "fc_layers": self.fc_layers,
            },
        }

    def to_model_block(self) -> dict[str, any]:
        return self.model_block


def _run_training(config: dict[str, any]) -> Path:
    """
    Writes a temporary YAML config, runs the training script, and returns
    the trained model path.
    """
    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as tmp:
        yaml.safe_dump(config, tmp)
        cfg_path = Path(tmp.name)

    try:
        subprocess.run(
            [
                sys.executable,
                "packages/f110_scripts/src/f110_scripts/train/train_nn.py",
                "--config",
                str(cfg_path),
            ],
            check=True,
        )
    finally:
        cfg_path.unlink(missing_ok=True)

    artifacts_cfg = config.get("artifacts", {})
    model_dir = Path(artifacts_cfg.get("model_dir", "data/models"))
    model_name = f"{config['data']['target_col']}_arch{config['model']['arch_id']}"
    return model_dir / f"{model_name}.pt"


def _build_training_config(
    model_block: dict[str, any],
    target_col: str,
    dataset_pth: str | None = None,
    artifact_root: Path | None = None,
) -> dict[str, any]:
    if dataset_pth is None:
        raise ValueError("dataset_path in _build_training_config is not specified.")

    cfg: dict[str, any] = {
        "data": {
            "train_path": dataset_pth,
            "target_col": target_col,
            "batch_size": 128,
            "num_workers": 8,
            "val_split": 0.1,
            "pin_memory": True,
            "prefetch_factor": 2,
        },
        "training": {
            "max_epochs": 30,
            "lr": 5e-5,
            "weight_decay": 1e-4,
            "early_stopping_patience": 15,
            "lr_patience": 20,
            "lr_scheduler_factor": 0.5,
            "optimizer": "adam",
            "scheduler": "reduce_on_plateau",
            "auto_lr_find": False,
            "resume": False,
            "precision": "32",
            "gradient_clip_val": 1.0,
            "profiler": None,
        },
        "model": deepcopy(model_block),
    }
    if artifact_root is not None:
        artifact_root = Path(artifact_root)
        cfg["artifacts"] = {
            "model_dir": str(artifact_root / "models"),
            "checkpoint_dir": str(artifact_root / "checkpoints"),
            "log_dir": str(artifact_root / "lightning_logs"),
        }
    return cfg


def objective(
    trial: optuna.trial.Trial,
    _unused_loader=None,
    n_epoch: int = 10,
    seed: int = 41,
    target_cols: tuple[str, ...] = DEFAULT_TARGET_COLS,
    dataset_pth: str = "nas/datasets/combined_all.npz",
    track_names: Sequence[object] | None = None,
) -> float:
    """
    Run one Optuna trial that samples a single arch8 config, trains the left/track/heading
    nets in parallel, waits for all three checkpoints (.pt) to land, then evaluates the trio
    once via test_cnn_arch and returns that combined RMSE.

    Args:
        track_names: Optional iterable of track names (matching EvaluationTrack enums) or
            EvaluationTrack instances to restrict which maps are evaluated.
    """
    del n_epoch, seed, _unused_loader  # unused

    optimizer = trial.suggest_categorical("optimizer", ["adam", "adamw"])
    model_blocks = {}
    for target in target_cols:
        architecture = DynamicCNN(trial, prefix=target)
        block = architecture.to_model_block()
        block["arch_id"] = 8
        model_blocks[target] = block

    evaluation_tracks = _select_evaluation_tracks(track_names)
    trial_artifact_root = OUTPUT_DIR / "trial_artifacts" / f"{SESSION_ID}_trial{trial.number:05d}"
    cfgs = []
    for target in target_cols:
        target_artifact_root = trial_artifact_root / target
        target_artifact_root.mkdir(parents=True, exist_ok=True)
        cfgs.append(
            _build_training_config(
                model_blocks[target],
                target,
                dataset_pth,
                artifact_root=target_artifact_root,
            )
        )
        cfgs[-1]["training"]["optimizer"] = optimizer

    trained_runs: list[tuple[dict[str, any], Path]] = []
    # TODO add specific partition of resources once the specific resource constraints (running on a gpu/cpu) are known
    with ThreadPoolExecutor(max_workers=len(cfgs)) as executor:
        futures = []
        for cfg in cfgs:
            futures.append(executor.submit(_run_training, cfg))
        for cfg, future in zip(cfgs, futures):
            target = cfg["data"]["target_col"]
            try:
                model_path = future.result()
            except subprocess.CalledProcessError:
                return float("inf")
            trained_runs.append((cfg, model_path))

    for cfg, model_path in trained_runs:
        target = cfg["data"]["target_col"]
        LATEST_MODEL_PATHS[target] = model_path

    track_configs = [(track.map_path, track.waypoints_path) for track in evaluation_tracks]

    try:
        average_rmse, track_rmses, average_metrics, track_metrics = test_cnn_arch(
            left_wall_dist_filepath=str(LATEST_MODEL_PATHS["left_wall_dist"]),
            track_width_filepath=str(LATEST_MODEL_PATHS["track_width"]),
            heading_error_filepath=str(LATEST_MODEL_PATHS["heading_error"]),
            track_configs=track_configs,
        )
    except TypeError as exc:
        raise RuntimeError("Missing trained checkpoints before running test_cnn_arch") from exc
    finally:
        for target in LATEST_MODEL_PATHS:
            LATEST_MODEL_PATHS[target] = None

    _log_trial_result(
        trial=trial,
        trained_runs=trained_runs,
        average_rmse=average_rmse,
        track_rmses=track_rmses,
        average_metrics=average_metrics,
        track_metrics=track_metrics,
        evaluation_tracks=evaluation_tracks,
    )
    return average_rmse


def _log_trial_result(
    trial: optuna.trial.Trial,
    trained_runs: list[tuple[dict[str, any], Path]],
    average_rmse: float,
    track_rmses: list[float],
    average_metrics: dict[str, float],
    track_metrics: list[dict[str, float]],
    evaluation_tracks: Sequence[EvaluationTrack],
) -> None:
    """
    Append a structured summary for each NAS trial to ``output/nas_trials.jsonl``.
    """
    target_summaries: list[dict[str, any]] = []
    for cfg, model_path in trained_runs:
        try:
            model = get_architecture(cfg["model"]["arch_id"], cfg["model"])
            architecture = repr(model)
            layers = _summarize_layers(model)
        except Exception as exc:  # pragma: no cover - logging best effort
            architecture = f"<error rendering architecture: {exc}>"
            layers = []

        target_summaries.append(
            {
                "target_col": cfg["data"]["target_col"],
                "arch_id": cfg["model"]["arch_id"],
                "conv_layers": cfg["model"]["dynamic"]["conv_layers"],
                "fc_layers": cfg["model"]["dynamic"]["fc_layers"],
                "activation": cfg["model"]["dynamic"]["activation"],
                "model_path": str(model_path),
                "architecture": architecture,
                "layers": layers,
            }
        )

    rmse_entries: list[dict[str, any]] = [
        {
            "type": "average",
            "value": average_rmse,
        }
    ]
    for track, rmse in zip(evaluation_tracks, track_rmses):
        rmse_entries.append(
            {
                "track": track.name,
                "map_path": track.map_path,
                "waypoints_path": track.waypoints_path,
                "value": rmse,
            }
        )

    per_track_metrics: list[dict[str, any]] = []
    for track, metrics in zip(evaluation_tracks, track_metrics):
        per_track_metrics.append(
            {
                "track": track.name,
                "map_path": track.map_path,
                "waypoints_path": track.waypoints_path,
                **metrics,
            }
        )

    entry = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "trial_number": trial.number,
        "rmse": rmse_entries,
        "metrics": {
            "average": average_metrics,
            "per_track": per_track_metrics,
        },
        "params": trial.params,
        "targets": target_summaries,
    }

    with LOG_PATH.open("a", encoding="utf-8") as f:
        f.write(json.dumps(entry))
        f.write("\n")


def _summarize_layers(model: nn.Module) -> list[dict[str, any]]:
    """
    Convert a Sequential model into a structured list of layer dictionaries for logging.
    """
    layer_summaries: list[dict[str, any]] = []
    for layer in model:
        info: dict[str, any] = {"type": layer.__class__.__name__}
        if isinstance(layer, nn.Conv1d):
            info.update(
                {
                    "in_channels": layer.in_channels,
                    "out_channels": layer.out_channels,
                    "kernel_size": layer.kernel_size[0],
                    "stride": layer.stride[0],
                    "padding": layer.padding[0],
                }
            )
        elif isinstance(layer, nn.MaxPool1d):
            info.update(
                {
                    "kernel_size": layer.kernel_size,
                    "stride": layer.stride,
                    "padding": layer.padding,
                }
            )
        elif isinstance(layer, nn.Linear):
            info.update(
                {
                    "in_features": layer.in_features,
                    "out_features": layer.out_features,
                    "bias": layer.bias is not None,
                }
            )
        elif isinstance(layer, nn.Flatten):
            info.update(
                {
                    "start_dim": layer.start_dim,
                    "end_dim": layer.end_dim,
                }
            )
        elif isinstance(layer, (nn.ELU, nn.ReLU)):
            if isinstance(layer, nn.ELU):
                info["alpha"] = layer.alpha
        else:
            info["repr"] = repr(layer)
        layer_summaries.append(info)
    return layer_summaries
