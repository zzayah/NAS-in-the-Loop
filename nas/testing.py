import json
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from typing import Sequence, Tuple

TrackConfig = Tuple[str, str]
TRACK_METRIC_KEYS: tuple[tuple[str, str], ...] = (
    ("Cross-Track Error", "crosstrack_rmse_m"),
    ("Cross-Track Error", "crosstrack_std_m"),
    ("Cross-Track Error", "crosstrack_max_m"),
    ("Heading Error", "heading_error_rmse_deg"),
    ("Lap Time", "collision"),
    ("Lap Time", "laps_completed"),
    ("Speed", "speed_mean_m_s"),
)


def _extract_track_metrics(summary: dict[str, dict[str, float]]) -> dict[str, float]:
    """Collect the paper-facing metrics from a simulator summary."""
    metrics: dict[str, float] = {}
    for section, key in TRACK_METRIC_KEYS:
        metrics[key] = float(summary[section][key])
    return metrics


def _run_single_track(
    map_filepath: str,
    waypoints_filepath: str,
    left_wall_dist_filepath: str,
    track_width_filepath: str,
    heading_error_filepath: str,
) -> dict[str, float]:
    """Execute the simulator for a single track and return tracked metrics."""
    cmd = [
        sys.executable,
        "packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py",
        "--planner",
        "dnn",
        "--map",
        map_filepath,
        "--waypoints",
        waypoints_filepath,
        "--render-mode",
        "None",
        "--max-laps",
        "2",
        "--left-wall-model",
        left_wall_dist_filepath,
        "--track-width-model",
        track_width_filepath,
        "--heading-model",
        heading_error_filepath,
    ]

    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)

    match = re.search(r"(\{.*?\})\s*---", proc.stdout, re.DOTALL)
    if not match:
        match = re.search(r"(\{.*\})", proc.stdout, re.DOTALL)

    if not match:
        print("Could not find JSON in output:")
        print(proc.stdout)
        sys.exit(1)

    try:
        summary = json.loads(match.group(1))
    except json.JSONDecodeError as exc:
        print(f"JSON parse error: {exc}")
        print("Raw output:")
        print(proc.stdout)
        sys.exit(1)

    metrics = _extract_track_metrics(summary)
    print(f"[{map_filepath}] Cross-Track RMSE: {metrics['crosstrack_rmse_m']:.4f} m")
    return metrics


def test_cnn_arch(
    left_wall_dist_filepath: str,
    track_width_filepath: str,
    heading_error_filepath: str,
    track_configs: Sequence[TrackConfig],
) -> tuple[float, list[float], dict[str, float], list[dict[str, float]]]:
    """
    Run the CNN checkpoints against multiple tracks and return aggregate metrics.
    """
    if not track_configs:
        raise ValueError("track_configs must include at least one map/waypoints pair")

    track_metrics: list[dict[str, float] | None] = [None for _ in track_configs]
    with ThreadPoolExecutor(max_workers=len(track_configs)) as executor:
        futures: list[tuple[int, any]] = []
        for idx, (map_filepath, waypoints_filepath) in enumerate(track_configs):
            futures.append(
                (
                    idx,
                    executor.submit(
                        _run_single_track,
                        map_filepath,
                        waypoints_filepath,
                        left_wall_dist_filepath,
                        track_width_filepath,
                        heading_error_filepath,
                    ),
                )
            )

        for idx, future in futures:
            track_metrics[idx] = future.result()

    metrics_by_track = [metrics for metrics in track_metrics if metrics is not None]
    rmses = [metrics["crosstrack_rmse_m"] for metrics in metrics_by_track]
    average_rmse = sum(rmses) / len(rmses)
    average_metrics = {
        key: sum(metrics[key] for metrics in metrics_by_track) / len(metrics_by_track)
        for _, key in TRACK_METRIC_KEYS
    }
    print(f"Average Cross-Track RMSE ({len(track_configs)} tracks): {average_rmse:.4f} m")
    return average_rmse, rmses, average_metrics, metrics_by_track


if __name__ == "__main__":
    test_cnn_arch(
        "data/models/left_wall_dist_arch8.pt",
        "data/models/track_width_arch8.pt",
        "data/models/heading_error_arch8.pt",
        [
            (
                "data/maps/F1/Sepang/Sepang_map",
                "data/maps/F1/Sepang/Sepang_centerline.tsv",
            ),
            (
                "data/maps/F1/YasMarina/YasMarina_map",
                "data/maps/F1/YasMarina/YasMarina_centerline.tsv",
            ),
            (
                "data/maps/F1/Austin/Austin_map",
                "data/maps/F1/Austin/Austin_centerline.tsv",
            ),
            (
                "data/maps/F1/Sakhir/Sakhir_map",
                "data/maps/F1/Sakhir/Sakhir_centerline.tsv",
            ),
            (
                "data/maps/F1/Melbourne/Melbourne_map",
                "data/maps/F1/Melbourne/Melbourne_centerline.tsv",
            ),
        ],
    )
