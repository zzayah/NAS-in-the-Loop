#!/usr/bin/env python3
"""entry point for supervised lidar architecture search."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor

import optuna

import cnn

TARGETS = ("left_wall_dist", "track_width", "heading_error")
SEED = 0
N_TRIALS = 120


def _run_search(target: str, train_path: str, test_path: str, n_trials: int) -> None:
    """Run the Optuna search for one target."""
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: cnn.objective(
            trial,
            target_col=target,
            train_path=train_path,
            validation_path=test_path,
        ),
        n_trials=n_trials,
    )


def main(
    train_path: str,
    test_path: str,
    output_dir: str,
    session_id: str,
) -> None:
    """Run the target searches in parallel."""
    cnn.configure_run(output_dir, session_id)
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as executor:
        futures = [
            executor.submit(_run_search, target, train_path, test_path, N_TRIALS)
            for target in TARGETS
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-path", required=True)
    parser.add_argument("--test-path", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    main(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        session_id=args.session_id,
    )
