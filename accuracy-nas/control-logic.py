#!/usr/bin/env python3
"""entry point for supervised lidar architecture search."""

from __future__ import annotations

import argparse
import sys
from concurrent.futures import ThreadPoolExecutor

SEED = 0

if __name__ == "__main__" and "--print-seed" in sys.argv:
    print(SEED)
    raise SystemExit(0)

import optuna

import cnn

TARGETS = ("left_wall_dist", "track_width", "heading_error")
N_TRIALS = 120


def _run_search(
    target: str, train_path: str, test_path: str, n_trials: int, seed: int
) -> None:
    """Run the Optuna search for one target."""
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)
    study.optimize(
        lambda trial: cnn.objective(
            trial,
            target_col=target,
            train_path=train_path,
            validation_path=test_path,
            seed=seed,
        ),
        n_trials=n_trials,
    )


def main(
    train_path: str,
    test_path: str,
    output_dir: str,
    session_id: str,
    seed: int = SEED,
) -> None:
    """Run the target searches in parallel."""
    cnn.configure_run(output_dir, session_id)
    with ThreadPoolExecutor(max_workers=len(TARGETS)) as executor:
        futures = [
            executor.submit(_run_search, target, train_path, test_path, N_TRIALS, seed)
            for target in TARGETS
        ]
        for future in futures:
            future.result()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-seed", action="store_true")
    parser.add_argument("--train-path")
    parser.add_argument("--test-path")
    parser.add_argument("--output-dir")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    if not all((args.train_path, args.test_path, args.output_dir, args.session_id)):
        parser.error("run paths and --session-id are required")
    main(
        train_path=args.train_path,
        test_path=args.test_path,
        output_dir=args.output_dir,
        session_id=args.session_id,
        seed=SEED,
    )
