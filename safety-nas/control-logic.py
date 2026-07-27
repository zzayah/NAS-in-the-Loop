import argparse

import optuna

import cnn
from cnn import EvaluationTrack

SEED = 0
N_TRIALS = 120
DATASET_PATH = "safety-nas/datasets/combined_all.npz"


def main(
    track: str,
    output_dir: str,
    session_id: str,
) -> None:
    """Run the Safety-NAS Optuna search."""
    cnn.configure_run(output_dir, session_id)
    track_names = [EvaluationTrack[track.strip().upper()]]

    # sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="minimize")
    study.optimize(
        lambda t: cnn.objective(
            t,
            track_names=track_names,
            dataset_pth=DATASET_PATH,
        ),
        n_trials=N_TRIALS,
    )

# This CLI stuff is specifically for the run-safety-nas.sl call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--track", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--session-id", required=True)
    args = parser.parse_args()
    main(
        track=args.track,
        output_dir=args.output_dir,
        session_id=args.session_id,
    )
