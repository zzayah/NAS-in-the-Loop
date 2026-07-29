import argparse
import sys

SEED = 0

if __name__ == "__main__" and "--print-seed" in sys.argv:
    print(SEED)
    raise SystemExit(0)

import optuna

import cnn
from cnn import EvaluationTrack

N_TRIALS = 120
DATASET_PATH = "safety-nas/datasets/combined_all.npz"

def main(
    track: str,
    output_dir: str,
    session_id: str,
    seed: int = SEED,
) -> None:
    """Run the Safety-NAS Optuna search."""
    cnn.configure_run(output_dir, session_id)
    track_names = [EvaluationTrack[track.strip().upper()]]

    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="minimize", sampler=sampler)

    study.optimize(
        lambda t: cnn.objective(
            t,
            track_names=track_names,
            dataset_pth=DATASET_PATH,
            seed=seed,
        ),
        n_trials=N_TRIALS,
    )

# This CLI stuff is specifically for the run-safety-nas.sl call
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--print-seed", action="store_true")
    parser.add_argument("--track")
    parser.add_argument("--output-dir")
    parser.add_argument("--session-id")
    args = parser.parse_args()
    if not all((args.track, args.output_dir, args.session_id)):
        parser.error("--track, --output-dir, and --session-id are required")
    main(
        track=args.track,
        output_dir=args.output_dir,
        session_id=args.session_id,
        seed=SEED,
    )
