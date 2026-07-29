#!/bin/bash
set -euo pipefail
source .venv/bin/activate

SEED="$(python accuracy-nas/control-logic.py --print-seed)"

RUN_ID="$(python -c 'import secrets, string; print("".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8)))')"
RUN_ROOT="data/accuracy-nas/accuracy-nas-seed-${SEED}"

python accuracy-nas/split-dataset.py \
    --output-dir "$RUN_ROOT/datasets/$RUN_ID" \
    --seed "$SEED"

python accuracy-nas/control-logic.py \
    --train-path "$RUN_ROOT/datasets/$RUN_ID/train.npz" \
    --test-path "$RUN_ROOT/datasets/$RUN_ID/test.npz" \
    --output-dir "$RUN_ROOT/nas/$RUN_ID" \
    --session-id "$RUN_ID"

python accuracy-nas/test-best.py \
    --left-trials "$RUN_ROOT/nas/$RUN_ID/left_wall_dist.jsonl" \
    --track-trials "$RUN_ROOT/nas/$RUN_ID/track_width.jsonl" \
    --heading-trials "$RUN_ROOT/nas/$RUN_ID/heading_error.jsonl" \
    --output-dir "$RUN_ROOT/test-best" \
    --run-id "$RUN_ID" \
    --seed "$SEED"

LEFT=("$RUN_ROOT"/test-best/"$RUN_ID"/left_wall_dist_arch*_trial*.pt)
TRACK=("$RUN_ROOT"/test-best/"$RUN_ID"/track_width_arch*_trial*.pt)
HEADING=("$RUN_ROOT"/test-best/"$RUN_ID"/heading_error_arch*_trial*.pt)

python accuracy-nas/compare-track.py \
    --left-model "${LEFT[0]}" \
    --track-model "${TRACK[0]}" \
    --heading-model "${HEADING[0]}" \
    --output-dir "$RUN_ROOT/compare-map" \
    --seed "$SEED"
