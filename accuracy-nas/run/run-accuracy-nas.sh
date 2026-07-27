#!/bin/bash
set -euo pipefail
source .venv/bin/activate

RUN_ID="seed0_accuracy_local_$(date +%Y%m%dT%H%M%S)"
RUN_DIR="data/accuracy-nas/compare-map-rerun-tp0/rerun-a-nas/$RUN_ID"

python accuracy-nas/split-dataset.py --output-dir "$RUN_DIR/datasets"

python accuracy-nas/control-logic.py \
    --train-path "$RUN_DIR/datasets/train.npz" \
    --test-path "$RUN_DIR/datasets/test.npz" \
    --output-dir "$RUN_DIR/nas" \
    --session-id "$RUN_ID"

python accuracy-nas/test-best.py \
    --left-trials "$RUN_DIR/nas/standard_trials_left_wall_dist_$RUN_ID.jsonl" \
    --track-trials "$RUN_DIR/nas/standard_trials_track_width_$RUN_ID.jsonl" \
    --heading-trials "$RUN_DIR/nas/standard_trials_heading_error_$RUN_ID.jsonl" \
    --output-dir "$RUN_DIR/test-best"

LEFT=("$RUN_DIR"/test-best/"$RUN_ID"/left_wall_dist_arch*_trial*.pt)
TRACK=("$RUN_DIR"/test-best/"$RUN_ID"/track_width_arch*_trial*.pt)
HEADING=("$RUN_DIR"/test-best/"$RUN_ID"/heading_error_arch*_trial*.pt)

python accuracy-nas/compare-track.py \
    --left-model "${LEFT[0]}" \
    --track-model "${TRACK[0]}" \
    --heading-model "${HEADING[0]}" \
    --output-dir "$RUN_DIR/compare"
