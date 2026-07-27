#!/bin/bash
set -euo pipefail
source .venv/bin/activate

RUN_ID="seed0_spielberg_local_$(date +%Y%m%dT%H%M%S)"
RUN_DIR="data/safety-nas/compare-map-rerun-tp0/rerun-s-nas/$RUN_ID"

python safety-nas/control-logic.py --track SPIELBERG --output-dir "$RUN_DIR/nas" --session-id "$RUN_ID"
python safety-nas/test-best.py --trials-file "$RUN_DIR/nas/nas_trials_$RUN_ID.jsonl" --output-dir "$RUN_DIR/test-best"

LEFT=("$RUN_DIR"/test-best/"$RUN_ID"/left_wall_dist_arch*_trial*.pt)
TRACK=("$RUN_DIR"/test-best/"$RUN_ID"/track_width_arch*_trial*.pt)
HEADING=("$RUN_DIR"/test-best/"$RUN_ID"/heading_error_arch*_trial*.pt)

python safety-nas/compare-track.py --checkpoint-triple "${LEFT[0]}" "${TRACK[0]}" "${HEADING[0]}" --output-dir "$RUN_DIR/compare" --training-track SPIELBERG
