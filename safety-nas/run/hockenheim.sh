#!/bin/bash
set -euo pipefail
source .venv/bin/activate

SEED="$(python safety-nas/control-logic.py --print-seed)"
TRACK_NAME="$(basename "$0" .sh)"

RUN_ID="$(python -c 'import secrets, string; print("".join(secrets.choice(string.ascii_lowercase + string.digits) for _ in range(8)))')"
RUN_DIR="data/safety-nas/safety-nas-seed-${SEED}/$TRACK_NAME"

python safety-nas/control-logic.py --track HOCKENHEIM --output-dir "$RUN_DIR/nas" --session-id "$RUN_ID"
python safety-nas/test-best.py --trials-file "$RUN_DIR/nas/$RUN_ID.jsonl" --output-dir "$RUN_DIR/test-best" --seed "$SEED"

LEFT=("$RUN_DIR"/test-best/"$RUN_ID"/left_wall_dist_arch*_trial*.pt)
TRACK=("$RUN_DIR"/test-best/"$RUN_ID"/track_width_arch*_trial*.pt)
HEADING=("$RUN_DIR"/test-best/"$RUN_ID"/heading_error_arch*_trial*.pt)

python safety-nas/compare-track.py --checkpoint-triple "${LEFT[0]}" "${TRACK[0]}" "${HEADING[0]}" --output-dir "$RUN_DIR/compare-map" --training-track HOCKENHEIM --seed "$SEED"
