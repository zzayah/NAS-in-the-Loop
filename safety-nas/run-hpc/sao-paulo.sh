#!/bin/bash
#SBATCH --job-name=f1_snas_sao_paulo
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=12:00:00
#SBATCH --output=safety-nas/run-hpc/slurm_logs/%x_%j.out
#SBATCH --error=safety-nas/run-hpc/slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zayah@unc.edu

set -euo pipefail
cd ~/f1tenth_ng_zc || exit 1
export PYTHONPATH="$HOME/f1tenth_ng_zc:${PYTHONPATH:-}"
if command -v module &>/dev/null; then
    module purge
    module load python/3.12.4
    module load cuda/12.4
fi
source .venv/bin/activate

RUN_ID="seed0_sao_paulo_hpc_$(date +%Y%m%dT%H%M%S)"
RUN_DIR="data/safety-nas/compare-map-rerun-tp0/rerun-s-nas/$RUN_ID"

python safety-nas/control-logic.py --track SAO_PAULO --output-dir "$RUN_DIR/nas" --session-id "$RUN_ID"
python safety-nas/test-best.py --trials-file "$RUN_DIR/nas/nas_trials_$RUN_ID.jsonl" --output-dir "$RUN_DIR/test-best"

LEFT=("$RUN_DIR"/test-best/"$RUN_ID"/left_wall_dist_arch*_trial*.pt)
TRACK=("$RUN_DIR"/test-best/"$RUN_ID"/track_width_arch*_trial*.pt)
HEADING=("$RUN_DIR"/test-best/"$RUN_ID"/heading_error_arch*_trial*.pt)

python safety-nas/compare-track.py --checkpoint-triple "${LEFT[0]}" "${TRACK[0]}" "${HEADING[0]}" --output-dir "$RUN_DIR/compare" --training-track SAO_PAULO
