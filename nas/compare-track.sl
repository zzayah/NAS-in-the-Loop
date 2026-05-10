#!/bin/bash
#SBATCH --job-name=f1_compare
#SBATCH --partition=a100-gpu
#SBATCH --qos=gpu_access
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16g
#SBATCH --time=07:00:00
#SBATCH --output=slurm_logs/%x_%j.out
#SBATCH --error=slurm_logs/%x_%j.err
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=zayah@unc.edu

cd ~/f1tenth_ng_zc || exit 1
export PYTHONPATH="$HOME/f1tenth_ng_zc:$PYTHONPATH"
if command -v module &>/dev/null; then
    module purge
    module load python/3.12.4
    module load cuda/12.4
fi
source .venv/bin/activate
python nas/compare-track.py
