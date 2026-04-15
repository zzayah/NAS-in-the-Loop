#!/bin/bash

set -euo pipefail

# Submit the next six train-track control-logic SLURM jobs in parallel.
sbatch nas/control-logic.sl --track CATALUNYA &
sbatch nas/control-logic.sl --track HOCKENHEIM &
sbatch nas/control-logic.sl --track BUDAPEST &
sbatch nas/control-logic.sl --track MONTREAL &
sbatch nas/control-logic.sl --track SPIELBERG &
sbatch nas/control-logic.sl --track ZANDVOORT &

wait
