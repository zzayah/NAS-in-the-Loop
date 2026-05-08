#!/bin/bash

set -euo pipefail

# Submit the remaining train-track control-logic SLURM jobs in parallel.
sbatch nas/control-logic.sl --track BRANDS_HATCH &
sbatch nas/control-logic.sl --track BUDAPEST &
sbatch nas/control-logic.sl --track CATALUNYA &
sbatch nas/control-logic.sl --track HOCKENHEIM &
sbatch nas/control-logic.sl --track IMS &
sbatch nas/control-logic.sl --track MONTREAL &
sbatch nas/control-logic.sl --track MOSCOW_RACEWAY &
sbatch nas/control-logic.sl --track OSCHERSLEBEN &
sbatch nas/control-logic.sl --track SAKHIR &
sbatch nas/control-logic.sl --track SAO_PAULO &
sbatch nas/control-logic.sl --track SPIELBERG &
sbatch nas/control-logic.sl --track YAS_MARINA &
sbatch nas/control-logic.sl --track ZANDVOORT &

wait
