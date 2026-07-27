#!/bin/bash
set -euo pipefail

# Submit the same track selection as safety-nas/run/run-safety-nas.sh.
# sbatch safety-nas/run-hpc/austin.sh
sbatch safety-nas/run-hpc/sepang.sh
sbatch safety-nas/run-hpc/melbourne.sh
sbatch safety-nas/run-hpc/brands-hatch.sh
# sbatch safety-nas/run-hpc/budapest.sh
# sbatch safety-nas/run-hpc/catalunya.sh
# sbatch safety-nas/run-hpc/sakhir.sh
# sbatch safety-nas/run-hpc/sao-paulo.sh
# sbatch safety-nas/run-hpc/yas-marina.sh
# sbatch safety-nas/run-hpc/zandvoort.sh

# Additional historical training maps.
sbatch safety-nas/run-hpc/additional-maps/hockenheim.sh
sbatch safety-nas/run-hpc/additional-maps/ims.sh
sbatch safety-nas/run-hpc/additional-maps/montreal.sh
sbatch safety-nas/run-hpc/additional-maps/moscow-raceway.sh
sbatch safety-nas/run-hpc/additional-maps/oschersleben.sh
sbatch safety-nas/run-hpc/additional-maps/spielberg.sh
