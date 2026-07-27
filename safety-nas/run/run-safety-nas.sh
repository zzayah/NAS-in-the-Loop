#!/bin/bash
set -euo pipefail

# bash safety-nas/run/austin.sh
bash safety-nas/run/sepang.sh
bash safety-nas/run/melbourne.sh
bash safety-nas/run/brands-hatch.sh
bash safety-nas/run/budapest.sh
bash safety-nas/run/catalunya.sh
bash safety-nas/run/sakhir.sh
bash safety-nas/run/sao-paulo.sh
bash safety-nas/run/yas-marina.sh
bash safety-nas/run/zandvoort.sh

# Additional historical training maps.
bash safety-nas/run/hockenheim.sh
bash safety-nas/run/ims.sh
bash safety-nas/run/montreal.sh
bash safety-nas/run/moscow-raceway.sh
bash safety-nas/run/oschersleben.sh
bash safety-nas/run/spielberg.sh
