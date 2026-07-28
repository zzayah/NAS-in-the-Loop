#!/bin/bash
set -euo pipefail

bash safety-nas/run/yas-marina.sh
bash safety-nas/run/zandvoort.sh

bash safety-nas/run/hockenheim.sh
bash safety-nas/run/ims.sh
bash safety-nas/run/montreal.sh
bash safety-nas/run/moscow-raceway.sh
bash safety-nas/run/oschersleben.sh
bash safety-nas/run/spielberg.sh
