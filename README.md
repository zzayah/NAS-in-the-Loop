# NAS-in-the-Loop

NAS-in-the-Loop is a tool to facilitate experiments on Neural Architecture Search (NAS) guided by simulation in the F1TENTH environment. 

For purpose of comparison, this repository includes a complementary NAS which is guided by validation loss.

Both of these pathways are found in the following locations:
- `safety-nas/`
- `accuracy-nas/`

And both workflows train small 1D CNNs that estimate:

- `left_wall_dist`
- `track_width`
- `heading_error`

## Citation

We kindly ask all users of this repository cite the following:

```bibtex
@inproceedings{cortright2026nas,
  author    = {Cortright, Zayah and Ganguli, Prateek and Zhu, Tingan and Chakraborty, Samarjit},
  title     = {{NAS-in-the-Loop}: Trajectory-Driven Neural Architecture Search for Safe Autonomous {CPS}},
  booktitle = {Proceedings of the 2026 Forum on Specification \& Design Languages (FDL)},
  year      = {2026},
  publisher = {IEEE}
}
```

## Repository Layout

Newly authored content:
- `safety-nas/`: simulator-in-the-loop Optuna search, best-trial config export, evaluation helpers, and comparison scripts.
- `accuracy-nas/`: supervised Optuna search and wrappers for evaluating an accuracy-selected checkpoint triplet.

Previously existing content:
- `packages/f110_gym/`: local Gymnasium F1TENTH simulator package.
- `packages/f110_planning/`: planners, metrics, model utilities, and simulation helpers.
- `packages/f110_scripts/`: data generation, training, RL, and simulator entry scripts.
- `data/maps/`: map images, YAML metadata, and centerline waypoint files.
- `data/models/`: baseline trained checkpoint files.

## Setup

To begin, we suggest using Python 3.12.4. Additionally, the LiDAR datasets this repository uses (*.npz files) require Git LFS when cloning.

The following code creates up a .venv and installs the necessary packages for the F1TENTH simulation, NAS, and visualization.

```bash
# Clone the repository 
git clone <repo-url>
cd NAS-in-the-Loop

# Create & activate python environment
python3 -m venv .venv
source .venv/bin/activate

# Install packages for simulation & optuna for Safety-NAS
python -m pip install -e packages/f110_gym
python -m pip install -e "packages/f110_planning[test]"
python -m pip install -e "packages/f110_scripts[test]"
python -m pip install optuna matplotlib
```

The easiest way to check if these packages were installed correctly is to run:

```bash
python packages/f110_scripts/src/f110_scripts/sim/reactive_planners.py
```

Should there be missing imports in this run, the rest of the repository certainly will not work. Most commonly, deleting the .venv and going back through the .venv instantiation process solves problems.

## Safety-NAS Workflow

Run experiment commands from the repository root. The scripts under
`safety-nas/run/` now execute the complete experiment pipeline for one training
track:

1. Read the seed from `safety-nas/control-logic.py`.
2. Generate an eight-character run ID.
3. Run the Safety-NAS Optuna search.
4. Select and retrain the best trial.
5. Stage the three target checkpoints.
6. Compare the selected checkpoint triplet with architectures 1--6 on the six
   evaluation maps.

For example, this runs the complete Austin experiment:

```bash
bash safety-nas/run/austin.sh
```

One script is available for each supported training track:

```text
austin, brands-hatch, budapest, catalunya, hockenheim, ims, melbourne,
montreal, moscow-raceway, oschersleben, sakhir, sao-paulo, sepang,
spielberg, yas-marina, zandvoort
```

The track scripts are intentionally split into two sequential batches for
long-running experiments:

```bash
bash safety-nas/run/run-safety-nas-1.sh
bash safety-nas/run/run-safety-nas-2.sh
```

The batch scripts are independent, so they may be launched separately on two
machines or jobs. Within each batch, tracks run one after another.

Each experiment is stored under a seed- and track-specific directory. A run
with seed `0`, track `austin`, and run ID `<run-id>` has this layout:

```text
data/safety-nas/safety-nas-seed-0/austin/
|-- nas/<run-id>.jsonl
|-- test-best/<run-id>/
|   |-- left_wall_dist_arch*_trial*.{pt,yaml}
|   |-- track_width_arch*_trial*.{pt,yaml}
|   `-- heading_error_arch*_trial*.{pt,yaml}
`-- compare-map/<run-id>/
    |-- metrics.jsonl
    `-- *.npz
```

The `.npz` files contain saved simulation traces for the comparison maps. The
comparison currently evaluates Silverstone, Sochi, Spa, Nuerburgring, Monza,
and Mexico City.

The primary experiment settings are still source constants:

- `SEED`, `N_TRIALS`, and `DATASET_PATH` in `safety-nas/control-logic.py`;
- `TRAINING_PROFILE`, `MODE`, and `SKIP_EVAL` in `safety-nas/test-best.py`; and
- the evaluation-map and simulation defaults in `safety-nas/compare-track.py`.

At present, Safety-NAS uses seed `0`, runs 120 Optuna trials per training track,
and uses training profile `0`. The shell scripts pass all paths and run IDs via
the CLI; no path editing is required for the standard flow.

To resume individual stages manually, use the same commands as a track script:

```bash
python safety-nas/control-logic.py \
  --track AUSTIN \
  --output-dir data/safety-nas/manual/austin/nas \
  --session-id <run-id>

python safety-nas/test-best.py \
  --trials-file data/safety-nas/manual/austin/nas/<run-id>.jsonl \
  --output-dir data/safety-nas/manual/austin/test-best \
  --seed 0

python safety-nas/compare-track.py \
  --checkpoint-triple <left.pt> <track.pt> <heading.pt> \
  --output-dir data/safety-nas/manual/austin/compare-map \
  --training-track AUSTIN \
  --seed 0
```

## Accuracy-NAS Workflow

Accuracy-NAS uses validation RMSE instead of simulator feedback. Its wrapper
runs the complete flow: create a seeded 80/20 dataset split, search all three
targets in parallel, retrain the best model for each target, and compare the
resulting triplet with architectures 1--6.

```bash
bash accuracy-nas/run/run-accuracy-nas.sh
```

At present, the wrapper reads seed `1` from `accuracy-nas/control-logic.py` and
runs 120 trials for each of the three targets. A generated run ID keeps the
split, trial logs, checkpoints, and comparison results associated:

```text
data/accuracy-nas/accuracy-nas-seed-1/
|-- datasets/<run-id>/{train,test}.npz
|-- nas/<run-id>/
|   |-- left_wall_dist.jsonl
|   |-- track_width.jsonl
|   `-- heading_error.jsonl
|-- test-best/<run-id>/
|   |-- left_wall_dist_arch*_trial*.{pt,yaml}
|   |-- track_width_arch*_trial*.{pt,yaml}
|   `-- heading_error_arch*_trial*.{pt,yaml}
`-- compare-map/<run-id>/
    |-- metrics.jsonl
    `-- *.npz
```

The search trains on `train.npz` and uses `test.npz` as its Optuna validation
set. After selection, `accuracy-nas/test-best.py` retrains using the full
`accuracy-nas/datasets/combined_all.npz` dataset. The split ratio is controlled
by `TRAIN_RATIO` in `accuracy-nas/split-dataset.py`; search seed and trial count
are controlled by `SEED` and `N_TRIALS` in `accuracy-nas/control-logic.py`; and
final-training behavior is controlled by `TRAINING_PROFILE` and `SKIP_EVAL` in
`accuracy-nas/test-best.py`.

Accuracy-NAS is normally run once per seed because it is not conditioned on a
training track. Individual stages can also be invoked through their CLIs:

```bash
python accuracy-nas/split-dataset.py --output-dir <split-dir> --seed 1

python accuracy-nas/control-logic.py \
  --train-path <split-dir>/train.npz \
  --test-path <split-dir>/test.npz \
  --output-dir <nas-dir> \
  --session-id <run-id>

python accuracy-nas/test-best.py \
  --left-trials <nas-dir>/left_wall_dist.jsonl \
  --track-trials <nas-dir>/track_width.jsonl \
  --heading-trials <nas-dir>/heading_error.jsonl \
  --output-dir <test-best-dir> \
  --run-id <run-id> \
  --seed 1

python accuracy-nas/compare-track.py \
  --left-model <left.pt> \
  --track-model <track.pt> \
  --heading-model <heading.pt> \
  --output-dir <compare-map-dir> \
  --seed 1
```

## Visualization

After Safety-NAS and/or Accuracy-NAS finishes, use
`visualizations/figures.ipynb` to produce aggregate figures. Point its input
paths at the desired `compare-map/<run-id>/metrics.jsonl` files and run the
notebook cells. The notebook can also visualize only one NAS pathway.

## Reproducibility

NAS runs are reproducible when the random seed, dataset, search and training configuration, software environment, and hardware architecture are held constant. Runs on different hardware architectures (for example, ARM and x86) may produce small numerical differences that cause later TPE suggestions to diverge.

The reported NAS experiments used the following hardware and software environment:

| Component | Configuration |
|---|---|
| Host | `daxgpupc1` |
| Operating system | Ubuntu 26.04 LTS (Linux 7.0.0-22-generic) |
| Architecture | x86-64 |
| CPU | AMD Ryzen 9 7900X, 12 cores / 24 threads |
| System memory | 122 GiB |
| GPU | NVIDIA TITAN RTX |
| GPU memory | 24 GiB (24,576 MiB) |
| GPU power limit | 280 W |
| NVIDIA driver | 595.71.05 |
| Driver-supported CUDA version | 13.2 |
| Python | 3.14.4 |
| PyTorch | 2.13.0+cu130 |
| PyTorch CUDA runtime | 13.0 |
| cuDNN version identifier | 92000 |
| Optuna | 4.9.0 |
| PyTorch deterministic algorithms | Disabled |
| cuDNN deterministic mode | Disabled |
| cuDNN benchmark mode | Disabled |

## Attribution

Both Neural Architecture Search architectures in this repository were created by Zayah Cortright and built on previous work by Prateek Ganguli and Tingan Zhu. This work was done within and supported by the Design Automation to X Lab, led by Dr. Samarjit Chakraborty, at the University of North Carolina at Chapel Hill Department of Computer Science. All inquiries should be emailed to zayah [at] unc [dot] edu.
