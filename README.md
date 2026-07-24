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
@inproceedings{
  author    = {Zayah Cortright, Prateek Ganguli, Tingan Zhu, and Samarjit Chakraborty},
  title     = {NAS-in-the-Loop: Trajectory-driven Neural Architecture Search for Safe Autonomous CPS},
  booktitle = {},
  year      = {2026},
  publisher = {},
  doi       = {}
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

Now that the environment is set up, we may run an Optuna architecture search. The following command uses the CLI in `control-logic.py` to run a safety-guided NAS of 5 trials whose neural network is tested the F1TENTH track "Melbourne."

```bash
python safety-nas/control-logic.py --track MELBOURNE --n-trials 5
```

To run the complete set of NAS using SLURM, run `bash run-safety-nas.sl`

The search writes JSONL results to:

```text
safety-nas/dnn-output/nas_trials_*.jsonl
```

At this point, the NAS should be complete. The training parameters are now complex, so we no longer work with the CLI.

For staged best-trial retraining, edit the constants near the top of
`safety-nas/test-best.py`, then call `main()` from inside that file.

After running `test-best.py`, there should now be fully trained .pt files ready for final evaluation on the testing tracks. Once again, edit the parameters at the top of `safety-nas/compare-track.py`, ensuring pathing is correct, then call `main()` from inside that file.

## Accuracy-NAS Workflow

With the safety-guided NAS complete, we now must generate data to compare them to. Accuracy-NAS is a similar NAS framework, but uses validation loss as the feedback component to the NAS.

To get started, we must split the `combined_all.npz` dataset into training and testing partitions. By default, this partition is 80/20, respectively, but this can be edited in the training parameters at the top of the file:

```bash
python accuracy-nas/split-dataset.py
```

We may now run the supervised Optuna searches with the following:

```bash
python accuracy-nas/control-logic.py
```

Note: Because we do not get feedback from specific tracks (and instead from the `combined_all.npz` dataset) we only run accuracy-NAS once.

Results are written as per-target JSONL files:

```text
accuracy-nas/dnn-output/standard_trials_<target>_*.jsonl
```

To fully train the best Neural Network (as selected from accuracy-NAS), modify the parameters at the top of `accuracy-nas/test-best.py`, and run that file.

When complete, configure the parameters and run `accuracy-nas/compare-track.py` to put Accuracy-NAS to the test.

## Visualization

When both Safety-NAS and Accuracy-NAS are done running, we may now visualize our results together. Note: the visualization DOES allow a single input if you would like to do visualization of only Safety-NAS or Accuracy-NAS.

To do this, modify the paths at the top of `vis.ipynb` or `final-analysis-vis.ipynb` to point to compare-map metrics.jsonl results (see `data/accuracy-nas/compare-map-tp0/metrics.jsonl` for an example) and run all cells.

## Attribution

Both Neural Architecture Search architectures in this repository were created by Zayah Cortright and built on previous work by Prateek Ganguli and Tingan Zhu. This work was done within and supported by the Design Automation to X Lab, led by Dr. Samarjit Chakraborty, at the University of North Carolina at Chapel Hill Department of Computer Science. All inquiries should be emailed to zayah [at] unc [dot] edu.
