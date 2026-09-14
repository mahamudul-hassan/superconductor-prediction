# Superconductor Critical Temperature: Subgroup-Valid Conformal Prediction

Code and figures accompanying the manuscript:

> Siddique, M.H. "Quantifying Subgroup-Valid Uncertainty in Superconducting
> Critical Temperature Prediction: A Conformal Calibration Case Study."
> Submitted to *Reliability Engineering & System Safety*.

This repository is provided for reproducibility checking. It contains every
script actually run to produce the numbers, tables, and figures in the
manuscript, in the order they were run, plus the final figure files.

## What's here

- **`code/`** — all analysis scripts, numbered in run order, with their own
  [README](code/README.md) explaining which track is the final reported
  pipeline vs. earlier exploratory work kept for transparency.
- **`figures/`** — the 8 figures used in the manuscript, as both vector PDF
  (used in the paper) and PNG (for quick viewing).
- **`requirements.txt`** — pinned package versions used for all reported
  results.

## Reproducing the results

1. Download `train.csv` and `unique_m.csv` from the UCI Machine Learning
   Repository: <https://archive.ics.uci.edu/dataset/464/superconductivty+data>
2. Create and activate a Python environment, then:
   ```bash
   pip install -r requirements.txt
   ```
3. Edit the hard-coded input path at the top of each script (currently
   `/mnt/user-data/uploads/train.csv` / `unique_m.csv`) to point at your local
   copy of the two files above.
4. Create an `artifacts/` directory next to the scripts — every script reads
   and writes its intermediate results there.
5. Run the scripts in `code/` in numeric order. See
   [`code/README.md`](code/README.md) for exactly which scripts constitute
   the final reported pipeline (as opposed to earlier exploratory scripts
   kept for transparency) and what each one produces.

## Data source

This work uses the publicly available UCI Superconductivity Data Set
(Hamidieh, K., 2018, *Computational Materials Science*, 154, 346-354,
doi:10.1016/j.commatsci.2018.07.052). The dataset itself is not
redistributed in this repository — download it directly from the link
above.

## Notes on reproducibility

- All model fitting was run single-threaded (`n_jobs=1`) on a single CPU
  core; wall-clock times reported in the paper (and reproduced by these
  scripts) reflect that environment, not a claim about the algorithms'
  inherent relative speed on other hardware.
- Optuna studies are stored in a local SQLite file (`optuna_study.db`,
  created on first run) so that additional trials can be added across
  multiple runs without losing prior ones.
- A known limitation, stated in the paper: the repeated-split evaluation
  holds the trained base learner fixed and only re-partitions the held-out
  calibration/test pool: it measures calibration-induced variance, not full
  retrain-per-split variance.

## Citation

If you use this code, please cite the manuscript above (citation details to
be updated once a DOI is assigned) and the original dataset (Hamidieh, 2018,
linked above).

## License

No license file is currently included. If you want this code to be reusable
by others under a specific license (e.g., MIT, Apache 2.0), add a `LICENSE`
file before or after pushing -- GitHub can generate one for you from a
template when creating or editing the repository.
