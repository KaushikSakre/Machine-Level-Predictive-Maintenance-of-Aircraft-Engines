# Code guide

How the code in this repository works, for someone who has not read it. It covers what exists after **stage 1**; later stages append their own sections. File names are relative to the repo root.

## 1. Big picture

The project predicts the **Remaining Useful Life (RUL)** of turbofan engines from the NASA C-MAPSS data, and classifies whether an engine will fail **within 30 cycles**. Every model is trained from scratch.

The single most important design rule is **no leakage**: anything that is *fitted* (a clusterer, a scaler, a sensor selector) must be fitted on training engines only. Everything else in the code is arranged around that rule, and a test enforces it.

```
CMAPSSData/*.txt  ──load──►  Preprocessor.fit(train engines only)
                                   │ transform
                                   ▼
                   normalised sensors ──windows (30 cycles)──► features
                                   │
                  train / validation (split by ENGINE)  +  official test (last window per engine)
```

## 2. Data format

Each of the four subsets (FD001–FD004) has three files: `train_FDxxx.txt`, `test_FDxxx.txt`, `RUL_FDxxx.txt`.

* Train and test files are whitespace-separated with **26 columns**: engine id, cycle, 3 operating settings, 21 sensors (`s1`…`s21`).
* Train engines run to failure. Test engines stop *before* failure; `RUL_FDxxx.txt` gives the true remaining cycles for each test engine, in order.
* FD001/FD003 have one operating condition; FD002/FD004 have six. FD003/FD004 have two fault modes (HPC and Fan degradation), FD001/FD002 one (HPC). This is from the dataset's own `readme.txt`.

In memory the code uses plain numpy arrays with this column layout (constants in `lib.py`):

| constant | meaning |
|---|---|
| `U = 0` | engine (unit) id |
| `C = 1` | cycle number |
| `OSI = slice(2, 5)` | the 3 operating settings |
| `SI = slice(5, 26)` | the 21 sensors |
| `R = 26` | RUL column, appended to the **train** array only |

> **Why numpy and not pandas?** On the development laptop Windows Smart App Control blocks pandas' unsigned binary extensions, so the code is numpy-only. This is deliberate and the security setting is not bypassed.

## 3. `src/cmapss/lib.py` — the pipeline

### Constants
`W = 30` (window length), `CAP = 125` (RUL cap used for training targets), `H = 30` (the classification horizon), `K` (number of operating-condition clusters per subset: 1 for FD001/FD003, 6 for FD002/FD004). `DATA` is the folder with the `.txt` files: `<repo>/CMAPSSData` by default, or the environment variable `CMAPSS_DATA`.

### `load(fd)`
Reads the three files for a subset and returns `(train, test, rul_test)`. It appends the RUL column to the train array: for each engine, `RUL = last_cycle − cycle`.

### `split_units(units, seed, frac=0.2)`
Shuffles the engine ids with a seeded generator and returns `(train_ids, validation_ids)`. The split is **by engine**, never by row, so no engine contributes windows to both sides.

### `Preprocessor`
Three fitted things in one class, all fitted by `fit()` on the rows it is given (the caller passes training-engine rows only):

1. **Clusterer** — KMeans on the 3 operating settings, with `k` clusters. This recovers the operating regime (the six regimes in FD002/FD004 are well separated).
2. **Per-regime z-score** — for each cluster, the mean and standard deviation of every sensor. `transform()` assigns each row to a cluster and normalises with *that cluster's* statistics. This removes the large regime-driven swings so the degradation trend is visible.
3. **Sensor selector** — keeps a sensor only if its within-regime standard deviation is above `1e-5`. Sensors that are constant within every regime carry no information and would blow up when divided by a near-zero standard deviation.

`params()` returns all fitted values; the leakage test compares them.

### `FeatureScaler`
A plain z-score for the *window features* (see below). Fitted on training windows only.

### `make_windows(x, units, rul=None, last_only=False)`
Slides a 30-cycle window along each engine. Engines shorter than 30 cycles are left-padded by repeating the first row. For every window it returns:

* the raw window `(30, n_sensors)` for the sequence models (GRU, TCN), and
* a flat feature vector of 6 summaries per sensor (`last, mean, std, min, max, slope`) for Ridge, LightGBM and the classifiers,
* the RUL at the window's last cycle, and the engine id.

With `last_only=True` it returns only each engine's final window. This is how the **test set is scored**: one prediction per test engine, from its last 30 cycles, compared with the official RUL.

### `build_dataset(fd, seed, ...)`
The one entry point the experiments use. In order:

1. load the data and split engines into train/validation (`split_units`);
2. fit the `Preprocessor` on the **training-engine rows only**;
3. transform *all* rows (train, validation, test) with it;
4. build windows, split them into train/validation by engine id;
5. fit the `FeatureScaler` on the **training windows only**;
6. return an object with `S_*` (sequences), `X_*` (flat features), `rul_*` (raw RUL), `y_*` (RUL capped at 125), `c_*` (binary label `RUL ≤ 30`) for train/validation/test.

Two arguments exist for tests and later stages: `fit_on="all"` is a **deliberately leaky** mode used only as a negative control; `pre=` accepts an already-fitted preprocessor, used for cross-subset tests (train on one subset, evaluate on another without refitting anything).

### Metrics
* `rmse(a, b)` — root mean squared error.
* `nasa(true, pred)` — the NASA asymmetric score: late predictions (`pred > true`) are penalised more (`exp(d/10) − 1`) than early ones (`exp(−d/13) − 1`); lower is better.
* `reg_metrics(true_raw, pred)` — returns RMSE against the **raw** RUL, RMSE against the RUL **capped at 125**, and the NASA score (against the raw RUL). Predictions are clipped to `[0, 125]`. Both RMSEs are always reported because the cap changes the picture a lot on FD002/FD004.

### Models
* `GRUNet` — one GRU layer (64 units) → linear head on the last time step.
* `TCNNet` — a temporal convolutional network: 4 residual blocks of **causal** dilated 1-D convolutions (dilations 1, 2, 4, 8; 32 channels), then a linear head on the last time step.
* `train_seq(kind, d, seed, ...)` — trains either network from scratch on the scaled target `RUL/125`, keeps the epoch with the best validation RMSE, and returns the metrics plus time per epoch. Different seeds change initialisation and batch order but not the data split.

## 4. `src/cmapss/audit.py` — the data audit

`python -m cmapss.audit` (with `PYTHONPATH=src`) checks that the data is exactly what the project assumes: SHA-256 of all 14 files against `src/cmapss/expected_sha256.json` (hashes of the copy verified byte-identical to the NASA download), engine counts (100/100, 260/259, 100/100, 249/248), raw row counts, 26 columns, no NaN/inf, contiguous cycle numbers for every engine, and one RUL row per test engine. It writes `results/data_audit.json` and exits with status 1 on any failure.

## 5. Tests (`tests/`)

Run with `PYTHONPATH=src python -m pytest` (`pytest.ini` also sets the path).

* **`test_no_leakage.py`** — the safety net, and it must stay green after every change.
  * *Poison test:* multiplies the validation engines and the entire test file by large factors, rebuilds the dataset, and asserts that **every fitted parameter is unchanged** (cluster centres, regime means/stds, selected sensors, feature-scaler statistics) and the training windows are identical. If any fitted object had seen validation/test data, its parameters would move. The cluster centres are compared with an absolute tolerance of `1e-9` because KMeans leaves ~1e-21 floating-point noise; real leakage moves them by orders of magnitude more.
  * *Split test:* the train/validation engine sets are disjoint, together cover all engines, and the preprocessor was fitted on exactly the training-engine rows.
  * *Negative control:* with `fit_on="all"` the poison test **must detect** the leak. This proves the test is not vacuous.
* **`test_audit.py`** — the audit passes on the real data and *fails* on a deliberately corrupted copy.

## 6. Directory layout

| path | purpose |
|---|---|
| `src/cmapss/` | library code (`lib.py`, `audit.py`, `expected_sha256.json`) |
| `tests/` | pytest suites |
| `results/` | generated results (JSON, logs, tables); stage outputs land here |
| `notebooks/` | EDA notebooks (stage 2) |
| `deploy/` | FastAPI, Streamlit, Dockerfile (stage 7) |
| `scripts/` | experiment runners (later stages) |
| `CMAPSSData/` | the data (git-ignored) |
| `feasibility/` | archive of the earlier feasibility spike, with its own venv (git-ignored) |
| `LOG.md` | every decision and each stage's results |

## 7. Conventions you will see in later stages

* Report **mean ± standard deviation over ≥ 5 seeds and several validation splits**; call models "comparable" when the gap is within the spread.
* Report RMSE against raw and capped RUL, plus the NASA score against raw RUL.
* Feature importance is a property of the *fitted model*. It is never presented as the cause of failure.
