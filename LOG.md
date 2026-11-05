# LOG

Every decision and each stage's results, in order. Dates are 2026.

## Stage 1: repo setup and data audit (2026-11-05)

### Decisions
1. **Repo root is `D:\DA-project`.** `git init` there; `CMAPSSData/` already lives at this level. The earlier feasibility spike stays on disk in `feasibility/` as an archive and is git-ignored (it holds its own venv, the spike reports and scratch scripts). Nothing was moved or deleted from it.
2. **Library code copied, not rewritten:** `feasibility/plan/cmapss_lib.py` -> `src/cmapss/lib.py`, and `feasibility/plan/tests/test_no_leakage.py` -> `tests/test_no_leakage.py`. Only two edits: the data folder now defaults to `<repo>/CMAPSSData` (override with env var `CMAPSS_DATA`), and the test imports `from cmapss import lib as L`. The pipeline logic and the tests are otherwise unchanged.
3. **numpy-only**, as instructed: Windows Smart App Control blocks pandas' unsigned extensions on this laptop (so `shap` cannot import either). Security setting not touched. Tree importances will use LightGBM's native TreeSHAP.
4. **Environment:** the spike venv `feasibility/venv` is reused rather than rebuilt (an install took about 18 min). `requirements.txt` pins the versions actually installed there, so a fresh `.venv` can be created from it; that fresh install has **not** been tested.
5. **Data is not committed** (public dataset, 43 MB). Instead `src/cmapss/expected_sha256.json` stores the SHA-256 of each of the 14 files; these hashes come from the local copy that was verified byte-identical to the NASA S3 mirror download (14/14 files, 0 mismatches, 2026-11-05). `python -m cmapss.audit` checks them.
6. **Data audit** (`src/cmapss/audit.py`) checks hashes, engine counts, raw row counts, 26 columns, finite values, contiguous cycle numbers per engine, and RUL rows = test engines. A test also checks that the audit *fails* on a corrupted copy.
7. Docs: `README.md` (setup, rules, stages) and `CODE_GUIDE.md` (how the code works).
8. `deploy/`, `notebooks/`, `results/` are created empty (`.gitkeep`) and filled in later stages. The spike's FastAPI/Streamlit smoke test remains in `feasibility/deploy/` and will be redone properly in the deployment stage.
9. Commits use the git identity already configured on this machine.

### Problems met
* PowerShell 5.1 `Set-Content -Encoding utf8` wrote a **UTF-8 BOM** into `expected_sha256.json`, which broke JSON parsing (2 audit tests failed with `JSONDecodeError`). Fixed by stripping the BOM from that file and from `lib.py` and `test_no_leakage.py`; no content change.
* I briefly started a background test run before fixing the BOM; it was stopped and rerun from scratch. The log in `results/stage1_pytest.log` is the clean run.

### Results
Data audit (`results/data_audit.json`): 14/14 files match expected SHA-256; every subset PASS.

| subset | train engines | test engines | train rows | test rows | RUL rows |
|---|---|---|---|---|---|
| FD001 | 100 | 100 | 20,631 | 13,096 | 100 |
| FD002 | 260 | 259 | 53,759 | 33,991 | 259 |
| FD003 | 100 | 100 | 24,720 | 16,596 | 100 |
| FD004 | 249 | 248 | 61,249 | 41,214 | 248 |

All engine counts equal the expected values; cycles are contiguous (1..n) for every engine in every train and test file; no NaN/inf.

Tests (`results/stage1_pytest.log`): **7 passed in 39.9 s** (2 audit tests, 5 leakage tests including the negative control that must detect a deliberate leak). The same leakage suite took 140 s in the spike when free RAM was lower, so timing varies with memory pressure.

### Not done / open
* Docker is not installed on this laptop (no `docker`, no WSL), so no container exists yet; the Dockerfile arrives in the deployment stage and will only be built if Docker becomes available.
* A fresh-venv install from `requirements.txt` was not tested.
* No modelling results exist in this repo yet; the spike numbers in `feasibility/REPORT.md` are single-split, single-seed and are not carried over.

## Stage 2: EDA (2026-11-05)

Code: `src/cmapss/eda.py` (run `PYTHONPATH=src python -m cmapss.eda`). Outputs: `results/figures/eda_trends_FD00{1..4}.png`, `eda_normalisation_FD002.png`, `eda_normalisation_FD004.png`, `eda_engine_lengths.png`, `results/eda_summary.json`, `results/eda_sensor_table.md`.

### Decisions
1. EDA uses the **training files only** (no test data, no models) and is purely descriptive. Correlations are Spearman ρ between a sensor and RUL: a trend measure, **not** a causal statement.
2. EDA applies the same per-regime normalisation as `lib.Preprocessor` (KMeans on the 3 operating settings, k = 1/6/1/6) but keeps all 21 sensors so flat ones stay visible. The script asserts that its flat-sensor rule agrees exactly with `lib.Preprocessor`'s selector on all four subsets (it did).
3. **No Jupyter notebook:** Jupyter is not installed and an unexecuted notebook would be misleading. EDA is a script that writes figures and tables; `notebooks/` stays empty for now.
4. `lib.py` is **unchanged**, so all stage-1 and spike results stay comparable.

### Findings (all from the data; see figures and tables)
* **Flat sensors** (constant within every operating regime, dropped by the selector): FD001 s1, s5, s10, s16, s18, s19; FD003 s1, s5, s16, s18, s19; FD002 and FD004 s1, s5, s18, s19. In FD001/FD003 they have exactly 1 unique value. In FD002/FD004 they take 6 unique values (2 for s19): they are constant *within* a regime and differ only *between* regimes, i.e. they carry only regime information.
* **Sensor sets differ across subsets:** s10 is flat in FD001 only; s16 is flat in FD001 and FD003 but kept in FD002/FD004 (where it has just 2 unique values).
* **Low-resolution (near-binary or stepped) sensors are kept by the selector:** FD001 s6 (2 unique values), FD002/FD004 s16 (2), plus stepped s6/s10/s17 in several subsets. **Open question for stage 3:** whether to add a cardinality rule. Not changed now, to keep results comparable.
* **Regimes:** every distinct operating-setting combination maps to exactly one KMeans cluster (checked), so k = 6 recovers the six regimes. Regime sizes are uneven: one regime holds 13,458 rows in FD002 and 15,395 in FD004; the other five hold about 8,000 and 9,100 each. That is a property of the data, not a clustering error.
* **Per-regime normalisation matters a lot in FD002/FD004:** the regime explains about 99% of the variance of each raw sensor (η² 0.99–1.00 for the top sensors). Spearman |ρ| with RUL for the strongest sensors for the four strongest sensors rises from about 0.12–0.18 (raw) to about 0.60–0.73 (normalised); e.g. FD004 s11 0.18 -> 0.73, s4 0.17 -> 0.69. The before/after figures show raw values jumping between six regime levels, and a clean rising trend after normalisation.
* **Strongest normalised trends (|ρ|):** FD001 s11 0.72, s4 0.70, s12 0.69, s7 0.68; FD002 s11 0.71, s4 0.68, s15 0.67; FD003 s11 0.76, s13 0.74, s8 0.74, s4 0.73; FD004 s11 0.73, s4 0.69, s17 0.64. Sensor s11 has the largest |ρ| in all four subsets.
* **FD003 differs from FD001:** s7, s12, s15, s20 and s21 have |ρ| about 0.08–0.09 in FD003 although they trend strongly in FD001 (|ρ| 0.66–0.69 for s7, s12, s15, s21). In the FD003 trend figure their median moves only slightly, while the 10–90% band fans out as RUL approaches 0, so engines move in different directions. FD003 has two fault modes (HPC and Fan) per the dataset readme. That is a *hypothesis* for the difference; it was **not tested** (engines are not labelled by mode).
* **Engine lengths:** train engines run 128–362 cycles (FD001), 128–378 (FD002), 145–525 (FD003), 128–543 (FD004); test engines are truncated (min 19–38, median 132–154 cycles).

### Open items
* Decide in stage 3 whether to drop near-binary sensors (needs the seed and split protocol to say anything about its effect).
* Docker still not installed; GitHub remote added (see below).

## Repository connection (2026-11-05)
`origin` = `https://github.com/KaushikSakre/Machine-Level-Predictive-Maintenance-of-Aircraft-Engines.git`. The remote has no branches (empty). **Nothing has been pushed**; the local branch is `master`. Pushing was left for the owner to confirm.
