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
