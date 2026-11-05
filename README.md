# Predictive maintenance on NASA C-MAPSS

Course project for *Data Analytics Laboratory* (M.Tech, solo, 45 days). Remaining-useful-life (RUL) regression and "fails within 30 cycles" classification on the four NASA C-MAPSS turbofan subsets (FD001–FD004), with a FastAPI + MLflow + Streamlit deployment. **All models are trained from scratch; no pretrained weights are used.**

> Status: **stage 1 of 7 (repo setup) complete.** Modelling stages follow; see `LOG.md` for every decision and each stage's results, and `CODE_GUIDE.md` for how the code works.

## Ground rules
1. Split by **engine**. Every scaler, clusterer and selector is fitted on training engines only; `tests/test_no_leakage.py` enforces this and must stay green.
2. Claims about rankings need **mean ± std over ≥ 5 seeds and several validation splits**; differences inside the spread are called "comparable".
3. RMSE is reported against the **raw** test RUL and against RUL **capped at 125**, with the NASA score against the raw RUL.
4. Sensor/feature importances are **model importances**, never causes of failure.
5. Nothing is invented; failed steps are reported as failed.

## Layout
```
src/cmapss/    lib.py (pipeline + models), audit.py (data audit), expected_sha256.json
tests/         test_no_leakage.py, test_audit.py
results/       generated outputs (data_audit.json, ...)
notebooks/     EDA (stage 2)
deploy/        FastAPI / Streamlit / Docker (stage 7)
scripts/       experiment runners (later stages)
CMAPSSData/    the dataset (git-ignored; see below)
feasibility/   archive of the earlier feasibility spike (git-ignored)
LOG.md  CODE_GUIDE.md  requirements.txt  pytest.ini
```

## Setup
```powershell
python -m venv .venv
.venv\Scripts\python -m pip install -r requirements.txt
```
Data: download "Turbofan Engine Degradation Simulation Data Set" (NASA Prognostics Data Repository; mirror used here: `https://phm-datasets.s3.amazonaws.com/NASA/6.+Turbofan+Engine+Degradation+Simulation+Data+Set.zip`), unzip the inner `CMAPSSData.zip` into `CMAPSSData/` at the repo root (or point `CMAPSS_DATA` at another folder).

On the development laptop the existing environment `feasibility\venv` is used instead of `.venv` (same packages).

## Run
```powershell
$env:PYTHONPATH = "src"
python -m cmapss.audit          # verifies the data; writes results/data_audit.json
python -m pytest                # data-audit tests + the leakage test (40 s to 2.5 min depending on free RAM)
```

## Windows note: no pandas / shap here
Windows Smart App Control on the development laptop blocks pandas' unsigned extensions (and `shap`, which imports pandas). The code is therefore **numpy-only**, and tree-model importances use **LightGBM's native TreeSHAP** (`predict(X, pred_contrib=True)`). The setting is deliberately not bypassed.

## Stages
1. Repo setup + data audit  ✅
2. EDA (sensor trends, constant sensors, per-regime normalisation plots)
3. Regression: Ridge, LightGBM, GRU, TCN, 5 seeds, small hyper-parameter search, RUL cap 100/125/150
4. Classification: 7 models, 5 seeds, + 3-class health stage
5. Analysis: TreeSHAP + permutation importance stability, early-warning lead time, cross-subset tests
6. Deployment: FastAPI, MLflow, Streamlit replay + drift check, Dockerfile (built only if Docker is installed)
7. Report assets: tables, figures, limitations, Markdown draft

Docker is **not installed** on the development laptop at the time of stage 1, so no container has been built or run.
