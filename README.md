# Scorify

Scorify is a local credit-default scoring project built around a LightGBM classifier and a FastAPI inference service. The model predicts the probability of serious delinquency within two years from the Give Me Some Credit dataset. The reported validation ROC-AUC is 0.866.

## Project Structure

```text
app/                  FastAPI service and request/response schemas
data/raw/             Source training and test CSV files
data/processed/       Prepared feature parquet files and submission
models/               Serialized LightGBM model
notebooks/            Exploratory analysis and pipeline notebooks
src/features/         Training-data feature construction
src/models/           Model training and batch inference scripts
tests/                API and offline/online alignment tests
```

## Local Setup

Use Python 3.11 for the same runtime as the production container.

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

The inference service expects `models/lightgbm_model.pkl` at the project-relative path. Start the API from the project root:

```powershell
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

The health endpoint is `GET http://localhost:8000/health`; interactive API documentation is available at `http://localhost:8000/docs`.

## Prediction API

Send the ten source attributes to `POST /predict`. Both the original hyphenated delinquency column names and the underscore variants used by the tests are accepted.

```json
{
  "RevolvingUtilizationOfUnsecuredLines": 0.05,
  "age": 45,
  "NumberOfTime30_59DaysPastDueNotWorse": 0,
  "DebtRatio": 0.1,
  "MonthlyIncome": 10000,
  "NumberOfOpenCreditLinesAndLoans": 5,
  "NumberOfTimes90DaysLate": 0,
  "NumberRealEstateLoansOrLines": 1,
  "NumberOfTime60_89DaysPastDueNotWorse": 0,
  "NumberOfDependents": 1
}
```

Example response:

```json
{
  "default_probability": 0.0123,
  "decision": "APPROVED"
}
```

The API derives the five engineered model features from the request. A score below 0.30 is `APPROVED`; a score at or above 0.30 is `REJECTED`. Missing monthly income is represented by a 5400.0 fallback and a missingness indicator; missing dependents use 0.0 and their own indicator. These defaults are service fallbacks, while the training feature pipeline imputes income using age-decile medians, so missing-value predictions may not exactly match offline preprocessing. The alignment test deliberately compares rows where these two fields are present.

## Tests

Run the API contract checks and sampled offline/online comparison from the project root:

```powershell
pytest tests/ -v
```

The alignment test samples complete source rows, matches each row to `data/processed/features.parquet`, and verifies that its direct model probability differs from the API probability by no more than 0.05.

## Docker and WSL

The production image contains the API and serialized model; test files and datasets are mounted only by the Compose test service. Build and run the API:

```bash
docker compose up --build -d api
curl http://localhost:8000/health
```

Run the API and offline/online tests inside a disposable container:

```bash
docker compose --profile test run --build --rm tests
```

Stop the API with `docker compose down`.

WSL needs the project to be on a drive mounted under `/mnt`. Google Drive `G:` is not mounted in the current Ubuntu WSL distribution. Copy the project from Windows PowerShell to a mounted local drive, then run Compose from Ubuntu:

```powershell
robocopy "G:\Мой диск\scorify" "$env:USERPROFILE\scorify" /E /XD .git .pytest_cache __pycache__
```

```bash
cd /mnt/c/Users/<WindowsUser>/scorify
docker compose --profile test run --build --rm tests
docker compose up --build -d api
curl http://localhost:8000/health
```

Replace `<WindowsUser>` with the Windows profile directory name. If the image build cannot reach Debian package repositories to install `libgomp1`, configure the Docker/WSL network proxy or allow outbound access, then rerun the build. The API becomes available on port 8000.

## Rebuilding the Model

`src/features/build_features.py` creates the training parquet with missingness flags and derived credit features. `src/models/train_model.py` fits the LightGBM classifier and writes `models/lightgbm_model.pkl`; `src/models/predict_model.py` runs batch inference for the test dataset. The scripts currently contain Google Drive paths from the original notebook workflow and need those paths adapted to the local environment before running them outside that workflow.