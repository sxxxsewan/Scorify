import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import joblib
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from app.main import FEATURE_COLUMNS, app

RAW_PATH = os.path.join(PROJECT_ROOT, "data", "raw", "cs-training.csv")
FEATURES_PATH = os.path.join(
    PROJECT_ROOT, "data", "processed", "features.parquet"
)
MODEL_PATH = os.path.join(PROJECT_ROOT, "models", "lightgbm_model.pkl")
RAW_FEATURE_COLUMNS = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]

client = TestClient(app)


def test_online_predictions_match_processed_features():
    raw_data = pd.read_csv(RAW_PATH)
    processed_data = pd.read_parquet(FEATURES_PATH)
    model = joblib.load(MODEL_PATH)

    eligible_rows = raw_data.dropna(subset=RAW_FEATURE_COLUMNS)
    sample = eligible_rows.sample(n=min(20, len(eligible_rows)), random_state=42)
    model_features = processed_data[FEATURE_COLUMNS]

    for _, raw_row in sample.iterrows():
        payload = raw_row[RAW_FEATURE_COLUMNS].to_dict()
        response = client.post("/predict", json=payload)
        assert response.status_code == 200, response.text

        matching_rows = np.ones(len(processed_data), dtype=bool)
        for column in RAW_FEATURE_COLUMNS:
            matching_rows &= np.isclose(
                processed_data[column].to_numpy(dtype=float),
                float(raw_row[column]),
                rtol=0,
                atol=1e-8,
            )
        matching_rows &= processed_data["MonthlyIncome_isna"].to_numpy() == 0
        matching_rows &= processed_data["NumberOfDependents_isna"].to_numpy() == 0
        assert matching_rows.any(), "Raw row was not found in processed features"

        offline_probabilities = model.predict_proba(
            model_features.loc[matching_rows]
        )[:, 1]
        online_probability = response.json()["default_probability"]
        assert np.max(np.abs(offline_probabilities - online_probability)) <= 0.05