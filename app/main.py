from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
from fastapi import FastAPI

from app.schemas import CreditApplication, PredictionResponse

MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "lightgbm_model.pkl"
APPROVAL_THRESHOLD = 0.30
DEFAULT_MONTHLY_INCOME = 5400.0
DEFAULT_DEPENDENTS = 0.0

FEATURE_COLUMNS = [
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
    "MonthlyIncome_isna",
    "NumberOfDependents_isna",
    "Total_PastDue_Count",
    "Severe_PastDue_Ratio",
    "Income_Per_Person",
    "Total_Credit_Lines",
    "Monthly_Debt_Amount",
]

app = FastAPI(title="Scorify Credit Scoring API", version="1.0.0")


@lru_cache(maxsize=1)
def load_model():
    return joblib.load(MODEL_PATH)


def build_features(application: CreditApplication) -> pd.DataFrame:
    income_is_missing = application.monthly_income is None
    dependents_are_missing = application.dependents is None
    income = (
        DEFAULT_MONTHLY_INCOME
        if income_is_missing
        else application.monthly_income
    )
    dependents = (
        DEFAULT_DEPENDENTS if dependents_are_missing else application.dependents
    )
    past_due_total = (
        application.past_due_30_59
        + application.past_due_60_89
        + application.past_due_90
    )

    features = {
        "RevolvingUtilizationOfUnsecuredLines": application.revolving_utilization,
        "age": application.age,
        "NumberOfTime30-59DaysPastDueNotWorse": application.past_due_30_59,
        "DebtRatio": application.debt_ratio,
        "MonthlyIncome": income,
        "NumberOfOpenCreditLinesAndLoans": application.open_credit_lines,
        "NumberOfTimes90DaysLate": application.past_due_90,
        "NumberRealEstateLoansOrLines": application.real_estate_lines,
        "NumberOfTime60-89DaysPastDueNotWorse": application.past_due_60_89,
        "NumberOfDependents": dependents,
        "MonthlyIncome_isna": int(income_is_missing),
        "NumberOfDependents_isna": int(dependents_are_missing),
        "Total_PastDue_Count": past_due_total,
        "Severe_PastDue_Ratio": application.past_due_90 / (past_due_total + 1),
        "Income_Per_Person": income / (dependents + 1),
        "Total_Credit_Lines": application.open_credit_lines
        + application.real_estate_lines,
        "Monthly_Debt_Amount": income * application.debt_ratio,
    }
    return pd.DataFrame([features], columns=FEATURE_COLUMNS)


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.post("/predict", response_model=PredictionResponse)
def predict(application: CreditApplication) -> PredictionResponse:
    features = build_features(application)
    probability = float(load_model().predict_proba(features)[0, 1])
    decision = "APPROVED" if probability < APPROVAL_THRESHOLD else "REJECTED"
    return PredictionResponse(default_probability=probability, decision=decision)