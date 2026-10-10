import os
import sys

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_approved():
    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.05,
        "age": 45,
        "NumberOfTime30_59DaysPastDueNotWorse": 0,
        "DebtRatio": 0.1,
        "MonthlyIncome": 10000.0,
        "NumberOfOpenCreditLinesAndLoans": 5,
        "NumberOfTimes90DaysLate": 0,
        "NumberRealEstateLoansOrLines": 1,
        "NumberOfTime60_89DaysPastDueNotWorse": 0,
        "NumberOfDependents": 1.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "default_probability" in data
    assert 0.0 <= data["default_probability"] <= 1.0
    assert data["decision"] == "APPROVED"

def test_predict_high_risk():

    payload = {
        "RevolvingUtilizationOfUnsecuredLines": 0.99,
        "age": 22,
        "NumberOfTime30_59DaysPastDueNotWorse": 3,
        "DebtRatio": 0.8,
        "MonthlyIncome": 2000.0,
        "NumberOfOpenCreditLinesAndLoans": 2,
        "NumberOfTimes90DaysLate": 2,
        "NumberRealEstateLoansOrLines": 0,
        "NumberOfTime60_89DaysPastDueNotWorse": 1,
        "NumberOfDependents": 0.0
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()

    assert data["default_probability"] > 0.3
    assert data["decision"] == "REJECTED"