"""
Day 2 - Step 2: FastAPI backend
Run: uvicorn src.api:app --reload
Then open: http://127.0.0.1:8000/docs to test it interactively.

Serves risk predictions + plain-language SHAP explanations for a single applicant.
"""

import joblib
import shap
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from src.preprocessing import engineer_single

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_names.pkl"

app = FastAPI(title="Explainable Credit Risk API")

# Allow the Day 3 React frontend (running on a different port) to call this API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
feature_names = joblib.load(FEATURES_PATH)
explainer = shap.TreeExplainer(model)


class ApplicantInput(BaseModel):
    RevolvingUtilizationOfUnsecuredLines: float = Field(..., example=0.45)
    age: int = Field(..., example=35)
    NumberOfTime30_59DaysPastDueNotWorse: int = Field(..., example=0, alias="NumberOfTime30-59DaysPastDueNotWorse")
    DebtRatio: float = Field(..., example=0.3)
    MonthlyIncome: float = Field(..., example=5000)
    NumberOfOpenCreditLinesAndLoans: int = Field(..., example=6)
    NumberOfTimes90DaysLate: int = Field(..., example=0)
    NumberRealEstateLoansOrLines: int = Field(..., example=1)
    NumberOfTime60_89DaysPastDueNotWorse: int = Field(..., example=0, alias="NumberOfTime60-89DaysPastDueNotWorse")
    NumberOfDependents: int = Field(..., example=1)

    class Config:
        populate_by_name = True


def risk_grade(prob: float) -> str:
    """Convert probability to an A-E risk grade, mimicking real scorecards."""
    if prob < 0.05:
        return "A"
    elif prob < 0.15:
        return "B"
    elif prob < 0.30:
        return "C"
    elif prob < 0.50:
        return "D"
    else:
        return "E"


def explain_in_words(feature: str, value: float) -> str:
    """Turn a SHAP feature name + value into a plain-English phrase."""
    direction = "increases" if value > 0 else "decreases"
    readable = feature.replace("_", " ")
    return f"{readable} {direction} risk"


@app.get("/")
def root():
    return {"status": "Explainable Credit Risk API is running"}


@app.post("/predict")
def predict(applicant: ApplicantInput):
    raw = applicant.dict(by_alias=True)
    X = engineer_single(raw, feature_names)

    prob = float(model.predict_proba(X)[:, 1][0])
    grade = risk_grade(prob)

    shap_values = explainer.shap_values(X)[0]
    contributions = sorted(
        zip(feature_names, shap_values), key=lambda x: abs(x[1]), reverse=True
    )[:3]

    top_factors = [
        {"feature": feat, "shap_value": round(float(val), 4), "explanation": explain_in_words(feat, val)}
        for feat, val in contributions
    ]

    return {
        "risk_probability": round(prob, 4),
        "risk_grade": grade,
        "decision": "Denied" if grade in ["D", "E"] else "Approved",
        "top_factors": top_factors,
    }
