"""
Day 1 - Step 2: Feature Engineering
Run: python src/feature_engineering.py

Reads data/raw/cs-training.csv, cleans it, adds engineered features,
and writes data/processed/cleaned.csv
"""

import pandas as pd
import numpy as np
import os

RAW_PATH = "data/raw/cs-training.csv"
OUT_PATH = "data/processed/cleaned.csv"
os.makedirs("data/processed", exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def main():
    df = pd.read_csv(RAW_PATH, index_col=0)

    # --- Handle missing values ---
    # MonthlyIncome and NumberOfDependents have missing values in this dataset
    df["MonthlyIncome"] = df["MonthlyIncome"].fillna(df["MonthlyIncome"].median())
    df["NumberOfDependents"] = df["NumberOfDependents"].fillna(0)

    # --- Clean obvious outliers/data errors ---
    # Age of 0 is invalid
    df = df[df["age"] > 0]

    # RevolvingUtilizationOfUnsecuredLines should logically be <= ~2 (allow some buffer);
    # cap extreme outliers instead of dropping rows
    df["RevolvingUtilizationOfUnsecuredLines"] = df["RevolvingUtilizationOfUnsecuredLines"].clip(upper=2)

    # DebtRatio has extreme outliers (some >10000) - cap at 99th percentile
    debt_cap = df["DebtRatio"].quantile(0.99)
    df["DebtRatio"] = df["DebtRatio"].clip(upper=debt_cap)

    # --- Feature engineering ---
    # Total number of times past due (combines 3 existing columns)
    df["TotalPastDue"] = (
        df["NumberOfTime30-59DaysPastDueNotWorse"]
        + df["NumberOfTime60-89DaysPastDueNotWorse"]
        + df["NumberOfTimes90DaysLate"]
    )

    # Income per dependent (avoid div by zero)
    df["IncomePerDependent"] = df["MonthlyIncome"] / (df["NumberOfDependents"] + 1)

    # Credit utilization bucket (industry-standard way underwriters think about this)
    df["UtilizationBucket"] = pd.cut(
        df["RevolvingUtilizationOfUnsecuredLines"],
        bins=[-0.01, 0.3, 0.6, 0.9, 2.0],
        labels=["low", "moderate", "high", "very_high"],
    )

    # Age bucket
    df["AgeBucket"] = pd.cut(
        df["age"],
        bins=[0, 30, 45, 60, 120],
        labels=["young", "adult", "middle_age", "senior"],
    )

    # One-hot encode the new categorical buckets
    df = pd.get_dummies(df, columns=["UtilizationBucket", "AgeBucket"], drop_first=True)

    df.to_csv(OUT_PATH, index=False)
    print(f"Saved cleaned + engineered dataset to {OUT_PATH}")
    print("Final shape:", df.shape)
    print("Final columns:", list(df.columns))


if __name__ == "__main__":
    main()
