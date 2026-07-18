"""
Shared preprocessing: turns a single raw applicant record into the same
engineered feature vector the model was trained on (mirrors
feature_engineering.py, but for one record at a time instead of a batch).
"""

import pandas as pd


def engineer_single(raw: dict, feature_names: list) -> pd.DataFrame:
    """
    raw: dict with the 10 original "Give Me Some Credit" fields:
        RevolvingUtilizationOfUnsecuredLines, age,
        NumberOfTime30-59DaysPastDueNotWorse, DebtRatio, MonthlyIncome,
        NumberOfOpenCreditLinesAndLoans, NumberOfTimes90DaysLate,
        NumberRealEstateLoansOrLines, NumberOfTime60-89DaysPastDueNotWorse,
        NumberOfDependents
    feature_names: the exact ordered column list the model was trained on
        (loaded from models/feature_names.pkl) — needed to reindex correctly.
    """
    row = dict(raw)

    # --- same cleaning rules as feature_engineering.py ---
    row["RevolvingUtilizationOfUnsecuredLines"] = min(row["RevolvingUtilizationOfUnsecuredLines"], 2)
    row["DebtRatio"] = min(row["DebtRatio"], 5)  # approx cap, mirrors 99th-pct capping at train time

    # --- engineered features ---
    row["TotalPastDue"] = (
        row["NumberOfTime30-59DaysPastDueNotWorse"]
        + row["NumberOfTime60-89DaysPastDueNotWorse"]
        + row["NumberOfTimes90DaysLate"]
    )
    row["IncomePerDependent"] = row["MonthlyIncome"] / (row["NumberOfDependents"] + 1)

    # utilization bucket -> one-hot
    util = row["RevolvingUtilizationOfUnsecuredLines"]
    for col in ["UtilizationBucket_moderate", "UtilizationBucket_high", "UtilizationBucket_very_high"]:
        row[col] = 0
    if util <= 0.3:
        pass  # "low" is the dropped baseline category
    elif util <= 0.6:
        row["UtilizationBucket_moderate"] = 1
    elif util <= 0.9:
        row["UtilizationBucket_high"] = 1
    else:
        row["UtilizationBucket_very_high"] = 1

    # age bucket -> one-hot
    age = row["age"]
    for col in ["AgeBucket_adult", "AgeBucket_middle_age", "AgeBucket_senior"]:
        row[col] = 0
    if age <= 30:
        pass  # "young" is the dropped baseline category
    elif age <= 45:
        row["AgeBucket_adult"] = 1
    elif age <= 60:
        row["AgeBucket_middle_age"] = 1
    else:
        row["AgeBucket_senior"] = 1

    df = pd.DataFrame([row])
    # reindex to the exact training column order; fills anything missing with 0
    df = df.reindex(columns=feature_names, fill_value=0)
    return df
