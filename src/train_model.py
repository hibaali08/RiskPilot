"""
Day 1 - Step 3: Train baseline + XGBoost models
Run: python src/train_model.py

Reads data/processed/cleaned.csv, trains a Logistic Regression baseline
and an XGBoost model, evaluates both with AUC-ROC and the KS statistic
(the metric real credit risk teams report), and saves the best model.
"""

import pandas as pd
import numpy as np
import joblib
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, classification_report, roc_curve
from xgboost import XGBClassifier

DATA_PATH = "data/processed/cleaned.csv"
MODEL_DIR = "models"
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def ks_statistic(y_true, y_prob):
    """Kolmogorov-Smirnov statistic - standard credit risk model metric.
    Measures the max separation between the cumulative distributions of
    good and bad borrowers' predicted scores."""
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    return max(tpr - fpr)


def main():
    df = pd.read_csv(DATA_PATH)
    X = df.drop(columns=[TARGET])
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # ---------- Baseline: Logistic Regression ----------
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lr = LogisticRegression(max_iter=1000, class_weight="balanced")
    lr.fit(X_train_scaled, y_train)
    lr_probs = lr.predict_proba(X_test_scaled)[:, 1]

    print("=" * 60)
    print("LOGISTIC REGRESSION (baseline)")
    print("AUC-ROC:", round(roc_auc_score(y_test, lr_probs), 4))
    print("KS statistic:", round(ks_statistic(y_test, lr_probs), 4))
    print(classification_report(y_test, lr.predict(X_test_scaled)))

    # ---------- XGBoost ----------
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    xgb = XGBClassifier(
        n_estimators=300,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        scale_pos_weight=scale_pos_weight,
        eval_metric="auc",
        random_state=42,
    )
    xgb.fit(X_train, y_train)
    xgb_probs = xgb.predict_proba(X_test)[:, 1]

    print("=" * 60)
    print("XGBOOST")
    print("AUC-ROC:", round(roc_auc_score(y_test, xgb_probs), 4))
    print("KS statistic:", round(ks_statistic(y_test, xgb_probs), 4))
    print(classification_report(y_test, xgb.predict(X_test)))

    # ---------- Save the better model (XGBoost should win, but check) ----------
    if roc_auc_score(y_test, xgb_probs) >= roc_auc_score(y_test, lr_probs):
        joblib.dump(xgb, f"{MODEL_DIR}/model.pkl")
        print(f"\nSaved XGBoost model to {MODEL_DIR}/model.pkl")
    else:
        joblib.dump({"model": lr, "scaler": scaler}, f"{MODEL_DIR}/model.pkl")
        print(f"\nSaved Logistic Regression model to {MODEL_DIR}/model.pkl")

    joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.pkl")
    print("Saved feature names for use in Day 2 (SHAP + API).")


if __name__ == "__main__":
    main()
