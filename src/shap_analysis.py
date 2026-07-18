"""
Day 2 - Step 1: SHAP Explainability
Run: python src/shap_analysis.py

Loads the trained model from Day 1 and generates:
  - a global feature importance plot (which features matter most overall)
  - a local explanation example (why one specific applicant got their score)
"""

import pandas as pd
import joblib
import shap
import matplotlib.pyplot as plt
import os

MODEL_PATH = "models/model.pkl"
FEATURES_PATH = "models/feature_names.pkl"
DATA_PATH = "data/processed/cleaned.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def main():
    model = joblib.load(MODEL_PATH)
    feature_names = joblib.load(FEATURES_PATH)
    df = pd.read_csv(DATA_PATH)
    X = df[feature_names]

    # Use a sample for speed (SHAP on the full 150k rows is slow)
    X_sample = X.sample(n=min(2000, len(X)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)

    # --- Global importance: which features matter most across all applicants ---
    plt.figure()
    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_global_importance.png", bbox_inches="tight")
    plt.close()
    print(f"Saved global importance plot to {OUT_DIR}/shap_global_importance.png")

    # --- Global summary (direction of effect, not just magnitude) ---
    plt.figure()
    shap.summary_plot(shap_values, X_sample, show=False)
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_summary.png", bbox_inches="tight")
    plt.close()
    print(f"Saved summary plot to {OUT_DIR}/shap_summary.png")

    # --- Local explanation: pick one high-risk applicant and explain their score ---
    probs = model.predict_proba(X_sample)[:, 1]
    X_sample = X_sample.reset_index(drop=True)
    high_risk_idx = probs.argmax()

    print("\n" + "=" * 60)
    print(f"Example local explanation (applicant #{high_risk_idx}, predicted risk = {probs[high_risk_idx]:.3f})")
    print("=" * 60)

    row_shap = shap_values[high_risk_idx]
    contributions = pd.Series(row_shap, index=feature_names).sort_values(key=abs, ascending=False)
    print("\nTop 5 factors driving this prediction:")
    for feat, val in contributions.head(5).items():
        direction = "increases" if val > 0 else "decreases"
        print(f"  {feat}: {direction} risk (SHAP value = {val:.4f})")

    plt.figure()
    shap.force_plot(
        explainer.expected_value, row_shap, X_sample.iloc[high_risk_idx], matplotlib=True, show=False
    )
    plt.tight_layout()
    plt.savefig(f"{OUT_DIR}/shap_local_example.png", bbox_inches="tight")
    plt.close()
    print(f"\nSaved local explanation plot to {OUT_DIR}/shap_local_example.png")


if __name__ == "__main__":
    main()
