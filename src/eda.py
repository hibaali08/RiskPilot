"""
Day 1 - Step 1: Exploratory Data Analysis
Run: python src/eda.py

Expects the raw Kaggle "Give Me Some Credit" file at:
    data/raw/cs-training.csv
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

DATA_PATH = "data/raw/cs-training.csv"
OUT_DIR = "outputs"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET = "SeriousDlqin2yrs"


def main():
    df = pd.read_csv(DATA_PATH, index_col=0)

    print("=" * 60)
    print("SHAPE:", df.shape)
    print("=" * 60)

    print("\nCOLUMN DTYPES:\n", df.dtypes)

    print("\nMISSING VALUES (count, %):")
    missing = df.isnull().sum()
    missing_pct = (missing / len(df) * 100).round(2)
    print(pd.concat([missing, missing_pct], axis=1, keys=["count", "pct"]))

    print("\nCLASS BALANCE (target = SeriousDlqin2yrs):")
    print(df[TARGET].value_counts())
    print(df[TARGET].value_counts(normalize=True).round(4))

    print("\nDESCRIBE:\n", df.describe().T)

    # Save class balance plot
    plt.figure(figsize=(5, 4))
    sns.countplot(x=TARGET, data=df)
    plt.title("Class Balance: Default (1) vs No Default (0)")
    plt.savefig(f"{OUT_DIR}/class_balance.png", bbox_inches="tight")
    plt.close()

    # Save correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Feature Correlation Heatmap")
    plt.savefig(f"{OUT_DIR}/correlation_heatmap.png", bbox_inches="tight")
    plt.close()

    print(f"\nPlots saved to {OUT_DIR}/class_balance.png and {OUT_DIR}/correlation_heatmap.png")
    print("\nDONE. Read the printed output above before moving to feature engineering.")


if __name__ == "__main__":
    main()
