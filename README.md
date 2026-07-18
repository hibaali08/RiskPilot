# Explainable Credit Risk Scoring — Day 1

## Setup
```bash
cd credit-risk-project
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Get the data
1. Go to https://www.kaggle.com/c/GiveMeSomeCredit/data (free Kaggle account required)
2. Download `cs-training.csv`
3. Place it at `data/raw/cs-training.csv`

## Run, in order
```bash
python src/eda.py                  # explore the data, saves plots to outputs/
python src/feature_engineering.py  # cleans + engineers features -> data/processed/cleaned.csv
python src/train_model.py          # trains LR baseline + XGBoost, saves best model -> models/model.pkl
```

## What you should have by end of Day 1
- `outputs/class_balance.png`, `outputs/correlation_heatmap.png`
- `data/processed/cleaned.csv`
- `models/model.pkl`, `models/feature_names.pkl`
- Printed AUC-ROC and KS statistic for both models (write these numbers down — you'll need them for the README and resume bullet later)
