
# MLOps Lab 1 — MLflow Experiment Tracking

## Overview
This lab demonstrates experiment tracking using MLflow on the Titanic dataset.
A Random Forest Classifier is used to predict passenger survival.

## Project Structure
mlops-mlflow-lab1/
├── README.md
├── requirements/
│   └── requirements.txt
└── src/
    └── train.py

## Setup

### 1. Create and activate virtual environment
python -m venv venv
source venv/bin/activate        # Mac/Linux
venv\Scripts\activate           # Windows

### 2. Install dependencies
pip install -r requirements/requirements.txt

### 3. Run the experiment
python src/train.py

### 4. View results in MLflow UI
mlflow ui
# Open http://127.0.0.1:5000 in your browser

## Model
- Dataset: Titanic (via seaborn)
- Model: Random Forest Classifier (scikit-learn)
- Tracked metrics: Accuracy, F1 Score, Precision, Recall
- Tracked params: n_estimators, max_depth, random_state