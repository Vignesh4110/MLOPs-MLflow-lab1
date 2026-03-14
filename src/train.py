import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.preprocessing import LabelEncoder
from mlflow.models import infer_signature

# ── 1. Load Titanic dataset (built into seaborn, no download needed) ──────────
import seaborn as sns
data = sns.load_dataset("titanic")

# ── 2. Simple preprocessing ───────────────────────────────────────────────────
# Keep only useful columns
data = data[["survived", "pclass", "sex", "age", "sibsp", "parch", "fare", "embarked"]]

# Fill missing values
data["age"].fillna(data["age"].median(), inplace=True)
data["embarked"].fillna("S", inplace=True)

# Convert text columns to numbers
data["sex"]      = LabelEncoder().fit_transform(data["sex"])       # male=1, female=0
data["embarked"] = LabelEncoder().fit_transform(data["embarked"])  # C=0, Q=1, S=2

# ── 3. Split into features (X) and target (y) ─────────────────────────────────
X = data.drop("survived", axis=1)
y = data["survived"]
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# ── 4. Helper function to compute metrics ─────────────────────────────────────
def eval_metrics(y_true, y_pred):
    acc       = accuracy_score(y_true, y_pred)
    f1        = f1_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall    = recall_score(y_true, y_pred)
    return acc, f1, precision, recall

# ── 5. Set the MLflow experiment name ─────────────────────────────────────────
mlflow.set_experiment("titanic-survival-prediction")

# ── 6. Run 1: Random Forest with default settings ─────────────────────────────
print("Training Run 1: Random Forest (default)...")
with mlflow.start_run(run_name="RF_default"):
    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc, f1, precision, recall = eval_metrics(y_test, preds)

    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

    mlflow.log_param("model_type",    "RandomForestClassifier")
    mlflow.log_param("n_estimators",  100)
    mlflow.log_param("max_depth",     "None")
    mlflow.log_param("random_state",  42)
    mlflow.log_metric("accuracy",     acc)
    mlflow.log_metric("f1_score",     f1)
    mlflow.log_metric("precision",    precision)
    mlflow.log_metric("recall",       recall)

    signature = infer_signature(X_train, model.predict(X_train))
    mlflow.sklearn.log_model(model, "model", signature=signature)

# ── 7. Run 2: Random Forest with tuned settings (so you have 2 runs to compare)
print("Training Run 2: Random Forest (tuned)...")
with mlflow.start_run(run_name="RF_tuned"):
    model2 = RandomForestClassifier(n_estimators=200, max_depth=6, random_state=42)
    model2.fit(X_train, y_train)
    preds2 = model2.predict(X_test)
    acc, f1, precision, recall = eval_metrics(y_test, preds2)

    print(f"  Accuracy: {acc:.3f} | F1: {f1:.3f} | Precision: {precision:.3f} | Recall: {recall:.3f}")

    mlflow.log_param("model_type",    "RandomForestClassifier")
    mlflow.log_param("n_estimators",  200)
    mlflow.log_param("max_depth",     6)
    mlflow.log_param("random_state",  42)
    mlflow.log_metric("accuracy",     acc)
    mlflow.log_metric("f1_score",     f1)
    mlflow.log_metric("precision",    precision)
    mlflow.log_metric("recall",       recall)

    signature = infer_signature(X_train, model2.predict(X_train))
    mlflow.sklearn.log_model(model2, "model", signature=signature)

print("\nAll done! Now run:  mlflow ui")