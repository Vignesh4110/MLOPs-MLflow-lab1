# MLOps Lab 1 — MLflow Experiment Tracking

## Overview
This lab demonstrates experiment tracking using MLflow on the Titanic dataset.
A Random Forest Classifier is trained to predict passenger survival, with two runs
logged and compared using the MLflow UI.

---

## Folder Structure
```
mlops-mlflow-lab1/
├── README.md
├── .gitignore
├── requirements/
│   └── requirements.txt
└── src/
    └── train.py
```

---

## Dataset
- **Source:** Titanic dataset (loaded via the seaborn library)
- **Target:** `survived` (0 = did not survive, 1 = survived)
- **Features used:** `pclass`, `sex`, `age`, `sibsp`, `parch`, `fare`, `embarked`

---

## Model
- **Algorithm:** Random Forest Classifier (scikit-learn)
- **Two runs tracked:**
  - `RF_default` — 100 trees, no depth limit
  - `RF_tuned` — 200 trees, max depth of 6

---

## Metrics Tracked
| Metric | Description |
|---|---|
| Accuracy | Overall correct predictions |
| F1 Score | Balance between precision and recall |
| Precision | Of predicted survivors, how many actually survived |
| Recall | Of actual survivors, how many were correctly identified |

---

## Setup Instructions

### 1. Clone the repository
```bash
git clone https://github.com/YOUR_USERNAME/mlops-mlflow-lab1.git
cd mlops-mlflow-lab1
```

### 2. Create and activate a virtual environment
```bash
python -m venv venv

# Mac/Linux
source venv/bin/activate

# Windows
venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements/requirements.txt
```

### 4. Run the experiment
```bash
python src/train.py
```

### 5. View results in MLflow UI
```bash
mlflow ui
```
Then open **http://127.0.0.1:5000** in your browser. Navigate to
**Training runs** to see both runs with all logged parameters and metrics.

---

## Results

| Run | n_estimators | max_depth | Accuracy | F1 Score |
|---|---|---|---|---|
| RF_default | 100 | None | 0.821 | 0.775 |
| RF_tuned | 200 | 6 | 0.816 | 0.752 |

---

## Tools Used
- Python 3.x
- MLflow 3.10.1
- scikit-learn
- pandas
- seaborn