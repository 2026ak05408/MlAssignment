"""
Train five classifiers on the Wisconsin Diagnostic Breast Cancer dataset.
Run: python train.py
This downloads the dataset through scikit-learn, creates a stratified train/test
split, saves the five fitted pipelines, test_data.csv, and metrics.csv.
"""
from pathlib import Path
import joblib
import pandas as pd

from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef
)

ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models"
MODEL_DIR.mkdir(exist_ok=True)

# Load the public Wisconsin Diagnostic Breast Cancer dataset.
dataset = load_breast_cancer(as_frame=True)
X = dataset.data
y = dataset.target  # 0 = malignant, 1 = benign (as defined by scikit-learn)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)

models = {
    "Logistic Regression": Pipeline([
        ("scaler", StandardScaler()),
        ("model", LogisticRegression(max_iter=3000, random_state=42))
    ]),
    "Decision Tree": DecisionTreeClassifier(
        max_depth=5, random_state=42
    ),
    "K-Nearest Neighbors": Pipeline([
        ("scaler", StandardScaler()),
        ("model", KNeighborsClassifier(n_neighbors=5))
    ]),
    "Gaussian Naive Bayes": GaussianNB(),
    "Support Vector Machine": Pipeline([
        ("scaler", StandardScaler()),
        ("model", SVC(kernel="rbf", probability=True, random_state=42))
    ]),
}

rows = []
for display_name, estimator in models.items():
    estimator.fit(X_train, y_train)
    predictions = estimator.predict(X_test)

    # All five estimators expose predict_proba in this configuration.
    positive_scores = estimator.predict_proba(X_test)[:, 1]

    row = {
        "Model": display_name,
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, positive_scores),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }
    rows.append(row)
    filename = display_name.lower().replace(" ", "_") + ".joblib"
    joblib.dump(estimator, MODEL_DIR / filename)
    print(f"Saved {display_name}: {row}")

# Test file includes all input features and the target column, as required.
test_df = X_test.copy()
test_df["target"] = y_test
test_df.to_csv(ROOT / "test_data.csv", index=False)

metrics_df = pd.DataFrame(rows)
metrics_df.to_csv(ROOT / "metrics.csv", index=False)

print("\nMetrics comparison:")
print(metrics_df.round(4).to_string(index=False))
print("\nCreated test_data.csv, metrics.csv, and models/*.joblib")
