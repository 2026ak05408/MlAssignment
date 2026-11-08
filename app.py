
from pathlib import Path
import joblib
import pandas as pd
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, recall_score,
    f1_score, matthews_corrcoef, confusion_matrix, classification_report
)

ROOT = Path(__file__).resolve().parent

# Model files are stored in the same directory as this Python file
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "k-nearest_neighbors.joblib",
    "Gaussian Naive Bayes": "gaussian_naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}

st.set_page_config(
    page_title="Classifier Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("Machine Learning Classifier Evaluation")

st.write(
    "Upload the held-out test CSV containing the 30 input features and the "
    "`target` column. The saved models are loaded for prediction; the app "
    "does not retrain them."
)

uploaded = st.file_uploader("Upload test_data.csv", type=["csv"])

if uploaded is None:
    st.info("Run `python train.py` first, then upload the generated test_data.csv.")
    st.stop()

data = pd.read_csv(uploaded)

if "target" not in data.columns:
    st.error("The uploaded CSV must contain a `target` column.")
    st.stop()

X_test = data.drop(columns=["target"])
y_test = data["target"]

# Check whether all model files exist in the current directory
missing_models = [
    f for f in MODEL_FILES.values()
    if not (ROOT / f).exists()
]

if missing_models:
    st.error(f"Saved model files are missing: {missing_models}")
    st.stop()

choice = st.selectbox("Choose a classifier", list(MODEL_FILES.keys()))

# Load selected model directly from the current directory
model = joblib.load(ROOT / MODEL_FILES[choice])

try:
    predictions = model.predict(X_test)
    scores = model.predict_proba(X_test)[:, 1]

    metrics = {
        "Accuracy": accuracy_score(y_test, predictions),
        "AUC": roc_auc_score(y_test, scores),
        "Precision": precision_score(y_test, predictions, zero_division=0),
        "Recall": recall_score(y_test, predictions, zero_division=0),
        "F1 Score": f1_score(y_test, predictions, zero_division=0),
        "MCC": matthews_corrcoef(y_test, predictions),
    }

    st.subheader(f"Results: {choice}")

    cols = st.columns(3)

    for i, (name, value) in enumerate(metrics.items()):
        cols[i % 3].metric(name, f"{value:.4f}")

    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_test, predictions)

    st.dataframe(
        pd.DataFrame(
            cm,
            index=["Actual 0", "Actual 1"],
            columns=["Predicted 0", "Predicted 1"]
        ),
        use_container_width=True
    )

    st.subheader("Classification Report")

    report = classification_report(
        y_test,
        predictions,
        output_dict=True,
        zero_division=0
    )

    st.dataframe(
        pd.DataFrame(report).transpose().round(4),
        use_container_width=True
    )

except Exception as exc:
    st.error(f"Could not evaluate this file: {exc}")
    st.write(
        "Check that the CSV has the same feature columns as the training dataset."
    )
