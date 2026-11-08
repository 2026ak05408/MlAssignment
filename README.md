# Machine Learning Assignment 2 – Classification Model Comparison

## a. Problem Statement
The objective is to build and compare five machine-learning classification models
for predicting whether a breast tumour is malignant or benign. The models are
evaluated using Accuracy, AUC, Precision, Recall, F1 Score, and Matthews
Correlation Coefficient (MCC). A Streamlit application allows a user to upload
held-out test data and view evaluation results.

## b. Dataset Description
- **Dataset:** Wisconsin Diagnostic Breast Cancer (WDBC)
- **Source:** UCI Machine Learning Repository; distributed through `sklearn.datasets.load_breast_cancer`
- **Instances:** 569
- **Input features:** 30 numeric features
- **Target:** Binary classification (malignant / benign)
- **Split:** 80% training and 20% testing, stratified, `random_state=42`

The target encoding follows scikit-learn: 0 = malignant and 1 = benign.

## c. GitHub Repository Link
[Paste your public GitHub repository URL here.]

## d. Models Used and Evaluation

The five models implemented on the same training/test split are:
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbors (KNN)
4. Gaussian Naive Bayes
5. Support Vector Machine (SVM)

Run `python train.py` to generate `metrics.csv`. Copy the resulting values into
the comparison table below.

| ML Model | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9211 | 0.9163 | 0.9565 | 0.9167 | 0.9362 | 0.8341 |
| KNN | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| Gaussian Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Support Vector Machine | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

### Observations on model performance
The following observations are based on the generated held-out test results. Review them and add your own brief interpretation before submission. Precision and recall treat class 1 (benign) as the positive class.
Do not claim a model is best unless the results support it.

| Model | Observation |
|---|---|
| Logistic Regression | Highest AUC (0.9954) and tied-highest accuracy (0.9825), F1 (0.9861) and MCC (0.9623); strongest overall result in this split. |
| Decision Tree | Lowest accuracy (0.9211), AUC (0.9163) and MCC (0.8341) among these models; it performed less strongly on this test split. |
| KNN | Good performance with 0.9561 accuracy and 0.9655 F1; it performed between the linear/kernel models and the decision tree. |
| Gaussian Naive Bayes | AUC is strong at 0.9878, but accuracy (0.9386) and MCC (0.8676) are lower than LR, SVM and KNN. |
| Support Vector Machine | Tied with Logistic Regression on accuracy, precision, recall, F1 and MCC; AUC is slightly lower at 0.9950. |
| Overall winner | Logistic Regression narrowly leads on AUC and ties SVM on the other listed metrics in this test split. |

## Streamlit Application
The application provides:
- CSV test-data upload
- Classifier selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report

The app loads saved model pipelines and evaluates the uploaded test data. It
does not train models on the uploaded test CSV.

## Repository Contents
- `train.py` – dataset loading, train/test split, training, evaluation and saving
- `app.py` – Streamlit interface
- `requirements.txt` – Python dependencies
- `test_data.csv` – held-out test set with target column
- `metrics.csv` – generated model comparison
- `models/` – saved trained model pipelines

## How to Run
```bash
pip install -r requirements.txt
python train.py
streamlit run app.py
```

## Live Streamlit App Link
[Paste your deployed Streamlit Community Cloud URL here.]

## BITS Virtual Lab Screenshot
[Insert your own screenshot of assignment execution on BITS Virtual Lab.]
