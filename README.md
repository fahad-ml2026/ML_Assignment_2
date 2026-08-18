# Machine Learning Assignment 2 — UCI Breast Cancer Classification

## a. Problem Statement
Implement multiple classification models on a public dataset, evaluate them using six metrics, and demonstrate the models through an interactive Streamlit application.

## b. Dataset Description
**Dataset:** Breast Cancer Wisconsin (Diagnostic)  
**Source:** UCI Machine Learning Repository  
**Instances:** 569  
**Features:** 30  
**Task:** Binary classification. The target distinguishes malignant and benign breast-mass cases.

Official UCI source: https://archive.ics.uci.edu/dataset/17/breast

The dataset exceeds the assignment minimum of 500 instances and 12 features.

## c. Github Repository Link
PASTE YOUR GITHUB REPOSITORY LINK HERE

## d. Models Used
1. Logistic Regression
2. Decision Tree Classifier
3. K-Nearest Neighbor (kNN)
4. Gaussian Naive Bayes
5. Random Forest (Ensemble)

### Comparison Table
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| Decision Tree | 0.9035 | 0.9373 | 0.9420 | 0.9028 | 0.9220 | 0.7969 |
| kNN | 0.9737 | 0.9944 | 0.9600 | 1.0000 | 0.9796 | 0.9442 |
| Naive Bayes | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| Random Forest | 0.9561 | 0.9944 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |

### Observations
- **Logistic Regression:** Strong baseline after feature scaling; performs well on this dataset.
- **Decision Tree:** Captures non-linear relationships but is more sensitive to tree depth.
- **kNN:** Benefits from feature scaling and an appropriate neighborhood size.
- **Naive Bayes:** Fast probabilistic baseline with competitive performance.
- **Random Forest:** Robust ensemble model that handles non-linear relationships effectively.

**Overall Winner (highest F1 on the held-out test set): Logistic Regression.**

## Streamlit Application
The app provides:
- CSV test-data upload
- Model-selection dropdown
- Accuracy, AUC, Precision, Recall, F1 and MCC
- Confusion matrix
- Classification report
- Test-data preview

**Live Streamlit App:** PASTE YOUR STREAMLIT APP LINK HERE

## Project Structure
```text
ML_Assignment_2/
├── app.py
├── requirements.txt
├── README.md
├── test_data.csv
├── metrics.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── random_forest.joblib
```

## Run Locally
```bash
pip install -r requirements.txt
streamlit run app.py
```
