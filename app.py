import joblib
from pathlib import Path
import pandas as pd
import streamlit as st
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "kNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Random Forest": "random_forest.joblib",
}
st.set_page_config(page_title="UCI Breast Cancer Classifier", page_icon="📊", layout="wide")
st.title("📊 UCI Breast Cancer Classification")
st.write("Interactive comparison of five classification models using the UCI Breast Cancer Wisconsin (Diagnostic) dataset.")

model_name = st.sidebar.selectbox("Select classification model", list(MODEL_FILES))
uploaded = st.sidebar.file_uploader("Upload test_data.csv", type=["csv"])
default = BASE / "test_data.csv"
if uploaded:
    data = pd.read_csv(uploaded)
elif default.exists():
    data = pd.read_csv(default)
else:
    st.error("Upload test_data.csv to continue.")
    st.stop()

if "target" not in data.columns:
    st.error("CSV must contain a 'target' column.")
    st.stop()

X = data.drop(columns=["target"])
y = data["target"]
model = joblib.load(MODEL_DIR / MODEL_FILES[model_name])
pred = model.predict(X)
prob = model.predict_proba(X)[:,1]

m = {
    "Accuracy": accuracy_score(y,pred),
    "AUC": roc_auc_score(y,prob),
    "Precision": precision_score(y,pred,zero_division=0),
    "Recall": recall_score(y,pred,zero_division=0),
    "F1 Score": f1_score(y,pred,zero_division=0),
    "MCC": matthews_corrcoef(y,pred)
}
st.subheader(f"Evaluation Metrics — {model_name}")
cols = st.columns(6)
for c,(k,v) in zip(cols,m.items()):
    c.metric(k, f"{v:.4f}")

c1,c2 = st.columns(2)
with c1:
    st.subheader("Confusion Matrix")
    cm = confusion_matrix(y,pred)
    st.dataframe(pd.DataFrame(cm,index=["Actual 0","Actual 1"],columns=["Predicted 0","Predicted 1"]),use_container_width=True)
with c2:
    st.subheader("Classification Report")
    r = classification_report(y,pred,output_dict=True,zero_division=0)
    st.dataframe(pd.DataFrame(r).T.round(4),use_container_width=True)

st.subheader("Test Data Preview")
st.dataframe(data.head(20),use_container_width=True)
