import io
import zipfile
import urllib.request
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier

BASE = Path(__file__).resolve().parent
MODEL_DIR = BASE / "model"
MODEL_DIR.mkdir(exist_ok=True)

# Official UCI Bank Marketing dataset
UCI_URL = "https://archive.ics.uci.edu/static/public/222/bank+marketing.zip"

def load_data():
    print("Downloading Bank Marketing dataset from UCI...")
    raw = urllib.request.urlopen(UCI_URL).read()
    z = zipfile.ZipFile(io.BytesIO(raw))
    name = next(n for n in z.namelist() if n.endswith("bank-full.csv"))
    with z.open(name) as f:
        return pd.read_csv(f, sep=";")

def build_preprocessor(X):
    numeric = X.select_dtypes(include=["number"]).columns.tolist()
    categorical = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore"))
    ])

    return ColumnTransformer([
        ("num", numeric_pipe, numeric),
        ("cat", categorical_pipe, categorical)
    ])

def main():
    df = load_data()
    df["y"] = df["y"].map({"no": 0, "yes": 1})

    X = df.drop(columns=["y"])
    y = df["y"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1500, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=8, min_samples_leaf=5, class_weight="balanced", random_state=42
        ),
        "kNN": KNeighborsClassifier(n_neighbors=15),
        "Naive Bayes": GaussianNB(),
        "Random Forest": RandomForestClassifier(
            n_estimators=250, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        )
    }

    metrics = []
    for name, estimator in models.items():
        preprocessor = build_preprocessor(X_train)

        # GaussianNB needs dense data; the other models work with sparse/dense.
        if name == "Naive Bayes":
            preprocessor.set_params(cat__onehot__sparse_output=False)

        pipe = Pipeline([
            ("preprocessor", preprocessor),
            ("model", estimator)
        ])

        print(f"Training {name}...")
        pipe.fit(X_train, y_train)

        pred = pipe.predict(X_test)
        prob = pipe.predict_proba(X_test)[:, 1]

        row = {
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, prob),
            "Precision": precision_score(y_test, pred, zero_division=0),
            "Recall": recall_score(y_test, pred, zero_division=0),
            "F1": f1_score(y_test, pred, zero_division=0),
            "MCC": matthews_corrcoef(y_test, pred)
        }
        metrics.append(row)

        filename = name.lower().replace(" ", "_").replace("-", "_") + ".joblib"
        joblib.dump(pipe, MODEL_DIR / filename)

    metrics_df = pd.DataFrame(metrics)
    metrics_df.to_csv(BASE / "metrics.csv", index=False)

    # Test data only, as requested by the assignment.
    test_data = X_test.copy()
    test_data["y"] = y_test.values
    test_data.to_csv(BASE / "test_data.csv", index=False)

    winner = metrics_df.loc[metrics_df["F1"].idxmax(), "ML Model Name"]
    print("\nMetrics:")
    print(metrics_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    print(f"\nOverall winner by F1 score: {winner}")
    print("\nCreated: test_data.csv, metrics.csv and model/*.joblib")

if __name__ == "__main__":
    main()
