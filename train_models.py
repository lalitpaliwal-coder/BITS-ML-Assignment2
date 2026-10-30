import os
import joblib
import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef
)

def train_and_save():
    # 1. Load Dataset
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name="target")
    
    # 2. Train-Test Split (80% Train, 20% Test)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # 3. Save Test Dataset CSV
    test_data = pd.concat([X_test, y_test], axis=1)
    test_data.to_csv("test_data.csv", index=False)
    print("Saved test_data.csv successfully.")
    
    # Create model directory
    os.makedirs("model", exist_ok=True)
    
    # 4. Define Pipelines & Models
    models = {
        "logistic_regression": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", LogisticRegression(random_state=42, max_iter=1000))
        ]),
        "decision_tree": Pipeline([
            ("classifier", DecisionTreeClassifier(random_state=42))
        ]),
        "knn": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier(n_neighbors=5))
        ]),
        "naive_bayes": Pipeline([
            ("classifier", GaussianNB())
        ]),
        "svm": Pipeline([
            ("scaler", StandardScaler()),
            ("classifier", SVC(probability=True, random_state=42))
        ])
    }
    
    # 5. Train, Evaluate, and Save Models
    results = []
    
    for key, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
        
        # Save trained pipeline model
        joblib.dump(model, f"model/{key}.joblib")
        
        # Calculate metrics
        acc = accuracy_score(y_test, y_pred)
        auc = roc_auc_score(y_test, y_proba)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        mcc = matthews_corrcoef(y_test, y_pred)
        
        results.append({
            "Model": key,
            "Accuracy": acc,
            "AUC": auc,
            "Precision": prec,
            "Recall": rec,
            "F1": f1,
            "MCC": mcc
        })
        print(f"Trained and saved: model/{key}.joblib")
        
    df_results = pd.DataFrame(results)
    print("\nTraining Metrics Summary:")
    print(df_results.to_string(index=False))

if __name__ == "__main__":
    train_and_save()