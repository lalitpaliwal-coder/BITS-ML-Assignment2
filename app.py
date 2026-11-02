import os
import joblib
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st

from sklearn.metrics import (
    accuracy_score, roc_auc_score, precision_score, 
    recall_score, f1_score, matthews_corrcoef, 
    confusion_matrix, classification_report
)

# Page configuration
st.set_page_config(
    page_title="ML Classifier Evaluation Dashboard",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Machine Learning Model Evaluation Dashboard")
st.markdown("""
This application evaluates pre-trained Machine Learning classifiers on test datasets 
without retraining models, displaying metrics, confusion matrices, and detailed classification reports.
""")

# Model Mapping
MODEL_DICT = {
    "Logistic Regression": "model/logistic_regression.joblib",
    "Decision Tree": "model/decision_tree.joblib",
    "KNN": "model/knn.joblib",
    "Naive Bayes": "model/naive_bayes.joblib",
    "Support Vector Machine": "model/svm.joblib"
}

# Sidebar Layout
st.sidebar.header("⚙️ Configuration")

# Permanently visible label and caption
st.sidebar.markdown("**Upload Custom Test Dataset (CSV)**")
st.sidebar.caption("ℹ️ By default, `test_data.csv` from the repository is loaded. You can upload your own CSV file below to override it.")

uploaded_file = st.sidebar.file_uploader(
    "Choose a CSV file", 
    type=["csv"],
    label_visibility="collapsed"
)

selected_model_name = st.sidebar.selectbox("Select ML Model", list(MODEL_DICT.keys()))

target_col = st.sidebar.text_input("Target Column Name", value="target")

# Dataset Resolution Logic (Uploaded file takes priority; falls back to default test_data.csv if present)
test_df = None

if uploaded_file is not None:
    try:
        test_df = pd.read_csv(uploaded_file)
        st.success(f"Successfully loaded uploaded dataset with shape: {test_df.shape}")
    except Exception as e:
        st.error(f"Error loading uploaded CSV file: {str(e)}")
elif os.path.exists("test_data.csv"):
    try:
        test_df = pd.read_csv("test_data.csv")
        st.info("ℹ️ Automatically loaded repository test dataset (`test_data.csv`). You can also upload a custom CSV using the sidebar.")
    except Exception as e:
        st.error(f"Error reading repository test_data.csv: {str(e)}")
else:
    st.info("👆 Please upload a `test_data.csv` file using the sidebar to view evaluations.")

if test_df is not None:
    try:
        # Display dataset preview
        with st.expander("👀 View Test Dataset Preview", expanded=False):
            st.dataframe(test_df.head(10))
            
        if target_col not in test_df.columns:
            st.error(f"Target column '{target_col}' not found in the CSV. Please specify the correct column name in the sidebar.")
        else:
            X_test = test_df.drop(columns=[target_col])
            y_test = test_df[target_col]
            
            model_path = MODEL_DICT[selected_model_name]
            
            if not os.path.exists(model_path):
                st.error(f"Model file '{model_path}' not found in directory. Run `python train_models.py` first.")
            else:
                model = joblib.load(model_path)
                
                # Predictions
                y_pred = model.predict(X_test)
                if hasattr(model, "predict_proba"):
                    y_proba = model.predict_proba(X_test)[:, 1]
                elif hasattr(model, "decision_function"):
                    y_proba = model.decision_function(X_test)
                else:
                    y_proba = y_pred
                
                # Compute Metrics
                acc = accuracy_score(y_test, y_pred)
                auc = roc_auc_score(y_test, y_proba)
                prec = precision_score(y_test, y_pred, average='weighted')
                rec = recall_score(y_test, y_pred, average='weighted')
                f1 = f1_score(y_test, y_pred, average='weighted')
                mcc = matthews_corrcoef(y_test, y_pred)
                
                st.subheader(f"📈 Model Evaluation: {selected_model_name}")
                
                # Display Metrics in KPI Columns
                col1, col2, col3, col4, col5, col6 = st.columns(6)
                col1.metric("Accuracy", f"{acc:.4f}")
                col2.metric("AUC Score", f"{auc:.4f}")
                col3.metric("Precision", f"{prec:.4f}")
                col4.metric("Recall", f"{rec:.4f}")
                col5.metric("F1 Score", f"{f1:.4f}")
                col6.metric("MCC Score", f"{mcc:.4f}")
                
                st.markdown("---")
                
                # Visualization Section
                col_left, col_right = st.columns(2)
                
                with col_left:
                    st.subheader("📌 Confusion Matrix")
                    cm = confusion_matrix(y_test, y_pred)
                    fig, ax = plt.subplots(figsize=(5, 4))
                    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax, cbar=False)
                    ax.set_xlabel("Predicted")
                    ax.set_ylabel("Actual")
                    st.pyplot(fig)
                    
                with col_right:
                    st.subheader("📋 Classification Report")
                    report_dict = classification_report(y_test, y_pred, output_dict=True)
                    report_df = pd.DataFrame(report_dict).transpose()
                    st.dataframe(report_df.style.format("{:.4f}"))

    except Exception as e:
        st.error(f"Error processing dataset or model: {str(e)}")
