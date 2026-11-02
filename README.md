# BITS Pilani - Machine Learning Assignment 2

## Mandatory Links
- **GitHub Repository Link:** https://github.com/lalitpaliwal-coder/BITS-ML-Assignment2
- **Live Streamlit App Link:** https://bits-ml-assignment2-drz5ndxejjh4gmhyqzbtwf.streamlit.app/

---

## a. Problem Statement
The objective of this assignment is to build, evaluate, and compare five distinct supervised Machine Learning classification algorithms on a high-dimensional dataset. The trained models are deployed as an interactive Streamlit web application hosted on Streamlit Community Cloud. The web interface allows end-users to upload test datasets in CSV format, select pre-trained models dynamically without retraining, and view real-time evaluation metrics, confusion matrices, and detailed classification reports.

---

## b. Dataset Description
The **UCI Breast Cancer Wisconsin (Diagnostic) Dataset** was selected for this study:
- **Source Repository:** UCI Machine Learning Repository / Scikit-Learn Datasets
- **Instance Count:** 569 instances (114 test instances used in an 80:20 stratified split)
- **Feature Count:** 30 numeric predictive features computed from digitized images of fine needle aspirates (FNA) of breast masses (exceeding minimum requirement of 12 features)
- **Target Classes:** Binary outcome (`target`: 0 = Malignant, 1 = Benign)
- **Features Included:** Mean, standard error, and worst dimension measurements for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

---

## c. GitHub Repository Directory Structure
```text
BITS-ML-Assignment2/
│── app.py
│── requirements.txt
│── README.md
│── test_data.csv
└── model/
    ├── logistic_regression.joblib
    ├── decision_tree.joblib
    ├── knn.joblib
    ├── naive_bayes.joblib
    └── svm.joblib
```
- **`app.py`:** Streamlit web interface application supporting CSV file upload, dynamic model loading, metric calculation, confusion matrix plotting, and classification report rendering.
- **`requirements.txt`:** Complete dependencies list ensuring consistent cloud runtime.
- **`test_data.csv`:** Unseen test subset (114 samples with 30 features and target label) exported during model training.
- **`README.md`:** Project overview, metrics table, and analytical observations.
- **`model/` directory:** Contains serialized pre-trained pipeline models (`logistic_regression.joblib`, `decision_tree.joblib`, `knn.joblib`, `naive_bayes.joblib`, `svm.joblib`).

---

## d. Models Used & Evaluation Metrics Comparison Table

All five classifiers were evaluated on the exact same test dataset (114 samples). The resulting metrics were calculated dynamically by the deployed Streamlit application as follows:

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 Score | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9161 | 0.9123 | 0.9130 | 0.8174 |
| **KNN** | 0.9561 | 0.9788 | 0.9561 | 0.9561 | 0.9560 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9384 | 0.9386 | 0.9384 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9825 | 0.9825 | 0.9825 | 0.9623 |

---

## e. Model Performance Observations

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Demonstrated exceptional overall performance with feature scaling. Achieved maximum Accuracy (98.25%) and the overall highest AUC Score (0.9954), proving that the feature space is strongly linearly separable after standardized scaling. |
| **Decision Tree** | Recorded the lowest generalizability among tested models (Accuracy 91.23%, MCC 0.8174). Axis-aligned decision boundaries created individual feature threshold splits that overfit training noise slightly compared to distance or margin-based approaches. |
| **KNN** | Achieved strong diagnostic performance (Accuracy 95.61%, MCC 0.9054). Standardized distance calculations ($k=5$) effectively captured local feature neighborhood geometry for cluster classification. |
| **Naive Bayes** | Produced robust probability estimations (Accuracy 93.86%, AUC 0.9878). Despite assuming conditional feature independence, Gaussian Naive Bayes handled feature correlations effectively. |
| **Support Vector Machine** | Tied for the highest classification accuracy (98.25%, MCC 0.9623, F1 0.9825). The RBF kernel accurately mapped continuous feature interactions into a hyperspace with maximum geometric separating margin. |
| **Overall Winner for Dataset** | **WINNER: Logistic Regression & Support Vector Machine (Tie)** — Both Logistic Regression and Support Vector Machine achieved identical top performance across Accuracy (0.9825), Precision (0.9825), Recall (0.9825), F1 Score (0.9825), and MCC (0.9623). Logistic Regression holds a slight edge due to its higher AUC Score (0.9954 vs 0.9950) and lower computational inference overhead. |