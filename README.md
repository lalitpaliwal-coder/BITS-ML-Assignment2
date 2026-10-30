# Machine Learning Assignment - 2: Classification Models & Deployment

## a. Problem Statement
The objective of this project is to implement, evaluate, and compare five supervised Machine Learning classification models on a high-dimensional tabular dataset. The models are packaged and deployed as an interactive web application on Streamlit Community Cloud. The app reads pre-trained models and dynamically computes classification performance metrics on an uploaded test dataset without retraining.

---

## b. Dataset Description
- **Dataset Name:** UCI Breast Cancer Wisconsin (Diagnostic) Dataset
- **Source:** UCI Machine Learning Repository
- **Total Features:** 30 numeric features (computed from digitized images of fine needle aspirates of breast masses)
- **Total Instances:** 569 instances (114 test instances used in 80-20 stratified split)
- **Target Variable:** `target` (0 = Malignant, 1 = Benign)
- **Features Include:** Mean, standard error, and worst values for radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, and fractal dimension.

---

## c. GitHub Repository Link
`https://github.com/YOUR_GITHUB_USERNAME/bits-ml-assignment2` *(Replace with your actual repo link)*

---

## d. Models Used & Performance Evaluation Comparison Table

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **Logistic Regression** | 0.9825 | 0.9954 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |
| **Decision Tree** | 0.9123 | 0.9157 | 0.9559 | 0.9028 | 0.9286 | 0.8174 |
| **KNN** | 0.9561 | 0.9788 | 0.9589 | 0.9722 | 0.9655 | 0.9054 |
| **Naive Bayes** | 0.9386 | 0.9878 | 0.9452 | 0.9583 | 0.9517 | 0.8676 |
| **Support Vector Machine** | 0.9825 | 0.9950 | 0.9861 | 0.9861 | 0.9861 | 0.9623 |

---

## e. Observations on Model Performance

| ML Model Name | Observation about model performance |
| :--- | :--- |
| **Logistic Regression** | Performed exceptionally well on the feature-scaled dataset. Achieved top accuracy (98.25%) and highest AUC (0.9954) due to clear linear separability in standardized feature space. |
| **Decision Tree** | Exhibited lower generalizability (Accuracy 91.23%) compared to distance-based and probabilistic models, likely suffering from slight over-fitting on individual feature splits without ensemble tree bagging/boosting. |
| **KNN** | Achieved strong predictive capability (Accuracy 95.61%, MCC 0.9054) after feature standardization, demonstrating that local neighborhood metrics hold high discriminative power for this domain. |
| **Naive Bayes** | Showed robust classification capabilities (Accuracy 93.86%, AUC 0.9878), performing reliably despite its conditional independence assumption across all 30 features. |
| **Support Vector Machine** | Tied for overall best performance (Accuracy 98.25%, F1 0.9861, MCC 0.9623). The RBF kernel mapped feature interactions into a hyperspace with maximum geometric margin separation. |
| **Overall Winner for Dataset** | **Logistic Regression & Support Vector Machine (Tie)** — Both models achieved an accuracy of 0.9825, F1 score of 0.9861, and MCC of 0.9623. Logistic Regression had the highest overall AUC (0.9954). |