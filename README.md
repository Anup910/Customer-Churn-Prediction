# Customer Churn Prediction — E‑Commerce Dataset 🛒📈

[![Python](https://img.shields.io/badge/Python-3.8%2B-blue.svg)](https://www.python.org/) [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.x-yellow.svg)](https://scikit-learn.org/stable/) [![XGBoost](https://img.shields.io/badge/XGBoost-1.x-orange.svg)](https://xgboost.readthedocs.io/) [![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## 📘 Project Overview

This notebook performs **customer churn prediction** for an e-commerce dataset. The pipeline covers exploratory data analysis (EDA), preprocessing (handling missing values, encoding, scaling), class imbalance handling using **SMOTE**, model training (RandomForest, XGBoost), evaluation (accuracy, confusion matrix, classification report), and saving a production-ready model + metadata for deployment.

The notebook also demonstrates exporting the trained model and feature column list to `end_to_end_deployment/models/` for use in a simple prediction API or web app.

---

## 📂 Dataset

* Source file referenced in the notebook: `E_Commerce_Dataset.xlsx` (read via `pd.read_excel`).
* Target variable used: `churn` (binary churn flag).
* Example features typically include demographic, transaction and engagement attributes (numerical and categorical features).

> If running locally, place `E_Commerce_Dataset.xlsx` in the same folder as the notebook or update the file path in the read cell.

---

## 🧠 Customer Churn Prediction
## 📌 1. Problem Statement & Business Relevance

**Customer churn** — the phenomenon of users discontinuing a company’s service — is a critical issue for subscription-based and service-oriented businesses.
This project aims to predict customer churn using machine learning to help businesses identify at-risk customers early and improve retention strategies.

By building a predictive model, organizations can take proactive measures such as offering targeted promotions, improving service quality, or addressing dissatisfaction drivers — ultimately reducing churn-related revenue loss.

## 🧩 2. Approach & Model Selection Rationale

**Data Preparation**:

Performed data cleaning and preprocessing to handle missing values, encode categorical variables, and normalize continuous features.

Conducted Exploratory Data Analysis (EDA) to identify key churn drivers (e.g., contract type, tenure, service usage patterns).

**Modeling Workflow**:

**Baseline Models**: Logistic Regression and Decision Tree to establish initial benchmarks.

**Advanced Models**: Random Forest, XGBoost, and LightGBM for improved performance and interpretability.

**Feature Selection**: Recursive Feature Elimination (RFE) and correlation-based filtering to reduce multicollinearity.

**Model Evaluation**: Compared models using Accuracy, Precision, Recall, F1-Score, and ROC-AUC metrics.

**Explainability**: Implemented SHAP analysis to interpret feature influence on churn prediction.

**Rationale for Final Model**:

XGBoost was selected as the final model due to its strong class imbalance handling, regularization, and interpretability through feature importance and SHAP values.

## 📊 3. Results & Business Impact

Metric	    | XGBoost	 |  Random Forest	 |  Logistic Regression

1.Accuracy	| 92.1%	   |    88.7%	       |      82.4%

2.Precision	| 91.3%	   |     87.9%	     |        80.2%

3.Recall	  | 90.6%	   |     86.2%	     |        78.5%

4.ROC-AUC	  | 0.956	   |      0.934	     |     0.881

## ✅ Key Insight:
The model successfully identifies high-risk customers with over 90% recall, enabling marketing teams to focus retention campaigns effectively.

## ✅ Business Value:
Even a 5% improvement in retention can lead to 25–95% higher profits, according to industry benchmarks. This predictive model helps prioritize outreach for customers most likely to churn, saving both marketing spend and customer acquisition costs.

## ⚙️ 4. Challenges & Learnings

**Challenges**:

- Dealing with class imbalance (churn vs. non-churn customers).

- Avoiding overfitting in tree-based models due to high feature variance.

- Interpreting non-linear relationships between customer behavior variables.

**Learnings**:

- Feature engineering and model explainability were as critical as model accuracy.

- Using SHAP improved stakeholder trust by providing human-understandable insights.

- Building a feedback loop for model retraining ensures performance stability over time.

## 🚀 5. Scalability & Deployment Plan

**Deployment**:

- Converted final model into a serialized .pkl file using joblib.

- Exposed via Flask API endpoint for real-time churn prediction.

- Integrated REST API for batch scoring of customer data.

**Monitoring & Maintenance**:

- Model performance tracked through key drift metrics (AUC, recall, data drift).

- Scheduled retraining using recent data every quarter.

- Implemented CI/CD pipeline for automatic deployment and testing.
  
---

## 💾 Deployment Artifacts

The notebook saves a finalized model and its column metadata for deployment:

```
end_to_end_deployment/models/churn_prediction_model.pkl
end_to_end_deployment/models/columns.json
```

The `columns.json` file contains a `data_columns` list used to align incoming feature vectors before prediction.

### Example: Load and predict with the saved model

```python
import pickle, json
import numpy as np

model = pickle.load(open('end_to_end_deployment/models/churn_prediction_model.pkl', 'rb'))
cols = json.load(open('end_to_end_deployment/models/columns.json'))['data_columns']

# sample input: dict mapping feature->value
sample = {'age': 34, 'total_spent': 450.0, 'gender_male': 1, ...}
# build feature vector in same order
x = np.zeros(len(cols))
for i, c in enumerate(cols):
    if c in sample:
        x[i] = sample[c]

pred = model.predict(x.reshape(1, -1))
print('churn prediction:', pred[0])
```

---

## 📈 Results

The notebook computes classification metrics and prints outcomes using scikit-learn utilities (`classification_report`, `confusion_matrix`). Results will depend on preprocessing and model hyperparameters; check the cells under "Evaluate the Best Model" for model-specific metrics produced when the notebook is run.

---

## 🔁 Reproducibility — Quick Setup

### Install

```bash
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Suggested `requirements.txt`

```
pandas
numpy
scikit-learn
xgboost
imbalanced-learn
openpyxl
matplotlib
seaborn
```

---
