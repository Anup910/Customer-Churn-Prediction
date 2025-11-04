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

# Flight Delay Analysis & Prediction — Data Science Case Study

**Goal:** Predict flight delays (flight status) and surface features that drive delays so airlines or airports can proactively mitigate risk and improve punctuality.

---

## 🧩 1. Problem Statement & Business Relevance

**Problem:** Predict whether a scheduled flight will be delayed (or late) using historical flight metadata, times, and operational features. Accurate delay predictions allow airlines and airports to proactively manage crews, gates, and passenger communications, reducing passenger dissatisfaction, connection misses, and downstream operational costs.

**Why it matters:** Even modest improvements in delay prediction reduce passenger rebooking costs, crew disruptions, and ripple delays across networks. Operationally actionable predictions help prioritize interventions (re-routing, buffer times, crew reassignments).

---

## ⚙️ 2. Approach & Model Selection Rationale

**Dataset & sample sizes (from notebook)**

- Full loaded dataset: **235,640 rows × 32 columns** (raw sample file `flights_sample_3m.csv`).
  
- Model training/evaluation used a sampled/processed subset (example model test split shown: **10,767** test rows in final evaluation outputs).

**Target & labels**

- Target column: **`FLIGHT_STATUS`** (binary: on-time vs delayed/late). The notebook filters out cancelled/diverted flights prior to modeling.

**Key preprocessing / feature engineering**

- Convert `FL_DATE` to `datetime`, extract `Year`, `Month`, `Day`, `DayOfWeek`, `Quarter`, `Season`.
    
- Remove cancelled and diverted flights from dataset.
    
- Compute and retain operational time features: `ARR_DELAY`, `DEP_DELAY`, `ELAPSED_TIME_DIFF`, `WHEELS_OFF_elapse`, `WHEELS_ON_elapse`, `TAXI_OUT`, `TAXI_IN`, and `Elapsed time` differences.
    
- One-hot / integer encode airlines and categorical identifiers; drop extraneous columns (`FL_DATE`, `AIRLINE`, `FL_NUMBER`, `ORIGIN_CITY`, `DEST_CITY`, `Quarter`, etc.) to avoid leakage or high-cardinality noise.
    
- Handle missing values; scale numeric features where needed.

**Sampling and imbalance**

- The notebook demonstrates use of **SMOTE** to balance training data before training selected models.

**Models evaluated**

- Baselines and ensembles: **Logistic Regression**, **K-Nearest Neighbors (KNN)**, **Random Forest**, **Gradient Boosting (GB)**, and others (KNN used as example in the notebook).
    
- Evaluation uses classification reports and confusion matrices; the notebook also calculates feature correlations with the target.

**Why these models?**

- For tabular operational data, tree ensembles (Random Forest / Gradient Boosting) typically capture complex interactions and non-linearities among schedule/timing features. KNN and logistic regression serve as interpretable baselines. SMOTE is used to help models learn minority-class patterns when class imbalance exists.
  
---

## 📊 3. Results & Business Impact (Metrics)

**Dataset & test-frame numbers (from notebook outputs)**

- Full file: **235,640 rows × 32 cols**.
    
- Example final evaluation frame: **Test = 10,767 rows** (used for classification reports in notebook outputs).

**Representative model output (example classification report and confusion matrix)**

- Classification report (sample shown in notebook; weighted results):
    
  - Precision / Recall / F1 are strong (weighted metrics ~0.98 in the printed report).
      
- Confusion matrix (printed example):  

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
