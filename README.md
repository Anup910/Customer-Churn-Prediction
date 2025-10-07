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

## 🔍 Exploratory Data Analysis (EDA)

The notebook performs standard EDA including:

* Count plots for categorical columns and histograms for numeric columns.
* Churn distribution visualizations across categories.
* Missing value inspection and imputation (numeric columns filled with mean).
* Basic correlation checks and visual diagnostics.

---

## ⚙️ Preprocessing & Class Balancing

Preprocessing steps implemented in the notebook:

* Missing numeric values filled with column mean.
* Categorical variables converted to numeric via one-hot encoding (`pd.get_dummies`).
* Feature scaling where appropriate (StandardScaler used in pipeline snippets).
* **SMOTE** (from `imblearn`) applied to training data to handle class imbalance.
* Train/test split for evaluation.

---

## 🤖 Models & Training

Models instantiated and evaluated in the notebook include:

* **Random Forest Classifier**
* **XGBoost (XGBClassifier)** — used as the final model in the notebook (saved to disk)

Evaluation uses `accuracy_score`, `classification_report`, and `confusion_matrix` to inspect performance and class-level precision/recall.

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
