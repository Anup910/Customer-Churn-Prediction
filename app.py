# -*- coding: utf-8 -*-
"""app.ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1MdypSvOKEzG6jcLq0h6EZJaYbMkqb2l7
"""

pip install streamlit pandas numpy scikit-learn xgboost matplotlib seaborn imbalanced-learn

import streamlit as st
import pickle
import pandas as pd
import numpy as np

# Load the trained model
model_path = '/content/churn_prediction_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load feature columns
with open("/content/columns.json", "r") as f:
    columns = pd.read_json(f)["data_columns"]

st.title("Customer Churn Prediction")

st.write("Enter customer details below to predict churn probability:")

# Input fields for user data
inputs = {}
for col in columns:
    inputs[col] = st.number_input(f"Enter {col}", value=0.0)

# Convert inputs to DataFrame
input_df = pd.DataFrame([inputs])

# Predict button
if st.button("Predict"):
    prediction = model.predict(input_df)
    st.write(f"Churn Prediction: {'Churn' if prediction[0] == 1 else 'Not Churn'}")

!streamlit run app.py
