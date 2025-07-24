import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score

st.title("🩺 Hypertension Risk Predictor")
st.write("Upload patient health details to predict hypertension risk.")

# Load data
@st.cache_data
def load_data():
    return pd.read_csv("hypertension_dataset.csv")

data = load_data()
st.subheader("Dataset Preview")
st.dataframe(data.head())

# Preprocess
st.subheader("Model Training")

target = 'Has_Hypertension'
X = data.drop(columns=[target])
y = data[target]

# Encode if categorical
X = pd.get_dummies(X)

# Train/Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train Model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Accuracy
acc = accuracy_score(y_test, model.predict(X_test))
st.write(f"Model Accuracy: **{acc:.2f}**")

# Predict on User Input
st.subheader("📋 Predict Hypertension Risk")
user_input = {}

for col in X.columns:
    if 'age' in col.lower():
        user_input[col] = st.slider(col, 0, 100, 25)
    elif 'weight' in col.lower() or 'height' in col.lower():
        user_input[col] = st.slider(col, 30, 150, 60)
    else:
        user_input[col] = st.number_input(col, min_value=0.0, max_value=500.0, value=1.0)

input_df = pd.DataFrame([user_input])
prediction = model.predict(input_df)[0]
st.write("### Prediction:")
st.success("🟢 No Hypertension Risk") if prediction == 0 else st.error("🔴 High Risk of Hypertension")

# Feature Importance
st.subheader("🔍 Feature Importance")
importances = pd.Series(model.feature_importances_, index=X.columns)
st.bar_chart(importances.sort_values(ascending=False))

