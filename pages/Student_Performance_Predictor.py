import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Load and preprocess data
@st.cache_data
def load_data():

    df = pd.read_csv("data\Student_Performance.csv")
    return df

df = load_data()


# Define features and target
features = [
    "Hours Studied",
    "Previous Scores",
    "Extracurricular Activities",
    "Sleep Hours",
    "Sample Question Papers Practiced",
]
target = "Performance Index"

# Split data
X = df[features]
y = df[target]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Preprocessing pipelines
numeric_features = X.select_dtypes(include=["int64", "float64"]).columns
categorical_features = ["Extracurricular Activities"]

numeric_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown="ignore")

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)

# Train model
model = Pipeline(
    [
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=100, random_state=42)),
    ]
)
model.fit(X_train, y_train)

# Add custom CSS styling
st.markdown(
    """
    <style>
    /* General styling */
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f5f5f5;
    }

    /* Title styling */
    .title {
        color: #2c6cc4;
        text-align: center;
        padding: 2rem 0;
        font-size: 2.5rem;
    }

    /* Input containers */
    .stTextInput, .stNumberInput {
        background-color: #ffffff;
        border: 1px solid #e0e0e0;
        border-radius: 8px;
        padding: 1rem;
        margin: 0.5rem 0;
    }

    /* Buttons */
    .stButton button {
        background-color: #2c6cc4;
        color: white;
        border-radius: 8px;
        padding: 1rem 2rem;
        font-size: 1.1rem;
    }
    .stButton button:hover {
        background-color: #1a4795;
    }

    /* Result display */
    .result {
        background-color: #e8f5e9;
        border-radius: 8px;
        padding: 1.5rem;
        margin: 2rem 0;
        text-align: center;
        font-size: 1.5rem;
        color: #2d6a4f;
    }

    /* Container padding */
    .stApp {
        max-width: 800px;
        margin: 0 auto;
        padding: 2rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Streamlit App
# Title
st.markdown('<h1 class="title">Student Performance Predictor</h1>', unsafe_allow_html=True)

# User input section
st.write("Enter student details:")
user_inputs = {}
for feature in features:
    if feature == "Extracurricular Activities":
        user_inputs[feature] = st.selectbox(
            f"{feature.replace('_', ' ').title()}",
            ["Yes", "No"],
            key=feature,
        )
    else:
        min_val = df[feature].min()
        max_val = df[feature].max()
        user_inputs[feature] = st.number_input(
            f"{feature.replace('_', ' ').title()}",
            min_value=min_val,
            max_value=max_val,
            value=(min_val + max_val) // 2,
            step=1 if feature != "Sample Question Papers Practiced" else 0.5,
            key=feature,
        )

# Prediction button
if st.button("Predict Performance"):
    input_df = pd.DataFrame([user_inputs])
    prediction = model.predict(input_df)[0]
    st.markdown(
        f'<div class="result">Predicted Performance Index: {prediction:.2f}</div>',
        unsafe_allow_html=True,
    )

# Show model performance
st.write("Model Performance (RMSE):", np.sqrt(mean_squared_error(y_test, model.predict(X_test))))