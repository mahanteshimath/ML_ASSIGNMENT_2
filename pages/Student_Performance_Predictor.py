import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_squared_error 
import time  # Added import for sleep


# Load and preprocess data
@st.cache_data
def load_data():
    # Update the file path to the correct location
    df = pd.read_csv("https://raw.githubusercontent.com/mahanteshimath/ML_ASSIGNMENT_2/refs/heads/main/data/Student_Performance.csv")
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

st.markdown('<h1 class="title">Student Performance Predictor</h1>', unsafe_allow_html=True)

# Input section in a form
with st.container():
    with st.form("student_input_form"):
        st.write("Enter student details:")
        user_inputs = {}
        for feature in features:
            if feature == "Extracurricular Activities":
                user_inputs[feature] = st.selectbox(
                    f"{feature.replace('_', ' ').title()}",
                    ["Yes", "No"],
                    key=f"{feature}_input",
                )
            else:
                min_val = float(df[feature].min())
                max_val = float(df[feature].max())
                step = 1.0 if feature != "Sample Question Papers Practiced" else 0.5
                user_inputs[feature] = st.number_input(
                    f"{feature.replace('_', ' ').title()}",
                    min_value=min_val,
                    max_value=max_val,
                    value=st.session_state.get(f"{feature}_value", (min_val + max_val) / 2),
                    step=step,
                    key=f"{feature}_input",
                )
        # Submit button for the form
        submitted = st.form_submit_button("Submit of Predictions")

# Prediction section
if submitted:
    with st.container():
        try:
            with st.spinner("Processing inputs and making predictions..."):
                # Step 1: Store the input values in session state
                st.write("Step 1: Storing input values...")
                for feature in features:
                    st.session_state[f"{feature}_value"] = user_inputs[feature]
                time.sleep(1)  # Simulate delay

                # Step 2: Convert user inputs to a DataFrame
                st.write("Step 2: Converting inputs to DataFrame...")
                input_df = pd.DataFrame([user_inputs])
                time.sleep(1)  # Simulate delay

                # Step 3: Ensure numeric columns are properly converted
                st.write("Step 3: Validating numeric inputs...")
                for col in numeric_features:
                    if col in input_df.columns:
                        input_df[col] = pd.to_numeric(input_df[col], errors='coerce')
                time.sleep(1)  # Simulate delay

                # Step 4: Ensure categorical columns match the expected format
                st.write("Step 4: Validating categorical inputs...")
                for col in categorical_features:
                    if col in input_df.columns:
                        input_df[col] = input_df[col].astype(str)
                time.sleep(1)  # Simulate delay

                # Step 5: Apply preprocessing
                st.write("Step 5: Applying preprocessing...")
                processed_input = preprocessor.transform(input_df)
                time.sleep(1)  # Simulate delay

                # Step 6: Make prediction
                st.write("Step 6: Making prediction...")
                prediction = model.named_steps["regressor"].predict(processed_input)[0]
                time.sleep(1)  # Simulate delay

                # Display results
                st.success("Prediction completed successfully!")
                st.markdown(
                    f'<div class="result">Predicted Performance Index: {prediction:.2f}</div>',
                    unsafe_allow_html=True,
                )
                st.balloons()
        except Exception as e:
            st.error(f"An error occurred during prediction: {e}")

# Show model performance with additional metrics
with st.container():
    st.write("### Model Performance Metrics:")
    rmse = np.sqrt(mean_squared_error(y_test, model.predict(X_test)))
    r2_score = model.score(X_test, y_test)
    st.metric(label="Root Mean Squared Error (RMSE)", value=f"{rmse:.2f}")
    st.metric(label="R² Score", value=f"{r2_score:.2f}")