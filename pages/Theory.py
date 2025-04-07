import streamlit as st
st.markdown("# :blue[ Authors: Srijit Ghatak(G24AIT2079) and Mahantesh Hiremath(G24AIT2178)]  :speech_balloon:")
st.divider()
st.markdown("""
# Student Performance Predictor working
### 🧠 **1. Model Purpose**

The goal of this Streamlit app is to **predict a student's "Performance Index"** based on multiple factors such as study hours, sleep, previous scores, and other activities using a **machine learning model**.

### 🔍 **2. Data Used**

The app loads a dataset containing various student attributes and their corresponding performance scores. The dataset includes features like:
- **Hours Studied**
- **Previous Scores**
- **Extracurricular Activities**
- **Sleep Hours**
- **Sample Question Papers Practiced**

These are used to predict the **Performance Index**, which is the target variable.

### 🏗️ **3. Model Architecture**

The app uses a **machine learning pipeline** that includes:
- **Preprocessing** of numerical and categorical data
- A **Random Forest Regressor**, which is an ensemble of decision trees

### 🔄 **4. How Prediction Happens (Step-by-Step)**

Here’s what happens behind the scenes when a user inputs values and clicks **"Submit for Predictions"**:

#### **Step 1: Capture User Input**
The app collects the inputs from the user (like hours studied, sleep hours, etc.) through a form.

#### **Step 2: Convert to DataFrame**
The inputs are transformed into a structured format (a DataFrame) so the model can understand them.

#### **Step 3: Preprocessing**
- **Numerical features** (like hours and scores) are scaled using **StandardScaler**.
- **Categorical features** (like extracurricular activities) are encoded using **OneHotEncoder**.

This ensures the input format matches what the model saw during training.

#### **Step 4: Prediction**
The preprocessed data is passed to the **Random Forest Regressor**, which:
- Aggregates predictions from multiple decision trees
- Outputs a final **Performance Index score**

The result is a **continuous numeric prediction** representing how well the student is expected to perform.



### 📈 **5. Model Evaluation**

To show how well the model is performing, it also displays:
- **Root Mean Squared Error (RMSE)**: Lower is better; shows how far predictions are from actual values.
- **R² Score**: Ranges from 0 to 1; closer to 1 means better predictive power.



### 🎯 Summary

- The app **takes user inputs**, **preprocesses them**, and uses a trained **Random Forest model** to **predict student performance**.
- It also provides **transparency** in the prediction process by breaking it down step-by-step and showing **model metrics** to the user.
- The use of **caching, session state**, and **delayed feedback** (via `time.sleep`) makes the UX smooth and interactive.
""")
st.divider()





















st.markdown(
    '''
    <style>
    .streamlit-expanderHeader {
        background-color: blue;
        color: white; # Adjust this for expander header color
    }
    .streamlit-expanderContent {
        background-color: blue;
        color: white; # Expander content color
    }
    </style>
    ''',
    unsafe_allow_html=True
)

footer="""<style>

.footer {
position: fixed;
left: 0;
bottom: 0;
width: 100%;
background-color: #2C1E5B;
color: white;
text-align: center;
}
</style>
<div class="footer">
<p>Developed with ❤️ by <a style='display: inline; text-align: center;' href="https://iitj-ml-learnings.streamlit.app/" target="_blank">Srijit and Mahantesh</a></p>
</div>
"""
st.markdown(footer,unsafe_allow_html=True)  