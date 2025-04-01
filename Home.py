import streamlit as st
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas
from pathlib import Path
import time
import pandas as pd
from PIL import Image
from io import BytesIO
import requests 

st.logo(
    image="https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg",
    link="https://www.linkedin.com/in/mahantesh-hiremath/",
    icon_image="https://upload.wikimedia.org/wikipedia/en/4/41/Flag_of_India.svg"
)

st.set_page_config(
  page_title="my_ml_learrning",
  page_icon="🔍",
  layout="wide",
  initial_sidebar_state="expanded",
) 

# --- Info ---

pg1 = st.Page(
    "pages/Theory.py",
    title="Theory",
    icon=":material/cognition:",
    default=True,
)

pg2 = st.Page(
    "pages/Student_Performance_Predictor.py",
    title="Student Performance Predictor",
    icon=":material/brain:"
)

pg = st.navigation(
    {
        "Info": [pg1],
        "ML Practicals": [pg2],
    }
)


pg.run()