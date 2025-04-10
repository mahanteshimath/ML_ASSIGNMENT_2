import streamlit as st

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
  page_title="iitj-ml-learning",
  page_icon=":material/cognition:",
  layout="wide",
  initial_sidebar_state="expanded",
) 


pg1 = st.Page(
    "pages/Theory.py",
    title="Theory",
    icon=":material/cognition:",
    default=True,
)

pg2 = st.Page(
    "pages/Student_Performance_Predictor.py",
    title="Student Performance Predictor",
    icon=":material/cognition:"
)

pg3 = st.Page(
    "pages/Rupee_Price_Prediction.py",
    title="Rupee Price Prediction",
    icon=":material/currency_rupee:"
)

pg = st.navigation(
    {
        "Info": [pg1],
        "ML Practicals": [pg2, pg3],
    }
)


pg.run()