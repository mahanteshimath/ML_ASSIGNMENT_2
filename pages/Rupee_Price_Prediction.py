import streamlit as st
import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt


st.title("💸 USD to INR Exchange Rate Forecast (Till 2050)")

# 🔁 Cached data loading from GitHub
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mahanteshimath/ML_ASSIGNMENT_2/refs/heads/main/data/HistoricalData_Currency.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop(columns=['Volume']).dropna()
    return df

# Load data
df = load_data()

# Preview data
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head(), use_container_width=True)

# Rename columns for Prophet
# Verify column names before renaming
st.write("Columns in dataset:", df.columns.tolist())
df = df.rename(columns={'Date': 'ds', 'Close*': 'y'})  # Adjust 'Close*' to match the actual column name
df = df[['ds', 'y']]

# Sidebar for forecast years
st.sidebar.header("🔧 Forecast Settings")
forecast_years = st.sidebar.slider(
    "Select number of years to forecast:",
    min_value=1,
    max_value=30,
    value=27,
    help="Forecast horizon from 1 to 30 years"
)
forecast_days = forecast_years * 365
last_year = df['ds'].dt.year.max()

# Train Prophet
st.subheader("⚙️ Training Prophet Model...")
model = Prophet()
model.fit(df)

# Forecast
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# Forecast plot
st.subheader(f"📈 Forecasted USD to INR till {last_year + forecast_years}")
fig1 = model.plot(forecast)
st.pyplot(fig1)

# Components
with st.expander("🔍 Show Forecast Components (Trend, Weekly, Yearly Seasonality)"):
    fig2 = model.plot_components(forecast)
    st.pyplot(fig2)

# Forecast table
st.subheader("📄 Forecast Table (Last 30 Days)")
forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
st.dataframe(forecast_display, use_container_width=True)

# Download CSV
st.subheader("⬇️ Download Forecast CSV")
st.download_button(
    label="Download Full Forecast as CSV",
    data=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(index=False),
    file_name="USD_INR_Forecast.csv",
    mime="text/csv"
)
