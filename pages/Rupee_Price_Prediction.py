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
    if 'Volume' in df.columns:
        df = df.drop(columns=['Volume'])
    df = df.dropna()
    return df

# Load data
df = load_data()

# Preview data
st.subheader("📊 Raw Data Preview")
st.dataframe(df.head(), use_container_width=True)
st.divider()

# Rename columns for Prophet
if 'Close/Last' in df.columns:
    df = df.rename(columns={'Date': 'ds', 'Close/Last': 'y'})
else:
    st.error("The dataset does not contain the expected 'Close/Last' column.")
    st.stop()
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

st.divider()

# Train Prophet
st.subheader("⚙️ Training Prophet Model...")
model = Prophet()
model.fit(df)

st.divider()

# Forecast
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

# Forecast plot
st.subheader(f"📈 Forecasted USD to INR till {last_year + forecast_years}")
fig1 = model.plot(forecast)
ax1 = fig1.gca()
ax1.set_xlabel("Date")
ax1.set_ylabel("Exchange Rate (USD to INR)")
st.pyplot(fig1)

st.divider()

# Components
with st.expander("🔍 Show Forecast Components (Trend, Weekly, Yearly Seasonality)"):
    fig2 = model.plot_components(forecast)
    for ax in fig2.axes:
        ax.set_xlabel("Date")
        ax.set_ylabel("Value")
    st.pyplot(fig2)

st.divider()

# Forecast table
st.subheader("📄 Forecast Table (Last 30 Days of Predictions)")
forecast_display = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(30)
forecast_display = forecast_display.rename(columns={
    'ds': 'Date',
    'yhat': 'Predicted Exchange Rate',
    'yhat_lower': 'Lower Bound',
    'yhat_upper': 'Upper Bound'
})
st.dataframe(forecast_display, use_container_width=True)

st.divider()

# Download CSV
st.subheader("⬇️ Download Forecast CSV")
st.download_button(
    label="Download Full Forecast as CSV",
    data=forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].rename(columns={
        'ds': 'Date',
        'yhat': 'Predicted Exchange Rate',
        'yhat_lower': 'Lower Bound',
        'yhat_upper': 'Upper Bound'
    }).to_csv(index=False),
    file_name="USD_INR_Forecast.csv",
    mime="text/csv"
)
