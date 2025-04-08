import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import numpy as np

# Try importing SARIMA related packages with error handling
try:
    from pmdarima import auto_arima
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    SARIMA_AVAILABLE = True
except ImportError:
    SARIMA_AVAILABLE = False
    st.warning("SARIMA functionality is not available. Only Prophet model will be used.")


# Custom styled title
st.markdown(
    """
    <div style='text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px;'>
        <h1 style='color: #2c7be5; margin: 0;'>💸 USD to INR Exchange Rate Forecast (Till 2050)</h1>
        <p style='color: #333; font-size: 18px; margin-top: 10px;'>Compare Prophet & SARIMA Models</p>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("------")

# Enhanced data loading with error handling
@st.cache_data
def load_data():
    try:
        url = "https://raw.githubusercontent.com/mahanteshimath/ML_ASSIGNMENT_2/refs/heads/main/data/HistoricalData_Currency.csv"
        df = pd.read_csv(url)
        df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
        df = df.sort_values('Date').reset_index(drop=True)
        
        # Save original data for raw preview
        raw_df = df.copy()
        
        if 'Volume' in df.columns:
            df = df.drop(columns=['Volume'])
        
        # Process data for time series models
        df = df[['Date', 'Close/Last']].dropna()
        df.columns = ['ds', 'y']
        
        return df, raw_df  # Return both processed and original data
    
    except Exception as e:
        st.error(f"❌ Data loading failed: {str(e)}")
        st.stop()

# Load data
processed_df, raw_df = load_data()

# Raw Data Preview with all original columns
st.subheader("📊 Raw Data Preview (All Columns)")
st.dataframe(
    raw_df.head(10),
    use_container_width=True,
    height=300
)

st.write(f"📅 Data spans **{len(raw_df)}** records from {raw_df['Date'].min().date()} to {raw_df['Date'].max().date()}")

st.divider()

# Forecast settings with dynamic max
last_year = processed_df['ds'].dt.year.max()
max_forecast = 2050 - last_year

st.sidebar.header("🔧 Forecast Settings")
forecast_years = st.sidebar.slider(
    "Forecast Horizon (Years):",
    min_value=1,
    max_value=max_forecast,
    value=27,
    help="Max set to reach 2050"
)

forecast_days = forecast_years * 365
end_year = last_year + forecast_years

# Model training with progress indicator and explanation
with st.spinner("Training Prophet Model..."):
    model = Prophet(yearly_seasonality=True, weekly_seasonality=True)
    model.fit(processed_df)
    
st.success("✅ Model trained successfully")

# Add model explanation
with st.expander("🔍 How does the model work?", expanded=False):
    st.write(
        """
        **Prophet Forecasting Methodology:**
        
        1. **Model Structure**: Additive regression model combining:
           - **Trend**: Piecewise linear/logistic growth curve
           - **Seasonality**: Weekly/yearly patterns using Fourier series
           - **Holiday Effects**: Special event impacts (not used here)
           
        2. **Training Process**: 
           - Learns patterns from historical data
           - Automatically detects changepoints in the trend
           - Optimizes parameters using MAP estimation
           
        3. **Prediction**: 
           - Extends learned patterns into the future
           - Provides uncertainty intervals using Monte Carlo sampling
        """
    )

st.divider()

# Enhanced forecast visualization for Prophet
future = model.make_future_dataframe(periods=forecast_days)
forecast = model.predict(future)

st.subheader(f"📈 Prophet Model Forecast ({forecast_years}-year projection till {end_year})")

# Primary forecast plot with hover
fig1 = plot_plotly(model, forecast)
fig1.update_traces(
    hovertemplate="<br>".join([
        "Date: %{x|%Y-%m-%d}",
        "Predicted Rate: %{y:.2f} INR"
    ])
)
fig1.update_layout(
    title_text=f"USD/INR Exchange Rate Forecast till {end_year}",
    xaxis_title="Date",
    yaxis_title="Exchange Rate (USD to INR)",
    hovermode="x unified"
)
st.plotly_chart(fig1, use_container_width=True)

# Components analysis
st.subheader("🔍 Prophet Model Components Analysis")
fig2 = plot_components_plotly(model, forecast)
fig2.update_layout(
    title_text="Breakdown of Forecast Components",
    hovermode="x unified"
)
st.plotly_chart(fig2, use_container_width=True)

st.divider()

# SARIMA Model Section
if SARIMA_AVAILABLE:
    st.header("Alternative Model: SARIMA Forecast")
    with st.spinner("Training SARIMA Model..."):
        train_data = processed_df.set_index('ds')['y']
        
        # Auto ARIMA
        stepwise_model = auto_arima(
            train_data,
            start_p=1, start_q=1,
            max_p=5, max_q=5,
            m=7,
            seasonal=True,
            stepwise=True,
            suppress_warnings=True,
            error_action='ignore',
            trace=False
        )
        
        sarima_model = SARIMAX(
            train_data,
            order=stepwise_model.order,
            seasonal_order=stepwise_model.seasonal_order
        )
        results = sarima_model.fit(disp=False)
        st.success(f"✅ SARIMA model trained successfully")

    # SARIMA forecast visualization
    forecast_sarima = results.get_forecast(steps=forecast_days)
    pred_ci = forecast_sarima.conf_int()
    start_date = processed_df['ds'].max() + pd.Timedelta(days=1)
    forecast_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')

    fig_sarima = plt.figure(figsize=(12, 6))
    plt.plot(train_data.index, train_data.values, label='Historical Data', alpha=0.5)
    plt.plot(forecast_dates, forecast_sarima.predicted_mean, label='SARIMA Forecast')
    plt.fill_between(
        forecast_dates,
        pred_ci.iloc[:, 0],
        pred_ci.iloc[:, 1],
        color='pink',
        alpha=0.3
    )
    plt.title(f"SARIMA Forecast till {end_year}")
    plt.xlabel("Date")
    plt.ylabel("Exchange Rate (USD to INR)")
    plt.legend()
    st.pyplot(fig_sarima)
else:
    st.info("SARIMA model is not available in this deployment. Using only Prophet model for forecasting.")

# Model Comparison
st.header("📊 Model Comparison")
col1, col2 = st.columns(2)
with col1:
    rmse_prophet = np.sqrt(np.mean((forecast['yhat'].values[:len(processed_df)] - processed_df['y'].values) ** 2))
    st.metric("Prophet RMSE", f"{rmse_prophet:.2f}")
with col2:
    if SARIMA_AVAILABLE:
        rmse_sarima = np.sqrt(np.mean((results.fittedvalues - train_data) ** 2))
        st.metric("SARIMA RMSE", f"{rmse_sarima:.2f}")

# Model limitations disclaimer
st.warning(
    "⚠️ Both models have their strengths: Prophet handles trends and seasonality well, " +
    "while SARIMA captures short-term patterns. Consider both forecasts for decision-making."
)


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