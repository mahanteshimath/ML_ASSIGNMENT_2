import streamlit as st
import pandas as pd
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import matplotlib.pyplot as plt
import numpy as np

# Careful SARIMA imports with detailed error handling
try:
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    from statsmodels.tsa.arima.model import SARIMA
    SARIMA_AVAILABLE = True
except ImportError as e:
    SARIMA_AVAILABLE = False
    st.warning("SARIMA functionality is not available. Using only Prophet model.")

# Custom styled title with updated text
st.markdown(
    """
    <div style='text-align: center; padding: 20px; background-color: #f0f8ff; border-radius: 10px;'>
        <h1 style='color: #2c7be5; margin: 0;'>💸 USD to INR Exchange Rate Forecast (Till 2050)</h1>
        <p style='color: #333; font-size: 18px; margin-top: 10px;'>Powered by Prophet Time Series Forecasting</p>
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
st.header(" Prophet Model Forecast")
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
    
st.success("✅ Prophet Model trained successfully")
st.toast("✅ Prophet Model trained successfully", icon='🎉')
st.balloons()

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

# After Prophet visualization section, before SARIMA section
st.subheader("🔍 Prophet Forecast Preview")
# Prepare Prophet forecast data for display
prophet_preview = pd.DataFrame({
    'Date': forecast['ds'],
    'Predicted Rate': forecast['yhat'].round(2),
    'Lower Bound': forecast['yhat_lower'].round(2),
    'Upper Bound': forecast['yhat_upper'].round(2)
})
prophet_preview.set_index('Date', inplace=True)

# Display Prophet preview
st.dataframe(
    prophet_preview.tail(10),
    use_container_width=True,
    height=200
)

# Download button for Prophet forecast
csv_prophet = prophet_preview.to_csv()
st.download_button(
    label="📥 Download Prophet Forecast",
    data=csv_prophet,
    file_name="prophet_forecast.csv",
    mime="text/csv"
)

st.divider()

# SARIMA Model Section
if SARIMA_AVAILABLE:
    st.header("Alternative Model: SARIMA Forecast")
    with st.spinner("Training SARIMA Model...this may take a while"):
        try:
            # Prepare data for SARIMA
            train_data = processed_df.set_index('ds')['y']
            
            # Simple ARIMA model with reasonable parameters
            sarima_model = SARIMA(
                train_data,
                order=(2, 1, 2),
                seasonal_order=(1, 1, 1, 12)
            )
            results = sarima_model.fit()
            st.success("✅ SARIMA model trained successfully")
            st.toast("✅SARIMA model trained successfully", icon='🎉')
            st.balloons()

            # SARIMA forecast visualization
            forecast_sarima = results.forecast(steps=forecast_days)
            start_date = processed_df['ds'].max() + pd.Timedelta(days=1)
            forecast_dates = pd.date_range(start=start_date, periods=forecast_days, freq='D')

            # Create SARIMA plot
            fig_sarima = plt.figure(figsize=(12, 6))
            plt.plot(train_data.index, train_data.values, label='Historical Data', alpha=0.5)
            plt.plot(forecast_dates, forecast_sarima, label='SARIMA Forecast')
            plt.title(f"SARIMA Forecast till {end_year}")
            plt.xlabel("Date")
            plt.ylabel("Exchange Rate (USD to INR)")
            plt.legend()
            st.pyplot(fig_sarima)

            # Calculate SARIMA metrics
            rmse_sarima = np.sqrt(np.mean((results.fittedvalues - train_data) ** 2))
            st.metric("SARIMA RMSE", f"{rmse_sarima:.2f}")

        except Exception as e:
            st.error(f"Error in SARIMA modeling: {str(e)}")
            st.info("Falling back to Prophet-only forecast")
            SARIMA_AVAILABLE = False

    st.subheader("🔍 SARIMA Forecast Preview")
    # Prepare SARIMA forecast data for display
    sarima_preview = pd.DataFrame({
        'Date': forecast_dates,
        'Predicted Rate': forecast_sarima.round(2)
    })
    sarima_preview.set_index('Date', inplace=True)

    # Display SARIMA preview
    st.dataframe(
        sarima_preview.tail(10),
        use_container_width=True,
        height=200
    )

    # Download button for SARIMA forecast
    csv_sarima = sarima_preview.to_csv()
    st.download_button(
        label="📥 Download SARIMA Forecast",
        data=csv_sarima,
        file_name="sarima_forecast.csv",
        mime="text/csv"
    )

    st.divider()

# Model Performance Comparison
st.header("📊 Model Performance Comparison")

# Calculate metrics for Prophet
rmse_prophet = np.sqrt(np.mean((forecast['yhat'].values[:len(processed_df)] - processed_df['y'].values) ** 2))
r2_prophet = 1 - np.sum((processed_df['y'].values - forecast['yhat'].values[:len(processed_df)])**2) / np.sum((processed_df['y'].values - processed_df['y'].mean())**2)
mae_prophet = np.mean(np.abs(forecast['yhat'].values[:len(processed_df)] - processed_df['y'].values))

# Create comparison columns
col1, col2 = st.columns(2)

with col1:
    st.subheader("Prophet Model")
    st.metric("RMSE", f"{rmse_prophet:.2f}")
    st.metric("R² Score", f"{r2_prophet:.2f}")
    st.metric("MAE", f"{mae_prophet:.2f}")

with col2:
    st.subheader("SARIMA Model")
    if SARIMA_AVAILABLE:
        rmse_sarima = np.sqrt(np.mean((results.fittedvalues - train_data) ** 2))
        r2_sarima = 1 - np.sum((train_data - results.fittedvalues)**2) / np.sum((train_data - train_data.mean())**2)
        mae_sarima = np.mean(np.abs(results.fittedvalues - train_data))
        
        st.metric("RMSE", f"{rmse_sarima:.2f}")
        st.metric("R² Score", f"{r2_sarima:.2f}")
        st.metric("MAE", f"{mae_sarima:.2f}")
    else:
        st.info("SARIMA model not available")

# Model comparison insights
st.subheader("📈 Model Comparison Insights")
if SARIMA_AVAILABLE:
    better_model = "Prophet" if rmse_prophet < rmse_sarima else "SARIMA"
    st.write(f"""
    - **Best Performing Model**: {better_model} (based on RMSE)
    - **Prophet Model** performs {'better' if rmse_prophet < rmse_sarima else 'worse'} at capturing long-term trends
    - **SARIMA Model** performs {'better' if rmse_sarima < rmse_prophet else 'worse'} at capturing short-term patterns
    """)
else:
    st.write("""
    - **Prophet Model** shows good performance in capturing both trends and seasonality
    - Consider using multiple models for more robust forecasting
    """)

# Update model limitations disclaimer
st.warning(
    "⚠️ Model performance metrics are based on historical fit. " +
    "Future predictions may vary due to changing economic conditions and external factors."
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