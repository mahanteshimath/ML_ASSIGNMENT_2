import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime, timedelta

# Load data
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/mahanteshimath/ML_ASSIGNMENT_2/refs/heads/main/data/HistoricalData_Currency.csv"
    df = pd.read_csv(url)
    df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%Y')
    df = df.sort_values('Date').reset_index(drop=True)
    df = df.drop(columns=['Volume']).dropna()
    return df

# Preprocess for LSTM
def preprocess_lstm(df, look_back=30):
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(df['Close/Last'].values.reshape(-1, 1))
    
    X, y = [], []
    for i in range(look_back, len(scaled_data)):
        X.append(scaled_data[i-look_back:i, 0])
        y.append(scaled_data[i, 0])
    
    X = np.array(X)
    y = np.array(y)
    X = np.reshape(X, (X.shape[0], X.shape[1], 1))
    return X, y, scaler

# Build LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    return model

# Title and description
st.title("USD-INR Exchange Rate Prediction")
st.write("Predict closing prices using LSTM or ARIMA models.")

# Load data
df = load_data()
st.subheader("Historical Closing Prices")
st.line_chart(df['Close/Last'])

# Sidebar for controls
st.sidebar.header("Model Configuration")
model_choice = st.sidebar.selectbox("Select Model", ["LSTM", "ARIMA"])
train_size = st.sidebar.slider("Training Data Size (%)", 50, 80, 80)
if model_choice == "LSTM":
    look_back = st.sidebar.slider("LSTM Look Back Period", 10, 60, 30)

# Split data
train_size = int(len(df) * (train_size/100))
train_df = df[:train_size]
test_df = df[train_size:]

# Train models
if st.sidebar.button("Train Model"):
    st.spinner("Training model...")
    if model_choice == "LSTM":
        # LSTM preprocessing
        X_train, y_train, scaler = preprocess_lstm(train_df, look_back)
        
        # Build and train LSTM
        model = build_lstm_model((X_train.shape[1], 1))
        model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)
        st.session_state.lstm_model = model
        st.session_state.scaler = scaler
        
        # Predict on test data
        X_test, _, _ = preprocess_lstm(test_df, look_back)
        lstm_pred = model.predict(X_test)
        lstm_pred = scaler.inverse_transform(lstm_pred)
        st.session_state.forecast = pd.DataFrame({
            'Date': test_df['Date'][look_back:],
            'Prediction': lstm_pred.flatten()
        })
        
    else:
        # ARIMA
        train = train_df['Close/Last'].values
        test = test_df['Close/Last'].values
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        forecast_values = model_fit.forecast(steps=len(test))[0]
        st.session_state.forecast = pd.DataFrame({
            'Date': test_df['Date'],
            'Prediction': forecast_values
        })
    st.success("Model trained successfully!")

# Show predictions
if 'forecast' in st.session_state:
    forecast = st.session_state.forecast
    st.subheader("Predictions vs Actual")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_df['Date'], test_df['Close/Last'], label='Actual')
    ax.plot(forecast['Date'], forecast['Prediction'], label='Predicted')
    plt.title(f"{model_choice} Predictions")
    plt.xlabel('Date')
    plt.ylabel('Closing Price (INR)')
    plt.legend()
    st.pyplot(fig)
    
    # Metrics
    if model_choice == "LSTM":
        rmse = np.sqrt(np.mean((forecast['Prediction'] - test_df['Close/Last'][look_back:])**2))
    else:
        rmse = np.sqrt(np.mean((forecast['Prediction'] - test_df['Close/Last'])**2))
    st.write(f"RMSE: {rmse:.2f}")

# Future prediction with sliding window
st.subheader("Extended Forecast to 2050")
end_year = st.number_input("Predict up to Year", min_value=2023, max_value=2050, value=2050)
if st.button("Generate Long-Term Forecast"):
    if model_choice == "LSTM":
        if 'lstm_model' in st.session_state and 'scaler' in st.session_state:
            model = st.session_state.lstm_model
            scaler = st.session_state.scaler
            
            # Initialize with last 'look_back' days
            last_days = df[-look_back:]['Close/Last'].values
            history = list(last_days)
            predictions = []
            start_date = df['Date'].iloc[-1]
            target_year = end_year
            current_date = start_date
            
            while current_date.year < target_year:
                # Prepare input (last 'look_back' days)
                X = np.array(history[-look_back:]).reshape(1, look_back, 1)
                # Predict next day
                scaled_pred = model.predict(X)
                pred = scaler.inverse_transform(scaled_pred)[0][0]
                predictions.append(pred)
                history.append(pred)
                current_date += timedelta(days=1)
            
            # Create DataFrame for predictions
            dates = pd.date_range(start=df['Date'].iloc[-1], periods=len(predictions)+1, freq='D')[1:]
            forecast_df = pd.DataFrame({'Date': dates, 'Prediction': predictions})
            
            st.write(f"Sliding Window Forecast up to {end_year}:")
            st.line_chart(forecast_df.set_index('Date'))
            st.warning("⚠️ LSTM predictions beyond 30 days may drift due to compounding errors.")
        else:
            st.warning("Train the LSTM model first!")
    else:
        # ARIMA multi-step forecast
        train = df['Close/Last'].values
        model = ARIMA(train, order=(1,1,1))
        model_fit = model.fit()
        
        # Calculate steps to 2050
        end_date = datetime(end_year, 12, 31)
        steps = (end_date - df['Date'].iloc[-1]).days
        forecast_values = model_fit.forecast(steps=steps)[0]
        
        dates = pd.date_range(start=df['Date'].iloc[-1], periods=steps+1, freq='D')[1:]
        forecast_df = pd.DataFrame({'Date': dates, 'Prediction': forecast_values})
        
        st.write(f"ARIMA Forecast up to {end_year}:")
        st.line_chart(forecast_df.set_index('Date'))
        st.warning("⚠️ ARIMA may not capture long-term trends accurately.")

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df.head(10))



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