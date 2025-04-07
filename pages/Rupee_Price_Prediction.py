import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense
from datetime import datetime

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

# Future prediction
st.subheader("Predict Future Prices")
days = st.number_input("Days to Predict", min_value=1, max_value=30, value=7)
if st.button("Predict Future"):
    if model_choice == "LSTM":
        if 'lstm_model' in st.session_state and 'scaler' in st.session_state:
            model = st.session_state.lstm_model
            scaler = st.session_state.scaler
            last_days = df[-look_back:]['Close/Last'].values.reshape(-1, 1)
            scaled = scaler.transform(last_days)
            X_test = np.array([scaled[-look_back:]])
            X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
            future_pred = model.predict(X_test)
            future_pred = scaler.inverse_transform(future_pred)[0][0]
            st.write(f"Predicted closing price in {days} days: {future_pred:.2f}")
        else:
            st.warning("Train the LSTM model first!")
    else:
        st.warning("ARIMA future prediction not implemented yet.")

# Display raw data
if st.checkbox("Show Raw Data"):
    st.subheader("Raw Data")
    st.write(df.head(10))