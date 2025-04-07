# Future prediction
st.subheader("Predict Future Prices")
days = st.number_input("Days to Predict", min_value=1, max_value=30, value=7)
if st.button("Predict Future"):
    if model_choice == "LSTM":
        if 'lstm_model' in st.session_state and 'scaler' in st.session_state:
            try:
                model = st.session_state.lstm_model
                scaler = st.session_state.scaler
                last_days = df[-look_back:]['Close/Last'].values.reshape(-1, 1)
                scaled = scaler.transform(last_days)
                X_test = np.array([scaled[-look_back:]])
                X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))
                future_pred = model.predict(X_test)
                future_pred = scaler.inverse_transform(future_pred)[0][0]
                st.write(f"Predicted closing price in {days} days: {future_pred:.2f}")
            except Exception as e:
                st.error(f"Error during future prediction: {e}")
        else:
            st.warning("Train the LSTM model first!")
    else:
        if 'forecast' in st.session_state:
            try:
                train = df['Close/Last'].values
                model = ARIMA(train, order=(1, 1, 1))
                model_fit = model.fit()
                forecast_values = model_fit.forecast(steps=days)[0]
                future_dates = pd.date_range(start=df['Date'].iloc[-1], periods=days + 1, freq='D')[1:]
                future_df = pd.DataFrame({'Date': future_dates, 'Prediction': forecast_values})
                st.write(future_df)
                st.line_chart(future_df.set_index('Date')['Prediction'])
            except Exception as e:
                st.error(f"Error during ARIMA future prediction: {e}")
        else:
            st.warning("Train the ARIMA model first!")

# Add axis labels to all graphs
def plot_predictions(test_dates, actual, predicted, model_name):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(test_dates, actual, label='Actual')
    ax.plot(test_dates, predicted, label='Predicted')
    ax.set_title(f"{model_name} Predictions")
    ax.set_xlabel('Date')
    ax.set_ylabel('Closing Price (INR)')
    ax.legend()
    st.pyplot(fig)

st.subheader("Historical Closing Prices")
fig, ax = plt.subplots(figsize=(12, 6))
ax.plot(df['Date'], df['Close/Last'], label='Closing Price')
ax.set_title("Historical Closing Prices")
ax.set_xlabel("Date")
ax.set_ylabel("Closing Price (INR)")
ax.legend()
st.pyplot(fig)
