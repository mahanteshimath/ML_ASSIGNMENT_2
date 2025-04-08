# Machine Learning Applications

A collection of Streamlit-based machine learning applications for educational and forecasting purposes.

## 📊 Applications

### 1. Student Performance Predictor
An ML-powered application that predicts student performance based on various factors:
- Uses Random Forest Regressor
- Features include study hours, sleep patterns, and extracurricular activities
- Real-time predictions with step-by-step visualization
- Performance metrics (RMSE, R² Score)

### 2. USD to INR Exchange Rate Forecaster
Long-term currency exchange rate prediction system:
- Dual model approach (Prophet and SARIMA)
- Forecast horizon up to 2050
- Interactive visualizations and component analysis
- Downloadable forecast data

## 🚀 Installation

1. Clone the repository:
```bash
git clone https://github.com/mahanteshimath/ML_ASSIGNMENT_2.git
cd ML_ASSIGNMENT_2
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## 💻 Usage

Launch the application:
```bash
streamlit run Home.py
```

## 🛠️ Technical Stack

### Common Framework
- **Web Framework**: Streamlit
- **Data Processing**: Pandas, NumPy
- **Visualization**: Plotly, Matplotlib

### Student Performance Predictor
- **ML Model**: Random Forest Regressor
- **Preprocessing**: StandardScaler, OneHotEncoder
- **Pipeline**: sklearn.pipeline

### Exchange Rate Forecaster
- **Time Series Models**: 
  - Facebook Prophet
  - SARIMA (Seasonal ARIMA)
- **Statistical Analysis**: statsmodels

## 📈 Features

### Student Performance Predictor
- Interactive input form
- Real-time predictions
- Step-by-step processing visualization
- Model performance metrics

### Exchange Rate Forecaster
- Dual model comparison
- Component analysis
- Confidence intervals
- CSV export functionality
- Historical data preview

## ⚠️ Limitations

### Student Performance Predictor
- Limited to provided input parameters
- Based on sample student data
- May not generalize to all educational contexts

### Exchange Rate Forecaster
- Based on historical patterns only
- Cannot predict unexpected economic events
- Accuracy decreases with forecast horizon

## 🧪 Model Performance

Both applications include detailed performance metrics:
- RMSE (Root Mean Square Error)
- R² Score (Coefficient of Determination)
- MAE (Mean Absolute Error)

## 👥 Contributors

- Srijit Ghatak (G24AIT2079)
- Mahantesh Hiremath (G24AIT2178)

## 🔗 Links

- [Live Demo](https://iitj-ml-learnings.streamlit.app/)
- [Project Repository](https://github.com/mahanteshimath/ML_ASSIGNMENT_2)

## 📝 License

This project is licensed under the MIT License.
