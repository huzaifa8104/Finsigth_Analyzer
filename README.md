# FinSight Analyzer

FinSight Analyzer is a powerful stock analysis tool built using Python, Streamlit, and Yahoo Finance API. It provides detailed insights into stock prices, technical indicators, fundamental data, and predictive analytics using machine learning models like LSTM and Prophet.

## Features

1. **Stock Price Analysis**:
   - View historical stock prices with interactive charts.
   - Analyze price movements, annual returns, and risk-adjusted returns.

2. **Fundamental Data**:
   - Access balance sheets, income statements, and cash flow statements.
   - View company information such as market cap, revenue, profit, and mutual fund holders.

3. **Technical Analysis**:
   - Golden Crossover Strategy (20-day and 50-day SMA).
   - Relative Strength Index (RSI).
   - Bollinger Bands.
   - Supertrend with EMA.

4. **Predictive Analytics**:
   - Prophet Forecasting: Time series forecasting using Facebook's Prophet model.
   - LSTM Prediction: Stock price prediction using Long Short-Term Memory (LSTM) neural networks.

5. **NIFTY 50 Analysis**:
   - Specialized analysis for the NIFTY 50 index, including pricing data and technical indicators.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-username/FinSight-Analyzer.git
   cd FinSight-Analyzer

2. Install the required dependencies:
    ```bash
    pip install -r requirements.txt

3. Run the Streamlit app:
    ```bash
    streamlit run FinSight_Analyzer.py

4. Open your browser and navigate to http://localhost:8501 to access the app.

## License
This FinSight Analyzer is open source and released under the MIT License. Feel free to use, modify, and distribute it as per the terms of the license.