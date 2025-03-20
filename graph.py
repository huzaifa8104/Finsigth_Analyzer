import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from neuralprophet import NeuralProphet
from neuralprophet.configure import ConfigSeasonality
from prophet import Prophet
import tensorflow as tf
from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional # type: ignore
from keras.callbacks import EarlyStopping, ModelCheckpoint # type: ignore
from tensorflow.keras.optimizers import Adam # type: ignore
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator # type: ignore
from sklearn.preprocessing import RobustScaler
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

def GoldenCrossoverSignal(data):
    df = data
    df['20_SMA'] = df['Close'].rolling(window=20, min_periods=1).mean()
    df['50_SMA'] = df['Close'].rolling(window=50, min_periods=1).mean()
    df['Signal'] = 0
    df['Signal'] = np.where(df['20_SMA'] > df['50_SMA'], 1, 0)
    df['Position'] = df['Signal'].diff()

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=df.index, y=df['Adj Close'], mode='lines', name='Adj Close Price', line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=df.index, y=df['20_SMA'], mode='lines', name='20-day SMA', line=dict(color='brown')))
    fig.add_trace(go.Scatter(x=df.index, y=df['50_SMA'], mode='lines', name='50-day SMA', line=dict(color='green')))
    buy_signals = df[df['Position'] == 1]
    fig.add_trace(go.Scatter(x=buy_signals.index, y=buy_signals['20_SMA'], mode='markers', name='Buy', marker=dict(symbol='triangle-up', color='green', size=15)))
    sell_signals = df[df['Position'] == -1]
    fig.add_trace(go.Scatter(x=sell_signals.index, y=sell_signals['20_SMA'], mode='markers', name='Sell', marker=dict(symbol='triangle-down', color='red', size=15)))
    fig.update_layout(
        title='Golden Crossover Strategy',
        xaxis_title='Date',
        yaxis_title='Price in Rupees',
        legend=dict(x=0, y=1, traceorder='normal'),
        template='plotly_dark'
    )
    st.plotly_chart(fig)

def relative_strength_index(data, ticker):
    def calculate_rsi(df, window=14):
        delta = df['Close'].diff()
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        avg_gain = gain.rolling(window=window, min_periods=window).mean()
        avg_loss = loss.rolling(window=window, min_periods=window).mean()
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        df['RSI'] = rsi
        return df

    def plot_rsi(data, ticker):
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=data.index,
            y=data['RSI'],
            mode='lines',
            name='RSI',
            line=dict(color='blue')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[70] * len(data),
            mode='lines',
            name='Overbought Line',
            line=dict(color='red', width=1, dash='dash')
        ))
        fig.add_trace(go.Scatter(
            x=data.index,
            y=[30] * len(data),
            mode='lines',
            name='Oversold Line',
            line=dict(color='green', width=1, dash='dash')
        ))
        fig.update_layout(
            title=f'Relative Strength Index (RSI) for {ticker}',
            xaxis_title='Date',
            yaxis_title='RSI',
            xaxis_rangeslider_visible=False,
            yaxis=dict(range=[0, 100])
        )
        st.plotly_chart(fig)

    if ticker:
        if not data.empty:
            df = calculate_rsi(data)
            plot_rsi(df, ticker)
        else:
            st.write(f"No data available for {ticker}.")



def bollinger_band(data, ticker):
    def calculate_bollinger_bands(df, window=20, num_std_dev=2):
        df['Rolling Mean'] = df['Close'].rolling(window=window).mean()
        df['Rolling Std'] = df['Close'].rolling(window=window).std()
        df['Upper Band'] = df['Rolling Mean'] + (df['Rolling Std'] * num_std_dev)
        df['Lower Band'] = df['Rolling Mean'] - (df['Rolling Std'] * num_std_dev)
        return df

    def plot_bollinger_bands(df, ticker):
        fig = go.Figure()
        fig.add_trace(go.Candlestick(
            x=df.index,
            open=df['Open'],
            high=df['High'],
            low=df['Low'],
            close=df['Close'],
            name='Candlestick'
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Upper Band'],
            mode='lines',
            name='Upper Band',
            line=dict(color='red', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Lower Band'],
            mode='lines',
            name='Lower Band',
            line=dict(color='red', width=1)
        ))
        fig.add_trace(go.Scatter(
            x=df.index,
            y=df['Rolling Mean'],
            mode='lines',
            name='Rolling Mean',
            line=dict(color='blue', width=1)
        ))
        fig.update_layout(
            title=f'Bollinger Bands for {ticker}',
            xaxis_title='Date',
            yaxis_title='Price',
            xaxis_rangeslider_visible=False
        )
        st.plotly_chart(fig)

    if ticker:
        if not data.empty:
            df = calculate_bollinger_bands(data)
            plot_bollinger_bands(df, ticker)
        else:
            st.write(f"No data available for {ticker}.")
            

def supertrend(data, ticker, ema_period=50):
    def calculate_supertrend(df, period=10, multiplier=3):
        df['ATR'] = df['High'] - df['Low']
        df['ATR'] = df[['ATR']].rolling(window=period).mean()
        df['Upper Basic'] = (df['High'] + df['Low']) / 2 + multiplier * df['ATR']
        df['Lower Basic'] = (df['High'] + df['Low']) / 2 - multiplier * df['ATR']
        df['Upper Band'] = df[['Upper Basic']].rolling(window=period).max()
        df['Lower Band'] = df[['Lower Basic']].rolling(window=period).min()
        df['Supertrend'] = df['Upper Band']
        df.loc[df['Close'] <= df['Upper Band'], 'Supertrend'] = df['Lower Band']
        return df

    def calculate_ema(df, period=50):
        df[f'EMA_{period}'] = df['Close'].ewm(span=period, adjust=False).mean()
        return df

    if ticker:        
        if not data.empty:
            df = calculate_supertrend(data)
            df = calculate_ema(df, ema_period)
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df.index,
                open=df['Open'],
                high=df['High'],
                low=df['Low'],
                close=df['Close'],
                name='Candlestick'
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df['Supertrend'],
                mode='lines',
                name='Supertrend',
                line=dict(color='orange', width=2)
            ))
            
            fig.add_trace(go.Scatter(
                x=df.index,
                y=df[f'EMA_{ema_period}'],
                mode='lines',
                name=f'EMA {ema_period}',
                line=dict(color='blue', width=2)
            ))
            
            fig.update_layout(title=f'Supertrend and EMA for {ticker}',
                            xaxis_title='Date',
                            yaxis_title='Price',
                            xaxis_rangeslider_visible=False)
            
            st.plotly_chart(fig)
        else:
            st.write(f"No data available for {ticker}.")


def prophet_prediction(ticker, end_date):
    if ticker != '^NSEI':
        ticker = f"{ticker}.NS"
    start_date = end_date - pd.DateOffset(years=3)
    data = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    data.reset_index(inplace=True)
    data = data[['Date', 'Close']]
    data.columns = ['ds', 'y']
    
    model = Prophet(daily_seasonality=False, weekly_seasonality=True, yearly_seasonality=True)
    model.fit(data)
    future_dates = model.make_future_dataframe(periods=90)
    prediction = model.predict(future_dates)
    end_date = pd.Timestamp(end_date)
    predictions = prediction[prediction['ds'] <= end_date]
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=data['ds'], 
        y=data['y'], 
        mode='lines', 
        name='Actual', 
        line=dict(color='green')
    ))
    
    fig.add_trace(go.Scatter(
        x=prediction['ds'], 
        y=prediction['yhat'], 
        mode='lines', 
        name='Forecast', 
        line=dict(color='blue')
    ))
    
    fig.update_layout(
        title='Prophet Forecast vs Actual',
        xaxis_title='Date',
        yaxis_title='Close Price',
        legend=dict(x=0, y=1, traceorder='normal'),
        template='plotly_dark'
    )
    st.plotly_chart(fig)
    
    if not predictions.empty:
        r_squared = r2_score(data['y'], predictions['yhat'][:len(data)])
        mse = mean_squared_error(data['y'], predictions['yhat'][:len(data)])
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(data['y'], predictions['yhat'][:len(data)])
        
        st.write(f"R-squared: {r_squared:.4f}")
        st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        st.write(f"Mean Absolute Error (MAE): {mae:.4f}")

def create_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(50, return_sequences=True, input_shape=input_shape))
    model.add(LSTM(50))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

def prepare_sequences(df, n_steps):
    X, y = [], []
    for i in range(len(df) - n_steps):
        X.append(df['Scaled_Close'].iloc[i:i+n_steps].values)
        y.append(df['Scaled_Close'].iloc[i+n_steps])
    return np.array(X), np.array(y)

def lstm(ticker, end_date):
    if ticker != '^NSEI':
        ticker = f"{ticker}.NS"
    start_date = end_date - pd.DateOffset(years=3)
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=False)
    df.columns = df.columns.droplevel(1)
    df = df[['Close']]
    scaler = MinMaxScaler(feature_range=(0, 1))
    df['Scaled_Close'] = scaler.fit_transform(df[['Close']])

    n_steps=60
    X, y = prepare_sequences(df, n_steps)
    X = X.reshape((X.shape[0], X.shape[1], 1))

    with st.spinner('Generating LSTM Models...'):
        model = create_lstm_model((X.shape[1], 1))
        model.fit(X, y, epochs=100, batch_size=32)
    
    forecast_period = 21
    with st.spinner('Predicting Future Price...'):
        last_sequence = df['Scaled_Close'].values[-n_steps:]
        predictions = []

        for _ in range(forecast_period):
            x = last_sequence.reshape((1, n_steps, 1))
            next_price = model.predict(x)
            next_price = scaler.inverse_transform(next_price)
            predictions.append(next_price[0][0])
            last_sequence = np.append(last_sequence[1:], scaler.transform([[next_price[0][0]]])[0])
    
    future_dates = [end_date + pd.DateOffset(days=i) for i in range(1, forecast_period + 1)]
    
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(x=df.index, y=df['Close'], mode='lines', name='Historical Prices'))
    
    fig.add_trace(go.Scatter(x=future_dates, y=predictions, mode='lines', name='Predicted Prices', line=dict(dash='dash')))
    
    fig.update_layout(
        title=f'LSTM Price Prediction',
        xaxis_title='Date',
        yaxis_title='Price'
    )
    
    st.plotly_chart(fig)






