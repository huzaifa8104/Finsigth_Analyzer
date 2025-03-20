import streamlit as st
import pandas as pd
import yfinance as yf
import datetime
import plotly.express as px
import numpy as np
import graph as gp

def nifty(data, ticker, start_date, end_date):
    pricing_data, technical_analysis = st.tabs(['Pricing Data', 'Technical Analysis'])

    with pricing_data:
        data2 = data
        data2['% Change'] = data['Adj Close'] / data['Adj Close'].shift(1) - 1
        annual_return = data2['% Change'].mean() * 252 * 100
        st.write('Annual Return is ', annual_return, '%')
        stdev = np.std(data2['% Change']) * np.sqrt(252)
        st.write('Standard Deviation is ', stdev * 100, '%')
        st.write('Risk Adjusted Return is ', annual_return / (stdev * 100))
        st.write('Price Movements')
        st.write(data2)

    with technical_analysis:
        gp.GoldenCrossoverSignal(data)
        gp.relative_strength_index(data, ticker)
        gp.bollinger_band(data, ticker)
        gp.supertrend(data, 'NIFTY 50')
        gp.prophet_prediction(ticker, end_date)
        gp.lstm(ticker, end_date)

