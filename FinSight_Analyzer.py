import streamlit as st
import pandas as pd
import datetime
from home import main
import yfinance as yf
import plotly.express as px
from nifty_50 import nifty

st.title('FinSight Analyzer')
df = pd.read_csv('list.csv')
selected_title = st.sidebar.selectbox('Stock Name :', df['title'].tolist())
ticker = df[df['title'] == selected_title]['SYMBOL'].values[0]

# ticker = st.sidebar.text_input('Stock Name :', value='', placeholder='Search...')
end_date = (datetime.datetime.now())
start_date = end_date - pd.DateOffset(years=1)

start_date = st.sidebar.date_input('Start Date :', start_date)
end_date = st.sidebar.date_input('End Date :',end_date)

if end_date < start_date:
    st.error('End date must be after start date.')
    
if st.sidebar.button('Search'):
    try:
        if ticker=='' or ticker=='^NSEI':
            data = yf.download('^NSEI', start=start_date, end=end_date, auto_adjust=False)
            data.columns = data.columns.droplevel(1)
            title = f'Stock Price for NIFTY 50'
        else:
            data = yf.download(f'{ticker}.NS', start=start_date, end=end_date, auto_adjust=False)
            data.columns = data.columns.droplevel(1)
            title = f'Stock Price for {ticker}'

        fig = px.line(data, x=data.index, y='Adj Close', title=title)

        if data.empty:
            st.error(f'No data found for {ticker} Stock in the given date range.')
        else:
            st.plotly_chart(fig)

    except Exception as e:
        st.error(f'An error occurred: {e}')
    
    if ticker=='' or ticker=='^NSEI':
        nifty(data, ticker, start_date, end_date)
    else:
        main(data, ticker, start_date, end_date)


        