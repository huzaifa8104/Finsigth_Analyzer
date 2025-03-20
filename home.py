import streamlit as st
import yfinance as yf
import plotly.express as px
import pandas as pd
import numpy as np
import graph as gp

def main(data, ticker, start_date, end_date):     
    pricing_data, fundamental_data, about, technical_analysis = st.tabs(['Pricing Data', 'Fundamental Data', 'About', 'Technical Analysis',])
    
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
    
    def handle_duplicate_columns(df):
        cols = pd.Series(df.columns)
        for dup in cols[cols.duplicated()].unique():
            cols[cols[cols == dup].index.values.tolist()] = [dup + '_' + str(i) if i != 0 else dup for i in range(sum(cols == dup))]
        df.columns = cols
        return df

    with fundamental_data:
        try:
            stock = yf.Ticker(f'{ticker}.NS')
            st.subheader('Balance Sheet')
            balance_sheet = stock.balance_sheet
            if not balance_sheet.empty:
                balance_sheet = handle_duplicate_columns(balance_sheet)
                bs = balance_sheet
                st.write(bs)
            else:
                st.write('No balance sheet data available.')

            st.subheader('Income Statement')
            income_statement = stock.financials
            if not income_statement.empty:
                income_statement = handle_duplicate_columns(income_statement)
                is1 = income_statement
                st.write(is1)
            else:
                st.write('No Income statement data available.')

            st.subheader('Cash Flow Statement')
            cash_flow = stock.cashflow
            if not cash_flow.empty:
                cash_flow = handle_duplicate_columns(cash_flow)
                cf = cash_flow
                st.write(cf)
            else:
                st.write('No Cash Flow statement data available.')

        except Exception as e:
            st.error(f'An error occurred while fetching fundamental data: {e}')


    def fetch_company_info(ticker_):
        stock_ = yf.Ticker(ticker_)
        info = stock_.info
        try:
            mutual_fund_holders = stock_.mutualfund_holders            
            if mutual_fund_holders is not None:
                if 'pctHeld' in mutual_fund_holders.columns:
                    mutual_fund_holders['pctHeld'] *= 100
                    mutual_fund_holders['Value'] /= 10**7 
                    mutual_fund_holders.rename(columns={'pctHeld': 'Held%', 'Value': 'Value(in Cr.)'}, inplace=True)                    
                mutual_fund_holders = mutual_fund_holders.round(2)  # Round to 2 decimal places
            else:
                mutual_fund_holders = None
            
        except Exception:
            mutual_fund_holders = 'Data not available'
            
        return {
            'Name': info.get('longName', 'N/A'),
            'Description': info.get('longBusinessSummary', 'N/A'),
            'Sector': info.get('sector', 'N/A'),
            'Industry': info.get('industry', 'N/A'),
            'Market Cap': info.get('marketCap', 'N/A'),
            'Revenue': info.get('totalRevenue', 'N/A'),
            'Profit': info.get('netIncomeToCommon', 'N/A'),
            'Profit Margin': info.get('profitMargins', 'N/A'),
            'Main Products': info.get('products', 'N/A'), 
            'Mutual Fund Holders': mutual_fund_holders
        }

    with about:
        st.subheader(f"About {ticker}")
        info = fetch_company_info(f'{ticker}.NS')

        st.write(f"**Company Name**: {info['Name']}")
        st.write(f"**Description**: {info['Description']}")
        st.write(f"**Sector**: {info['Sector']}")
        st.write(f"**Industry**: {info['Industry']}")
        st.write(f"**Market Cap**: {info['Market Cap']}")
        st.write(f"**Revenue**: {info['Revenue']}")
        st.write(f"**Profit**: {info['Profit']}")
        st.write(f"**Profit Margin**: {info['Profit Margin']}")
        st.write(f"**Main Products**: {info['Main Products']}")
        
        if isinstance(info['Mutual Fund Holders'], pd.DataFrame) and not info['Mutual Fund Holders'].empty:
            st.write("**Mutual Fund Holders**:")
            st.dataframe(info['Mutual Fund Holders'])


    with technical_analysis:
        gp.GoldenCrossoverSignal(data)
        gp.relative_strength_index(data, ticker)
        gp.bollinger_band(data, ticker)
        gp.supertrend(data, ticker)
        gp.prophet_prediction(ticker, end_date)
        gp.lstm(ticker, end_date)
