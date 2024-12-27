import streamlit as st
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_pacf, plot_acf
import numpy as np
def set_bg_hack_url():
    st.markdown(
         f"""
         <style>
         .stApp {{
             background: url("https://static.vecteezy.com/system/resources/previews/006/852/804/original/abstract-blue-background-simple-design-for-your-website-free-vector.jpg");
             background-size: cover;
         }}
         </style>
         """,
         unsafe_allow_html=True
     )

@st.cache_data
def load_data(ticker, start, end):
    data = yf.download(ticker, start, end)['Close']
    return data

def plot_raw_data(data, stock_name):
    fig = plt.figure(figsize=(12, 6))
    plt.plot(data)
    plt.title(f'{stock_name} Closing Price Over Time', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Price (USD)')
    st.pyplot(fig)

def plot_returns(returns, stock_name):
    fig = plt.figure(figsize=(16, 4))
    plt.plot(returns)
    plt.title(f'{stock_name} Returns', fontsize=20)
    plt.xlabel('Date')
    plt.ylabel('Percentage Returns')
    st.pyplot(fig)

def plot_pacf_acf(data):
    lag_acf = range(2, 50)
    lag_pacf = range(2, 40)
    height = 4
    width = 12
    f, ax = plt.subplots(nrows=2, ncols=1, figsize=(width, 2 * height))
    fig1 = plot_acf(data, lags=lag_acf, ax=ax[0])
    fig2 = plot_pacf(data, lags=lag_pacf, ax=ax[1], method='ols')
    plt.tight_layout()
    st.pyplot(f)

@st.cache_resource
def fit_arima_model(train, p, d, q):
    model = ARIMA(train, order=(p, d, q))
    model_fit = model.fit()
    return model_fit

def forecast_last_month(model_fit, returns, end_date):
    forecast = model_fit.get_forecast(steps=30)
    pred = forecast.predicted_mean
    pred_conf = forecast.conf_int()

    last_month_start = returns.index[-30]
    actual = returns[last_month_start:]

    pred.index = actual.index[:30]
    pred_conf.index = actual.index[:30]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(actual.index, actual, label='Actual Returns', color='blue')
    ax.plot(pred.index, pred, label='Forecast', color='red')
    ax.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1], color='pink', alpha=0.3)
    ax.set_title('ARIMA Forecast vs Actual for Last Month', fontsize=20)
    ax.set_xlabel('Date')
    ax.set_ylabel('Returns')
    ax.legend()
    st.pyplot(fig)

def main():
    st.title("Stock Prediction App")
    set_bg_hack_url()

    start_date = st.date_input("Start Date", datetime(2022, 1, 1))
    end_date = datetime.today().date()

    stock_name = st.text_input("Enter Stock Ticker (e.g., AAPL)", 'AAPL')

    data_loader_state = st.text("Loading data...")
    data = load_data(stock_name, start_date, end_date)
    data_loader_state.text("Loading data...done!")

    st.subheader('Raw Data')
    st.write(data.tail())

    st.subheader("Closing Price vs Time Chart")
    plot_raw_data(data, stock_name)

    returns = 100 * data.pct_change().dropna()

    st.subheader("Returns")
    plot_returns(returns, stock_name)

    st.subheader("ACF and PACF Plots")
    plot_pacf_acf(data)

    st.subheader("Model Selection and Summary")
    p = st.selectbox("Select AR Order (p)", [1, 2, 3,4,5,6,7])
    d = st.selectbox("Select Difference Order (d)", [0, 1, 2,3,4,5,6,7])
    q = st.selectbox("Select MA Order (q)", [1, 2, 3,4,5,6,7])

    train_data = returns[:-30]
    model_fit = fit_arima_model(train_data, p, d, q)
    st.write(model_fit.summary())

    st.subheader("ARIMA Forecast vs Actual for Last Month")
    forecast_last_month(model_fit, returns, end_date)

if __name__ == '__main__':
    main()
