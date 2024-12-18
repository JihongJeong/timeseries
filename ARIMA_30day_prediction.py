import yfinance as yf
import pandas as pd
import numpy as np
import json
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, MACD
from datetime import datetime
from data_preprocessing import scale, inverse_scale
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error 

import warnings
warnings.filterwarnings(action='ignore')

def load_data():
    start_time = datetime.strptime("2021-11-20", "%Y-%m-%d")
    end_time = datetime.strptime("2024-11-20", "%Y-%m-%d")
    df = yf.download("BTC-USD", end = end_time, interval="1d")
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    sma = SMAIndicator(close, window = 20)
    df["SimpleMovingAverage"] = sma.sma_indicator().squeeze()
    
    stoch = StochasticOscillator(close, high, low, window=14, smooth_window = 3)
    df["StochasticSlowK"] = stoch.stoch().squeeze()
    
    rsi = RSIIndicator(close, window = 14)
    df["RSI"] = rsi.rsi().squeeze()
    
    bollinger = BollingerBands(close, window = 20)
    df["UpperBand"] = bollinger.bollinger_hband().values.ravel()
    df["LowerBand"] = bollinger.bollinger_lband().values.ravel()
    
    macd = MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd().squeeze()

    df = df.dropna()
    features = ["Close", "SimpleMovingAverage", "StochasticSlowK", "RSI", "UpperBand", "LowerBand", "MACD"]

    return df[features], features

df_original, features = load_data()
index = df_original.index
df_scaled, scaler = scale(df_original)
df = pd.DataFrame(df_scaled, columns=features, index = index)

df.index = pd.to_datetime(df.index)
df = df.resample('D').asfreq()
df = df.interpolate(method = 'linear')

result = adfuller(df['Close'])
print('ADF Statistic:', result[0])
print('p-value:', result[1])

if result[1] > 0.05:
    df['Close_diff'] = df['Close'].diff()
    df = df.dropna()
result = adfuller(df['Close_diff'])
# print('ADF Statistic:', result[0])
# print('p-value:', result[1])

# plot_acf(df['Close_diff'].dropna(), lags=20)
# plot_pacf(df['Close_diff'].dropna(), lags=20)
# plt.show()

exog = df[features[1:]]
exog = exog.dropna()

# Use Close price as a target
y = df['Close'][len(df)-len(exog):]

with open("optimal_pdq.json", "r") as pdq_json:
    optimal_pdq = json.load(pdq_json)
model = ARIMA(y, order = optimal_pdq["optimal_pdq"], exog = exog)
model_fit = model.fit()
exog = exog.iloc[-1, :].values
exog = np.repeat(np.array([exog]), 25, axis = 0)

forecast = model_fit.forecast(steps = 25, exog = exog).to_list()
result = inverse_scale([forecast], scaler)

start_time = datetime.strptime("2024-11-21", "%Y-%m-%d")
end_time = datetime.strptime("2024-12-16", "%Y-%m-%d")
df = yf.download("BTC-USD", start = start_time, end = end_time, interval="1d")

df_with_pred = df['Close']
df_with_pred.columns = ['Close']
df_with_pred['Pred'] = result.reshape(-1, 1)

print(f"MAE : {mean_absolute_error(df_with_pred['Close'], df_with_pred['Pred'])}")
print(f"RMSE : {np.sqrt(mean_squared_error(df_with_pred['Close'], df_with_pred['Pred']))}")