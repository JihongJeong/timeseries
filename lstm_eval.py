import torch
import yfinance as yf
import pandas as pd
import numpy as np
import talib
from torch.utils.data import DataLoader, TensorDataset
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler
from MyModel import LSTMModel
import matplotlib.pyplot as plt

# 데이터 로드 (df는 주식 데이터)
df = yf.download("BTC-USD")
close = np.squeeze(np.array(df["Close"]))
high = np.squeeze(np.array(df["High"]))
low = np.squeeze(np.array(df["Low"]))

df["SimpleMovingAverage"] = talib.SMA(close, timeperiod = 20)
df["StochasticSlowK"], df["StochasticSlowD"] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
df["RSI"] = talib.RSI(close, timeperiod = 14)
df["UpperBand"], df["MidBand"], df["LowerBand"] = talib.BBANDS(close, timeperiod=20)
df["MACD"], macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)


features = ["Close", "SimpleMovingAverage", "StochasticSlowK", "RSI", "UpperBand", "LowerBand", "MACD"]
target = "Close"

plt.figure(figsize = (18, 12))
plt.subplot(3, 1, 1)
plt.plot(df["Close"], color = "blue", label = "Close price")
plt.plot(df["SimpleMovingAverage"], color = "red", label = "SMA")
plt.plot(df["UpperBand"], color = "black", label = "Bollinger Band")
plt.plot(df["LowerBand"], color = "black", label = "Bollinger Band")
plt.legend()
plt.xlabel("Date")
plt.ylabel("Price")

plt.subplot(3, 1, 2)
plt.plot(df["RSI"], color = "green", label = "RSI")
plt.legend()
plt.xlabel("Date")
plt.ylabel("RSI Score")

plt.subplot(3, 1, 3)
plt.plot(df["MACD"], color = "purple", label = "MACD")
plt.legend()
plt.xlabel("Date")
plt.ylabel("MACD")
plt.show()