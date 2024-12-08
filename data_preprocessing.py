import yfinance as yf
import pandas as pd
import numpy as np
import ta
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(all: bool = True):
    df = yf.download("BTC-USD") if all else yf.download("BTC-USD", period = '3mo')
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

def plot_data(df, with_prediction = False):
    if with_prediction:
        plt.figure(figsize = (12, 8))
        plt.title("Close Prise vs Predicted Price")
        plt.plot(df["Close"], color = "blue", label = "Close price")    
        plt.plot(df["Predict"], color = "red", label = "predicted")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
        plt.show()
        
    else:
        plt.figure(figsize = (18, 12))
        plt.subplot(3, 1, 1)
        plt.title("Close Prise & Technical Indicators")
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

def scale(df):
    scaler = MinMaxScaler()
    scaler_ti = MinMaxScaler()
    df = np.array(df)
    df_scaled_close = scaler.fit_transform(df[:, 0].reshape(-1, 1))
    df_scaled_ti = scaler_ti.fit_transform(df[:, 1:].reshape(-1, 6))
    df_scaled = np.concatenate((df_scaled_close, df_scaled_ti), axis = 1)
    return df_scaled, scaler

def inverse_scale(df_scaled, scaler : MinMaxScaler):
    df = scaler.inverse_transform(df_scaled)
    
    return df

def create_sequences(X, y, seq_length=30):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        targets.append(y[i+seq_length])
    return np.array(sequences), np.array(targets)

def train_test_split(df):
    len_train = int(len(df)*0.8)
    X_train = df[:len_train]
    y_train = df[:len_train, 0]

    X_test = df[len_train:]
    y_test = df[len_train:, 0]
    
    return X_train, y_train, X_test, y_test

def add_result(df, result):
    len_result = len(result)
    df_pd = pd.DataFrame(df)
    pred = [None for _ in range(df.shape[0] - len_result)] + result
    df_pd["Predict"] = pred
    
    return df_pd