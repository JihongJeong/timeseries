import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data(all: bool = True):
    df = yf.download("BTC-USD") if all else yf.download("BTC-USD", period = '3mo')
    close = np.squeeze(np.array(df["Close"]))
    high = np.squeeze(np.array(df["High"]))
    low = np.squeeze(np.array(df["Low"]))

    df["SimpleMovingAverage"] = talib.SMA(close, timeperiod = 20)
    df["StochasticSlowK"], df["StochasticSlowD"] = talib.STOCH(high, low, close, fastk_period=14, slowk_period=3, slowk_matype=0, slowd_period=3, slowd_matype=0)
    df["RSI"] = talib.RSI(close, timeperiod = 14)
    df["UpperBand"], df["MidBand"], df["LowerBand"] = talib.BBANDS(close, timeperiod=20)
    df["MACD"], macd_signal, macd_hist = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)

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