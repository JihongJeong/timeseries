import yfinance as yf
import pandas as pd
import numpy as np
import talib
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

def load_data():
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
    
    return df[features], features

def plot_data(df, with_prediction = False):
    if with_prediction:
        plt.figure(figsize = (12, 8))
        plt.plot(df["Close"], color = "blue", label = "Close price")    
        plt.plot(df["Pred"], color = "red", label = "predicted")
        plt.legend()
        plt.xlabel("Date")
        plt.ylabel("Price")
    else:
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

def scale(df):
    # 데이터 정규화
    scaler = MinMaxScaler()
    df_scaled = scaler.fit_transform(df)

    return df_scaled, scaler

def inverse_scale(df_scaled, scaler : MinMaxScaler):
    df = scaler.invers_transform(df_scaled)
    
    return df

# LSTM 모델을 위한 데이터셋 생성
def create_sequences(X, y, seq_length=30):
    sequences = []
    targets = []
    for i in range(len(X) - seq_length):
        sequences.append(X[i:i+seq_length])
        targets.append(y[i+seq_length])
    return np.array(sequences), np.array(targets)

def train_test_split(df):
    strat_point = int(len(df)*0.1)
    len_train = int(len(df)*0.9)
    X_train = df[strat_point:len_train]
    y_train = df[strat_point:len_train, 0]

    X_test = df[len_train:]
    y_test = df[len_train:, 0]
    
    return X_train, y_train, X_test, y_test



