import yfinance as yf
import pandas as pd
import numpy as np
import torch
from torch import nn
from MyModel import LSTMModel
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.volatility import BollingerBands
from ta.trend import SMAIndicator, MACD
from datetime import datetime
from data_preprocessing import scale, inverse_scale
from sklearn.metrics import mean_absolute_error, mean_squared_error 

# 기술 지표 계산 함수
def calculate_technical_indicators(df):
    close = df["Close"].squeeze()
    high = df["High"].squeeze()
    low = df["Low"].squeeze()

    sma = SMAIndicator(close, window=20)
    df["SimpleMovingAverage"] = sma.sma_indicator().squeeze()
    
    stoch = StochasticOscillator(close, high, low, window=14, smooth_window=3)
    df["StochasticSlowK"] = stoch.stoch().squeeze()
    
    rsi = RSIIndicator(close, window=14)
    df["RSI"] = rsi.rsi().squeeze()
    
    bollinger = BollingerBands(close, window=20)
    df["UpperBand"] = bollinger.bollinger_hband().squeeze()
    df["LowerBand"] = bollinger.bollinger_lband().squeeze()
    
    macd = MACD(close, window_fast=12, window_slow=26, window_sign=9)
    df["MACD"] = macd.macd().squeeze()
    
    df = df.dropna()
    return df

# 데이터 준비
def get_bitcoin_data():
    start_time = datetime.strptime("2021-11-20", "%Y-%m-%d")
    end_time = datetime.strptime("2024-11-20", "%Y-%m-%d")
    df = yf.download("BTC-USD", end = end_time, interval="1d")  # 지난 2년간 일별 데이터
    df = calculate_technical_indicators(df)
    features = ["Close", "SimpleMovingAverage", "StochasticSlowK", "RSI", "UpperBand", "LowerBand", "MACD"]
    return df[features]

# 데이터 정규화
def normalize_data(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0), np.mean(data, axis=0), np.std(data, axis=0)

def reverse(data, mean, std):
    result = []
    for i in data:
        result.append(i * std + mean)
    return result

def predict_future_prices(model, data, seq_length, steps):
    predictions = []
    input_seq = data[-seq_length:]  # 마지막 시퀀스를 가져옴
    input_seq = torch.tensor(input_seq, dtype=torch.float32).unsqueeze(0)
    
    model.eval()
    with torch.no_grad():
        for _ in range(steps):
            pred = model(input_seq)
            predictions.append(pred.item())
            pred = torch.cat((pred, input_seq[:, -1, 1:]), dim = 1)
            # 다음 입력에 추가
            new_seq = torch.cat((input_seq[:, 1:, :], pred.unsqueeze(0)), dim=1)
            input_seq = new_seq
            
    return np.array(predictions)

# 모델 로드
def load_model(model_path, input_size, hidden_size, output_size, seq_length):
    model = LSTMModel(input_size, hidden_size, output_size, seq_length)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

# 메인 실행
if __name__ == "__main__":
    # 파라미터 설정
    seq_length = 30
    steps = 25  # 30일 예측
    input_size = 7  # 7개의 feature
    hidden_size = 10
    output_size = 1
    model_path = f"./checkpoints/train_with_100Epochs.pt"  # 학습된 모델의 경로

    # 데이터 준비
    df = get_bitcoin_data()
    data, scaler = scale(df.values)
    # 모델 로드
    model = load_model(model_path, input_size, hidden_size, output_size, seq_length)
    
    # 30일 예측
    future_prices = predict_future_prices(model, data, seq_length, steps)
    future_prices = inverse_scale(future_prices.reshape(-1, 1), scaler)
    
    # 결과 출력    
    start_time = datetime.strptime("2024-11-21", "%Y-%m-%d")
    end_time = datetime.strptime("2024-12-16", "%Y-%m-%d")
    df = yf.download("BTC-USD", start = start_time, end = end_time, interval="1d")
    df_ex = yf.download("KRW=X", start = start_time, end = end_time, interval="1d")
    
    df_with_pred = df['Close']
    df_with_pred.columns = ['Close']
    df_with_pred['Pred'] = future_prices.reshape(-1, 1)
    df_with_pred['Exchange'] = df_ex['Close']
    df_with_pred = df_with_pred.ffill()
    df_with_pred['Close_won'] = df_with_pred['Close'] * df_with_pred['Exchange']
    df_with_pred['Pred_won'] = df_with_pred['Pred'] * df_with_pred['Exchange']
    
    print(df_with_pred)
    print(f"MAE : {mean_absolute_error(df_with_pred['Close'], df_with_pred['Pred'])}")
    print(f"MSE : {mean_squared_error(df_with_pred['Close'], df_with_pred['Pred'])}")
    
    print(f"MAE(Won) : {mean_absolute_error(df_with_pred['Close'], df_with_pred['Pred'])}")
    print(f"MSE(Won) : {mean_squared_error(df_with_pred['Close_won'], df_with_pred['Pred_won'])}")