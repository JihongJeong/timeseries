import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from MyModel import LSTMModel
from ta.momentum import RSIIndicator, StochasticOscillator
from ta.trend import MACD, SMAIndicator
from ta.volatility import BollingerBands
from sklearn.preprocessing import MinMaxScaler

# 초기 설정
initial_balance = 10000  # 초기 자본 (KRW)
btc_balance = 0  # 초기 BTC 보유량
last_trade_price = None  # 직전 거래 가격
trade_log = []  # 거래 기록 저장
threshold = 0.002  # 2% 변화 기준

# 데이터 가져오기
def fetch_historical_data(start_time, end_time):
    # 특정 시간 범위 데이터 가져오기
    data = yf.download(tickers="BTC-USD", start=start_time, end=end_time, interval="1m")
    return data

def preprocess_data(data):
    df = data
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

# 모델을 통해 가격 예측
def predict_price(model, data, features):
    """
    LSTM 모델로 1분 후의 가격을 예측하는 함수
    """
    # 데이터 준비
    input_data = data[features].values[-30:]  # 최근 30개 데이터
    mean = np.mean(input_data, axis=0)
    std = np.std(input_data, axis=0)
    scaled_data = (input_data - mean) / std  # 정규화
    
    # LSTM 입력 형식으로 변환
    input_data = torch.tensor(scaled_data.reshape(1, 30, len(features)), dtype = torch.float32)  # (배치 크기, 시간 창, 특징 개수)

    # 예측 실행
    predicted_price = model(input_data)
    result_scaled = np.array(predicted_price.detach())
    result = result_scaled*std[0] + mean[0]
    
    return result[0][0]


# 거래 함수
def execute_trade(action, price, balance, btc_balance):
    global last_trade_price

    if action == "buy":
        btc_to_buy = balance / price
        btc_balance += btc_to_buy * (1-0.05)
        balance = 0
        last_trade_price = price
        print(f"BUY: {btc_to_buy:.6f} BTC at ${price:.2f}")
    elif action == "sell":
        balance += btc_balance * price * (1-0.05)
        btc_balance = 0
        last_trade_price = price
        print(f"SELL: {balance:.2f} KRW at ${price:.2f}")
    else:
        print(f"HOLD at ${price:.2f}")

    return balance, btc_balance

# 거래 시뮬레이션
start_time = datetime.strptime("2024-12-17 17:30:00", "%Y-%m-%d %H:%M:%S")
test_time = datetime.strptime("2024-12-17 19:00:00", "%Y-%m-%d %H:%M:%S")
end_time = datetime.strptime("2024-12-17 21:30:00", "%Y-%m-%d %H:%M:%S")

# 데이터 가져오기
data = fetch_historical_data(start_time, end_time)
data, feature = preprocess_data(data)
data = data.loc["2024-12-17 19:00:00":"2024-12-17 21:30:00"]


# LSTM 모델 불러오기 (학습된 모델 경로를 지정하세요)
lstm = LSTMModel(input_dim=len(feature), hidden_dim=10, output_dim=1,seq_len=30)
checkpoint = f"./checkpoints/train_with_100Epochs.pt"
lstm.load_state_dict(torch.load(checkpoint))

# 데이터가 충분한지 확인
if len(data) < 30:
    print("Not enough data for simulation. Ensure at least 30 minutes of data is available.")
else:
    print(f"Fetched {len(data)} rows of data.")

    # 데이터프레임으로 거래 진행
    for i in range(len(data) - 30):  # 30분 후부터 데이터 처리
        print(f"\nIteration {i + 1}")
        # 슬라이딩 윈도우로 데이터 준비
        window_data = data.iloc[i:i + 30]  # 30개 가격 데이터
        
        # 현재 가격
        current_price = data['Close'].iloc[i + 30]
        
        # 가격 예측
        predicted_price = predict_price(lstm, window_data, feature)
        print(f"Predicted Price: ${predicted_price:.2f}, Current Price: ${current_price.values[0]:.2f}")
        
        # 초기 거래 가격 설정
        if last_trade_price is None:
            last_trade_price = current_price.values[0]
        
        # 거래 전략 결정
        if predicted_price > last_trade_price * (1 + threshold):
            action = "buy"
        elif predicted_price < last_trade_price * (1 - threshold):
            action = "sell"
        else:
            action = "hold"
        
        # 거래 실행
        initial_balance, btc_balance = execute_trade(action, current_price.values[0], initial_balance, btc_balance)
        
        # 거래 기록 저장
        trade_log.append({
            "iteration": i + 1,
            "action": action,
            "price": current_price.values[0],
            "balance": initial_balance + current_price.values[0] * btc_balance,
            "btc_balance": btc_balance,
            "timestamp": data.index[i + 30]
        })


# 거래 로그 저장
trade_log_df = pd.DataFrame(trade_log)
trade_log_df.to_csv("mock_trade_log.csv", index=False)
print("Trading completed. Log saved to 'mock_trade_log.csv'.")
