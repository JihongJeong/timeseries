import argparse
import pandas as pd
import itertools
import json
from data_preprocessing import load_data, plot_data, scale, inverse_scale
from statsmodels.tsa.stattools import adfuller
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.arima.model import ARIMA

import warnings
warnings.filterwarnings(action='ignore')

parser = argparse.ArgumentParser()
parser.add_argument('-m', dest = 'mode', action = 'store', default = 'train/test', help = 'available modes : train/test, inference')
parser.add_argument('-lp', dest = 'latest_price', type = float, action = 'store', help = 'latest price to decide action')
info = parser.parse_args()

# Load bitcoin daily price data from yfinance library
# Calculate technical indicator(Simple Moving Average, RSI, Bollinger Band, MACD)
# Get Close price and technical indicators as features
df_original, features = load_data()
index = df_original.index
df_scaled, scaler = scale(df_original)
df = pd.DataFrame(df_scaled, columns=features, index = index)

# Apply frequency as daily
df.index = pd.to_datetime(df.index)
df = df.resample('D').asfreq()
df = df.interpolate(method = 'linear')

# Check Stationary of data
# If its non-stationary, appply differencing
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

# Use technical features as exogeneous
exog = df[features[1:]]
exog = exog.dropna()

# Use Close price as a target
y = df['Close'][len(df)-len(exog):]

if info.mode == 'train/test':
    # Test/train split
    len_train = int(len(y)*0.8)
    len_test = len(y) - len_train
    y_train, exog_train = y[:len_train], exog[:len_train]
    y_test, exog_test = y[len_train:], exog[len_train:]

    # Find optimal p, d, q value of ARIMA model
    p = range(0, 3)
    d = [1]
    q = range(0, 3)

    pdq = list(itertools.product(p, d, q))
    result = {}
    for i in pdq:
        model = ARIMA(y_train, order = i, exog = exog_train)
        model_fit = model.fit()
        # print(f"PDQ : {i}, AIC : {model_fit.aic}")
        result[i] = model_fit.aic

    result = sorted(result.items(), key = lambda item: item[1])
    optimal_pdq = result[0][0]
    with open("optimal_pdq.json", "w") as pdq_json:
        json.dump({"optimal_pdq" : optimal_pdq}, pdq_json)
    
    # Build ARIMA model using optimal p, d, q value
    model = ARIMA(y_train, order = optimal_pdq, exog = exog_train)
    model_fit = model.fit()
    print(optimal_pdq)
    print(model_fit.summary())

    # Predict Close price 
    forecast = model_fit.forecast(steps = len_test, exog = exog_test)
    predicted = forecast.to_frame()
    predicted.columns = ["Forecast"]
    forecasted_data = predicted["Forecast"].to_numpy().reshape(-1, 1)

    # Restore the scale
    original_scale = inverse_scale(forecasted_data, scaler = scaler)
    predicted["Predict"] = original_scale
    df_pred = pd.concat([df_original["Close"], predicted["Predict"]], axis = 1)
    df_pred.columns = ["Close", "Predict"]

    # Plot predicted price
    plot_data(df_pred, with_prediction=True)
    
elif info.mode == 'inference':
    with open("optimal_pdq.json", "r") as pdq_json:
        optimal_pdq = json.load(pdq_json)
    model = ARIMA(y, order = optimal_pdq["optimal_pdq"], exog = exog)
    model_fit = model.fit()
    forecast = model_fit.forecast(steps = 1, exog = exog.iloc[-1, :]).to_list()
    result = inverse_scale([forecast], scaler)
    
    prev = info.latest_price if info.latest_price else df_original['Close'].values[-1][0]
    pred = result[0][0]
    
    change = (pred - prev)/prev * 100
    if change > 2:
        action = 'Buy'
    elif change < -2:
        action = 'Sell'
    else:
        action = 'Hold'
    
    print(f"{action}, predicted increase/decrease rate is {change:.2f}%")
    
else:
    print("Wrong parser!")