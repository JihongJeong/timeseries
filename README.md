## TimeSeries Analysis

## Bitcoin price prediction using LSTM

## How to Use
Train LSTM model
```
python lstm_prediction.py -m train/test
```

Decide actions base on your latest bitcoin price
```
python lstm_prediction.py -m inference -lp [your latest price]
```
For example, if you bought 1 bitcoin with 90000$, type 
```
python lstm_prediction.py -m inference -lp 90000
```
to see how bitcoin price increased or decreased and suggested action to 'Buy', 'Hold' or 'Sell' 