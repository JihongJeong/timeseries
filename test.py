import yfinance as yf

df = yf.download("KRW=X")

print(df)