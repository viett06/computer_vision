#Unsupervised Learning Trading Strategy
# Download/Load SP500 stocks prices data.
# Calculate different features and indicators on each stock.
# Aggregate on monthly level and filter top 150 most liquid stocks.
# Calculate Monthly Returns for different time-horizons.
# Download Fama-French Factors and Calculate Rolling Factor Betas.
# For each month fit a K-Means Clustering Algorithm to group similar assets based on their features.
# For each month select assets based on the cluster and form a portfolio based on Efficient Frontier max sharpe ratio optimization.
# Visualize Portfolio returns and compare to SP500 returns.

#Download/Load SP500 stocks prices data.
from statsmodels.regression.rolling import RollingOLS
import pandas_datareader.data as web
import matplotlib.pyplot as plt
import statsmodels.api as sm
import pandas as pd
import numpy as np
import datetime as dt
import yfinance as yf
import warnings
import requests
import io

warnings.filterwarnings('ignore')

wiki_url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
headers = {'User-Agent': 'Mozilla/5.0'}
html = requests.get(wiki_url, headers=headers, timeout=10).text
sp500 = pd.read_html(io.StringIO(html))[0]
sp500['Symbol'] = sp500['Symbol'].str.replace('.', '-')
symbols_list = sp500['Symbol'].unique().tolist()
end_date = '2023-09-27'
start_date = pd.to_datetime(end_date) - pd.DateOffset(365*8)
df = yf.download(tickers = symbols_list, start = start_date, end = end_date).stack()
df.index.names = ['date', 'ticker']
df.columns = df.columns.str.lower()

pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)

print(df.shape)
print(df.head(20))

# 2. Calculate features and technical indicators for each stock.
# Garman-Klass Volatility
# RSI
# Bollinger Bands
# ATR
# MACD
# Dollar Volume

# Garman-Klass Volatility = (ln(High) - ln(Low))^2 / 2 - (2*ln(2) - 1) * (ln(Adj Close) - ln(Open))^2

df['garman_klass_vol'] = ((np.log(df['high'])- np.log(df['low']))**2)/2 - ((2*np.log(2)-1) * (np.log(df['adj close'])- np.log(df['open']))**2)


def rsi(close, length=20):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/length, min_periods=length, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


df['rsi'] = df.groupby(level=1)['adj close'].transform(lambda x: rsi(x, length=20))


def bbands(close, length=20, std=2):
    mid = close.rolling(length).mean()
    dev = std * close.rolling(length).std()
    lower = mid - dev
    upper = mid + dev
    return lower, mid, upper


df['bb_low'] = df.groupby(level=1)['adj close'].transform(lambda x: bbands(x, length=20, std=2).iloc[:, 0])
df['bb_mid'] = df.groupby(level=1)['adj close'].transform(lambda x: bbands(x, length=20, std=2).iloc[:, 1])
df['bb_high'] = df.groupby(level=1)['adj close'].transform(lambda x: bbands(x, length=20, std=2).iloc[:, 2])

def compute_atr(stock_data):
    high = stock_data['high']
    low = stock_data['low']
    close = stock_data['adj close']
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
    return atr.sub(atr.mean()).div(atr.std())


df['atr'] = df.groupby(level=1, group_keys=False).apply(compute_atr)

def compute_macd(close):
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    macd = ema_12 - ema_26
    return macd.sub(macd.mean()).div(macd.std())


df['macd'] = df.groupby(level=1)['adj close'].transform(compute_macd)

df['macd'] = df.groupby(level=1, group_keys=False)['adj close'].apply(compute_macd)

df['dollar_volume'] = (df['adj close']*df['volume'])/1e6

df
