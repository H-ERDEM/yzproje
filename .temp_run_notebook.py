import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import ta

from sklearn import preprocessing



root = os.path.expanduser('~/Desktop/yzproje')

raw_file = os.path.join(root, 'data', 'raw', 'bitcoin', 'btcusd_1-min_data.csv')

out_dir = os.path.join(root, 'data', 'processed')

out_file = os.path.join(out_dir, 'bitcoin_hourly_features.csv')

os.makedirs(out_dir, exist_ok=True)



print('Reading', raw_file)

df = pd.read_csv(raw_file)

print('Loaded rows:', len(df))



print('

First 5 rows:' )

print('

Columns:' )

print('

Missing counts:' )

print(df.columns.tolist())

print('

print(df.isna().sum())

markdown

#VSC-e35c2245

markdown

# Price data preprocessing

Bu notebook BTC/USD bir-dakikalık verisini saatlik özelliklere çevirir ve teknik indikatörler ekler.

code

#VSC-7281dd75

python

import os

import pandas as pd

import numpy as np

import matplotlib.pyplot as plt

import ta

from sklearn import preprocessing



root = os.path.expanduser('~/Desktop/yzproje')

raw_file = os.path.join(root, 'data', 'raw', 'bitcoin', 'btcusd_1-min_data.csv')

out_dir = os.path.join(root, 'data', 'processed')

out_file = os.path.join(out_dir, 'bitcoin_hourly_features.csv')

os.makedirs(out_dir, exist_ok=True)



print('Reading', raw_file)

df = pd.read_csv(raw_file)

print('Loaded rows:', len(df))



print('\nFirst 5 rows:' )

print(df.head().to_string())

print('\nColumns:' )

print(df.columns.tolist())

print('\nMissing counts:' )

print(df.isna().sum())



# Timestamp kolonu tespit etme

ts_col = None

for c in ['Timestamp', 'timestamp', 'date', 'time', 'Date']:

    if c in df.columns:

        ts_col = c

        break

if ts_col is None:

    ts_col = df.columns[0]

    print('Using first column as timestamp:', ts_col)



# Datetime dönüşümü

try:

    df[ts_col] = pd.to_datetime(df[ts_col], unit='s')

except Exception:

    df[ts_col] = pd.to_datetime(df[ts_col])



# Index olarak ayarla

df = df.set_index(ts_col)



# Saatlik frekansa dönüştürme

# Eğer beklenen sütun isimleri varsa açıkça aggregate et

if set(['Open','High','Low','Close','Volume']).issubset(df.columns):

    df_hour = df.resample('H').agg({'Open':'first','High':'max','Low':'min','Close':'last','Volume':'sum'})

else:

    df_hour = df.resample('H').mean()



# Eksikleri forward fill ile doldur

df_hour = df_hour.ffill()



# Teknik indikatörler için ta kütüphanesi ile çalışacağımız uygun kolon isimleri

data_for_ta = df_hour.copy()

renames = {}

if 'Open' in data_for_ta.columns:

    renames['Open']='open'

if 'High' in data_for_ta.columns:

    renames['High']='high'

if 'Low' in data_for_ta.columns:

    renames['Low']='low'

if 'Close' in data_for_ta.columns:

    renames['Close']='close'

if 'Volume' in data_for_ta.columns:

    renames['Volume']='volume'

if renames:

    data_for_ta = data_for_ta.rename(columns=renames)



# RSI

if 'close' in data_for_ta.columns:

    data_for_ta['rsi'] = ta.momentum.RSIIndicator(data_for_ta['close']).rsi()

# MACD ve signal

if 'close' in data_for_ta.columns:

    macd = ta.trend.MACD(data_for_ta['close'])

    data_for_ta['macd'] = macd.macd()

    data_for_ta['macd_signal'] = macd.macd_signal()

# Bollinger band

if 'close' in data_for_ta.columns:

    bb = ta.volatility.BollingerBands(data_for_ta['close'])

    data_for_ta['bollinger_h'] = bb.bollinger_hband()

    data_for_ta['bollinger_l'] = bb.bollinger_lband()



# Return hesaplama

if 'close' in data_for_ta.columns:

    data_for_ta['return'] = data_for_ta['close'].pct_change()

else:

    data_for_ta['return'] = data_for_ta[data_for_ta.select_dtypes(include=[np.number]).columns[0]].pct_change()



# future volatility: rolling std window 24 shift -24

data_for_ta['future_volatility'] = data_for_ta['return'].rolling(window=24).std().shift(-24)



# NaN sil

data_clean = data_for_ta.dropna()



# Kaydet

data_clean.to_csv(out_file)

print('Saved to', out_file)

code

#VSC-ccb49490

python

# Sonuçları yükleyip göster

import os

df = pd.read_csv(out_file, parse_dates=True, index_col=0)

print('df.head():')

print(df.head().to_string())

print('













54print('

File exists:', os.path.exists(out_file))print(df.shape)print('

 df.shape:')print(df.columns.tolist()) df.columns:')