import os
import pandas as pd
import numpy as np
# pyrefly: ignore [missing-import]
import ta

# Paths
root = os.path.expanduser('~/Desktop/yzproje')
raw_path = os.path.join(root, 'data', 'raw', 'bitcoin', 'btcusd_1-min_data.csv')
out_dir = os.path.join(root, 'data', 'processed')
out_file = os.path.join(out_dir, 'bitcoin_hourly_features.csv')

os.makedirs(out_dir, exist_ok=True)

print('Reading', raw_path)
# Read CSV (infer timestamp column)
df = pd.read_csv(raw_path)
print('Loaded rows:', len(df))

# Show initial info
print('\nFirst 5 rows:')
print(df.head().to_string())
print('\nColumns:')
print(df.columns.tolist())
print('\nMissing counts:')
print(df.isna().sum())

# Expect a timestamp column - try common names
ts_col = None
for c in ['Timestamp', 'timestamp', 'date', 'time', 'Date']:
    if c in df.columns:
        ts_col = c
        break
if ts_col is None:
    # If there's column named 'Unix' or first column looks like integer epoch
    ts_col = df.columns[0]
    print('Using first column as timestamp:', ts_col)

# Convert to datetime
try:
    df[ts_col] = pd.to_datetime(df[ts_col], unit='s')
except Exception:
    df[ts_col] = pd.to_datetime(df[ts_col])

# Set index
df = df.set_index(ts_col)

# Ensure numeric columns
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
print('\nNumeric columns:', numeric_cols)

# Resample to hourly frequency
df_hour = df.resample('H').agg({
    'Open':'first',
    'High':'max',
    'Low':'min',
    'Close':'last',
    'Volume':'sum'
})

# If those column names don't exist, try fallback using numeric columns
if df_hour.isna().all().all():
    # Generic resample using numeric columns
    df_hour = df.resample('H').mean()

# Forward fill missing
df_hour = df_hour.ffill()

# Rename columns to standard if possible
cols_lower = {c:c.lower() for c in df_hour.columns}

# Compute returns
if 'Close' in df_hour.columns:
    price_col = 'Close'
elif 'close' in [c.lower() for c in df_hour.columns]:
    # find actual name
    price_col = [c for c in df_hour.columns if c.lower()=='close'][0]
else:
    # fallback to first numeric
    price_col = df_hour.select_dtypes(include=[np.number]).columns[0]

# Ensure ta works with expected column names
data_for_ta = df_hour.copy()
# normalize to lower-case columns required by ta
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

# Compute RSI
if 'close' in data_for_ta.columns:
    data_for_ta['rsi'] = ta.momentum.RSIIndicator(data_for_ta['close']).rsi()
# MACD and signal
if 'close' in data_for_ta.columns:
    macd = ta.trend.MACD(data_for_ta['close'])
    data_for_ta['macd'] = macd.macd()
    data_for_ta['macd_signal'] = macd.macd_signal()
# Bollinger
if 'close' in data_for_ta.columns:
    bb = ta.volatility.BollingerBands(data_for_ta['close'])
    data_for_ta['bollinger_h'] = bb.bollinger_hband()
    data_for_ta['bollinger_l'] = bb.bollinger_lband()

# Return column
if 'close' in data_for_ta.columns:
    data_for_ta['return'] = data_for_ta['close'].pct_change()
else:
    data_for_ta['return'] = data_for_ta[price_col].pct_change()

# Future 24 hour volatility target: rolling std window 24 shifted -24
data_for_ta['future_volatility'] = data_for_ta['return'].rolling(window=24).std().shift(-24)

# Drop NaNs
data_clean = data_for_ta.dropna()

# Save
print('\nSaving to', out_file)
data_clean.to_csv(out_file)

print('\nSaved. Result info:')
print('head:\n', data_clean.head().to_string())
print('\ncolumns:\n', data_clean.columns.tolist())
print('\nshape:\n', data_clean.shape)
print('\nFile exists:', os.path.exists(out_file))
