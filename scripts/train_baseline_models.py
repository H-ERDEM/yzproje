#!/usr/bin/env python3

import os
import sys
import logging
import joblib
import numpy as np
import pandas as pd

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.expanduser('~/Desktop/yzproje')
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')
OUT_DIR = os.path.join(ROOT, 'outputs')
MODEL_DIR = os.path.join(ROOT, 'models')
RESULTS_FILE = os.path.join(OUT_DIR, 'baseline_results.csv')
RF_MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_baseline.pkl')

FEATURE_COLS = [
    'open','high','low','close','volume','rsi','macd','macd_signal','bollinger_h','bollinger_l','return',
    'sentiment_score','tweet_count','likes','retweets'
]
TARGET_COL = 'future_volatility'


def safe_load_data(path):
    if not os.path.exists(path):
        logger.error('Data file not found: %s', path)
        sys.exit(1)
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    return df


def clean_df(df):
    # Keep only required columns if present
    keep = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    df = df[keep].copy()
    # Replace inf with NaN, then drop rows with NaN in target or features
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[TARGET_COL])
    df = df.dropna()
    return df


def train_and_eval(X_train, y_train, X_test, y_test):
    results = []

    # Linear Regression
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    mae_lr = mean_absolute_error(y_test, y_pred_lr)
    rmse_lr = float(np.sqrt(mean_squared_error(y_test, y_pred_lr)))
    r2_lr = r2_score(y_test, y_pred_lr)
    results.append({'model':'LinearRegression','MAE':mae_lr,'RMSE':rmse_lr,'R2':r2_lr})

    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mae_rf = mean_absolute_error(y_test, y_pred_rf)
    rmse_rf = float(np.sqrt(mean_squared_error(y_test, y_pred_rf)))
    r2_rf = r2_score(y_test, y_pred_rf)
    results.append({'model':'RandomForestRegressor','MAE':mae_rf,'RMSE':rmse_rf,'R2':r2_rf})

    return results, rf


def main():
    df = safe_load_data(DATA_PATH)
    logger.info('Loaded data shape: %s', df.shape)

    df_clean = clean_df(df)
    logger.info('After cleaning shape: %s', df_clean.shape)

    # train/test split sequentially (no shuffle)
    n = len(df_clean)
    split = int(n * 0.8)
    X = df_clean[FEATURE_COLS]
    y = df_clean[TARGET_COL]

    X_train = X.iloc[:split].values
    X_test = X.iloc[split:].values
    y_train = y.iloc[:split].values
    y_test = y.iloc[split:].values

    # Scale features using MinMaxScaler fitted on train only
    scaler = MinMaxScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    results, rf_model = train_and_eval(X_train_scaled, y_train, X_test_scaled, y_test)

    # Prepare outputs
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_FILE, index=False)

    # Save Random Forest model
    joblib.dump(rf_model, RF_MODEL_FILE)

    # Print final requested info
    print('train shape:', X_train.shape)
    print('test shape:', X_test.shape)
    print('feature sayısı:', X_train.shape[1])
    print('baseline_results.csv var mı?:', os.path.exists(RESULTS_FILE))
    print('model dosyası var mı?:', os.path.exists(RF_MODEL_FILE))
    print('\nResults:\n', results_df.to_string(index=False))


if __name__ == '__main__':
    main()
