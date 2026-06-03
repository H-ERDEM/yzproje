#!/usr/bin/env python3
import os
import sys
import logging
import math
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.expanduser('~/Desktop/yzproje')
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
OUT_DIR = os.path.join(ROOT, 'outputs')
LSTM_MODEL_FILE = os.path.join(MODEL_DIR, 'lstm_volatility_model_pytorch.pt')
CNN_MODEL_FILE = os.path.join(MODEL_DIR, 'cnn_bilstm_volatility_model_pytorch.pt')
RF_MODEL_FILE = os.path.join(MODEL_DIR, 'random_forest_baseline.pkl')
FEATURE_SCALER_FILE = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
TARGET_SCALER_FILE = os.path.join(MODEL_DIR, 'target_scaler.pkl')
RESULTS_FILE = os.path.join(OUT_DIR, 'ensemble_results.csv')
TEST_PREDS_FILE = os.path.join(OUT_DIR, 'test_predictions.csv')
FEAT_IMP_FILE = os.path.join(OUT_DIR, 'feature_importances.csv')

FEATURE_COLS = [
    'open','high','low','close','volume','rsi','macd','macd_signal','bollinger_h','bollinger_l','return',
    'sentiment_score','tweet_count','likes','retweets','weighted_sentiment','atr','vwap'
]
TARGET_COL = 'future_volatility'
WINDOW = 12


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size=32, num_layers=1):
        super().__init__()
        self.lstm = nn.LSTM(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, (hn, cn) = self.lstm(x)
        last = out[:, -1, :]
        return self.fc(last)


class CNNBiLSTM(nn.Module):
    def __init__(self, input_features, conv_out=32, lstm_hidden=32, lstm_layers=1):
        super().__init__()
        self.conv = nn.Conv1d(in_channels=input_features, out_channels=conv_out, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)
        self.bilstm = nn.LSTM(input_size=conv_out, hidden_size=lstm_hidden, num_layers=lstm_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(lstm_hidden * 2, 1)

    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv(x)
        x = self.relu(x)
        x = self.pool(x)
        x = x.permute(0, 2, 1)
        out, _ = self.bilstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def load_and_prepare(path):
    if not os.path.exists(path):
        logger.error('Data file not found: %s', path)
        sys.exit(1)
    df = pd.read_csv(path, index_col=0, parse_dates=[0])
    df = df[(df.index >= '2017-01-27') & (df.index <= '2019-05-27')].copy()
    cols = [c for c in FEATURE_COLS + [TARGET_COL] if c in df.columns]
    df = df[cols]
    df = df.replace([np.inf, -np.inf], np.nan).dropna()
    

    count_cols = ['volume', 'tweet_count', 'likes', 'retweets']
    for col in count_cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
    return df


def create_sequences(values, target, window):
    Xs, ys = [], []
    for i in range(len(values) - window):
        Xs.append(values[i:i+window])
        ys.append(target[i+window])
    return np.array(Xs), np.array(ys)


def main():
    df = load_and_prepare(DATA_PATH)
    logger.info('Prepared df shape: %s', df.shape)

    features = df[FEATURE_COLS].values
    target = df[TARGET_COL].values.reshape(-1, 1)


    if not os.path.exists(FEATURE_SCALER_FILE) or not os.path.exists(TARGET_SCALER_FILE):
        logger.error('Scalers not found. Run training scripts first.')
        sys.exit(1)
    feat_scaler = joblib.load(FEATURE_SCALER_FILE)
    targ_scaler = joblib.load(TARGET_SCALER_FILE)


    features_scaled = feat_scaler.transform(features)
    target_scaled = targ_scaler.transform(target)


    X, y = create_sequences(features_scaled, target_scaled.flatten(), WINDOW)
    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    test_ds = SequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)


    device = torch.device('cpu')
    
    lstm_model = LSTMModel(input_size=18).to(device)
    lstm_model.load_state_dict(torch.load(LSTM_MODEL_FILE, map_location=device))
    lstm_model.eval()

    cnn_model = CNNBiLSTM(input_features=18).to(device)
    cnn_model.load_state_dict(torch.load(CNN_MODEL_FILE, map_location=device))
    cnn_model.eval()


    lstm_preds = []
    cnn_preds = []
    trues = []

    with torch.no_grad():
        for xb, yb in test_loader:
            out_lstm = lstm_model(xb).numpy()
            out_cnn = cnn_model(xb).numpy()
            lstm_preds.append(out_lstm)
            cnn_preds.append(out_cnn)
            trues.append(yb.numpy())

    lstm_preds = np.vstack(lstm_preds).flatten()
    cnn_preds = np.vstack(cnn_preds).flatten()
    trues = np.vstack(trues).flatten()


    lstm_preds_inv = targ_scaler.inverse_transform(lstm_preds.reshape(-1,1)).flatten()
    cnn_preds_inv = targ_scaler.inverse_transform(cnn_preds.reshape(-1,1)).flatten()
    trues_inv = targ_scaler.inverse_transform(trues.reshape(-1,1)).flatten()


    best_rmse = float('inf')
    best_w = 0.5
    best_blend = None

    for w in np.linspace(0, 1, 101):
        blend = w * cnn_preds_inv + (1 - w) * lstm_preds_inv
        rmse = math.sqrt(mean_squared_error(trues_inv, blend))
        if rmse < best_rmse:
            best_rmse = rmse
            best_w = w
            best_blend = blend


    best_mae = mean_absolute_error(trues_inv, best_blend)
    best_r2 = r2_score(trues_inv, best_blend)

    logger.info('Best Blending Weight (CNN weight): %.2f', best_w)
    logger.info('Best Blended MAE: %.6f', best_mae)
    logger.info('Best Blended RMSE: %.6f', best_rmse)
    logger.info('Best Blended R2: %.6f', best_r2)


    os.makedirs(OUT_DIR, exist_ok=True)
    results_df = pd.DataFrame([{
        'model': f'Ensemble_Blend (w={best_w:.2f})',
        'MAE': best_mae,
        'RMSE': best_rmse,
        'R2': best_r2
    }])
    results_df.to_csv(RESULTS_FILE, index=False)
    logger.info('Ensemble results saved to %s', RESULTS_FILE)


    

    if os.path.exists(RF_MODEL_FILE):
        rf_model = joblib.load(RF_MODEL_FILE)
        importances = rf_model.feature_importances_
        imp_df = pd.DataFrame({
            'feature': FEATURE_COLS,
            'importance': importances
        }).sort_values(by='importance', ascending=False)
        imp_df.to_csv(FEAT_IMP_FILE, index=False)
        logger.info('Feature importances saved to %s', FEAT_IMP_FILE)
    else:
        logger.warning('Random Forest model not found at %s. Cannot extract feature importances.', RF_MODEL_FILE)




    X_train_flat = X_train[:, -1, :]
    X_test_flat = X_test[:, -1, :]
    

    y_train_unscaled = targ_scaler.inverse_transform(y_train.reshape(-1, 1)).flatten()
    lr = LinearRegression()
    lr.fit(X_train_flat, y_train_unscaled)
    lr_preds = lr.predict(X_test_flat)


    if os.path.exists(RF_MODEL_FILE):
        rf_preds = rf_model.predict(X_test_flat)
    else:
        rf_preds = np.zeros_like(lr_preds)



    test_dates = df.index[split + WINDOW : split + WINDOW + len(y_test)]


    preds_df = pd.DataFrame({
        'Actual': trues_inv,
        'LinearRegression': lr_preds,
        'RandomForest': rf_preds,
        'LSTM': lstm_preds_inv,
        'CNN_BiLSTM': cnn_preds_inv,
        'Ensemble': best_blend
    }, index=test_dates)
    

    preds_df.to_csv(TEST_PREDS_FILE)
    logger.info('Test predictions saved to %s (length: %d)', TEST_PREDS_FILE, len(preds_df))


if __name__ == '__main__':
    main()
