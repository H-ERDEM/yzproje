#!/usr/bin/env python3

import os
import sys
import logging
import joblib
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.expanduser('~/Desktop/yzproje')
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
OUT_DIR = os.path.join(ROOT, 'outputs')
MODEL_FILE = os.path.join(MODEL_DIR, 'cnn_bilstm_volatility_model_pytorch.pt')
RESULTS_FILE = os.path.join(OUT_DIR, 'cnn_bilstm_results.csv')
LOSS_PLOT = os.path.join(OUT_DIR, 'cnn_bilstm_training_loss.png')

FEATURE_COLS = [
    'open','high','low','close','volume','rsi','macd','macd_signal','bollinger_h','bollinger_l','return',
    'sentiment_score','tweet_count','likes','retweets','weighted_sentiment','atr','vwap'
]
TARGET_COL = 'future_volatility'

WINDOW = 12
TAIL_N = 30000
EPOCHS = 10
BATCH_SIZE = 64
LR = 0.001


class SequenceDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).float().unsqueeze(1)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


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
        out, (hn, cn) = self.bilstm(x)
        last = out[:, -1, :]
        return self.fc(last)


def load_prepare(path):
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
    df = load_prepare(DATA_PATH)
    logger.info('Prepared df shape: %s', df.shape)

    features = df[FEATURE_COLS].values
    target = df[TARGET_COL].values.reshape(-1,1)


    split_idx = int(len(features) * 0.8)
    feat_scaler = MinMaxScaler()
    targ_scaler = MinMaxScaler()
    feat_scaler.fit(features[:split_idx])
    targ_scaler.fit(target[:split_idx])


    features_scaled = feat_scaler.transform(features)
    target_scaled = targ_scaler.transform(target)

    X, y = create_sequences(features_scaled, target_scaled.flatten(), WINDOW)
    logger.info('Sequences X shape: %s, y shape: %s', X.shape, y.shape)

    n = len(X)
    split = int(n * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    train_ds = SequenceDataset(X_train, y_train)
    test_ds = SequenceDataset(X_test, y_test)
    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)

    if torch.backends.mps.is_available():
        device = torch.device('mps')
    elif torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    logger.info('Using device: %s', device)

    model = CNNBiLSTM(input_features=X.shape[2], conv_out=32, lstm_hidden=32, lstm_layers=1).to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)

    train_losses = []
    for epoch in range(1, EPOCHS+1):
        model.train()
        epoch_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        train_losses.append(mean_loss)
        print(f'Epoch {epoch}/{EPOCHS} - train loss: {mean_loss:.6f}')


    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save(model.state_dict(), MODEL_FILE)


    model.eval()
    preds = []
    trues = []
    with torch.no_grad():
        for xb, yb in test_loader:
            xb = xb.to(device)
            out = model(xb).cpu().numpy()
            preds.append(out)
            trues.append(yb.cpu().numpy())
    preds = np.vstack(preds).flatten()
    trues = np.vstack(trues).flatten()

    preds_inv = targ_scaler.inverse_transform(preds.reshape(-1,1)).flatten()
    trues_inv = targ_scaler.inverse_transform(trues.reshape(-1,1)).flatten()

    mae = mean_absolute_error(trues_inv, preds_inv)
    rmse = math.sqrt(mean_squared_error(trues_inv, preds_inv))
    r2 = r2_score(trues_inv, preds_inv)


    os.makedirs(OUT_DIR, exist_ok=True)
    results_df = pd.DataFrame([{'model':'CNN_BiLSTM','MAE':mae,'RMSE':rmse,'R2':r2}])
    results_df.to_csv(RESULTS_FILE, index=False)


    plt.figure()
    plt.plot(range(1, EPOCHS+1), train_losses, marker='o')
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('CNN+BiLSTM Training Loss')
    plt.grid(True)
    plt.savefig(LOSS_PLOT)


    print('\nX_train shape:', X_train.shape)
    print('X_test shape:', X_test.shape)
    print('feature sayısı:', X_train.shape[2])
    print('MAE:', mae)
    print('RMSE:', rmse)
    print('R2:', r2)
    print('model dosyası var mı?:', os.path.exists(MODEL_FILE))
    print('results dosyası var mı?:', os.path.exists(RESULTS_FILE))


if __name__ == '__main__':
    main()
