#!/usr/bin/env python3
"""Create project visualizations and save them under outputs/visualizations."""
import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.expanduser('~/Desktop/yzproje')
OUT_DIR = os.path.join(ROOT, 'outputs')
VIS_DIR = os.path.join(OUT_DIR, 'visualizations')
os.makedirs(VIS_DIR, exist_ok=True)

def read_results(fname):
    path = os.path.join(OUT_DIR, fname)
    if os.path.exists(path):
        return pd.read_csv(path)
    logger.warning('Results file not found: %s', path)
    return None


def plot_metric_comparison(dfs, metric, outname):
    plt.figure()
    models = []
    values = []
    for name, df in dfs.items():
        if df is None:
            continue
        if metric in df.columns:
            models.append(name)
            values.append(df[metric].iloc[0])
    if not models:
        logger.warning('No data for metric %s', metric)
        return
    plt.bar(models, values)
    plt.ylabel(metric)
    plt.title(f'{metric} comparison')
    plt.tight_layout()
    outpath = os.path.join(VIS_DIR, outname)
    plt.savefig(outpath)
    plt.close()
    logger.info('Saved %s', outpath)


def plot_timeseries(df, col, outname, title=None):
    plt.figure(figsize=(12,4))
    plt.plot(df.index, df[col])
    plt.title(title or col)
    plt.xlabel('Date')
    plt.ylabel(col)
    plt.tight_layout()
    outpath = os.path.join(VIS_DIR, outname)
    plt.savefig(outpath)
    plt.close()
    logger.info('Saved %s', outpath)


def main():
    # Read result tables
    baseline = read_results('baseline_results.csv')
    lstm = read_results('lstm_results.csv')
    cnn = read_results('cnn_bilstm_results.csv')

    dfs = {'Baseline': baseline, 'LSTM': lstm, 'CNN_BiLSTM': cnn}

    # Metrics comparisons
    plot_metric_comparison(dfs, 'RMSE', 'comparison_rmse.png')
    plot_metric_comparison(dfs, 'MAE', 'comparison_mae.png')
    plot_metric_comparison(dfs, 'R2', 'comparison_r2.png')

    # Time series from final_multimodal_dataset
    data_path = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')
    if os.path.exists(data_path):
        df = pd.read_csv(data_path, index_col=0, parse_dates=[0])
        # ensure columns exist
        if 'close' in df.columns:
            plot_timeseries(df, 'close', 'timeseries_close.png', 'Bitcoin Close Price')
        if 'rsi' in df.columns:
            plot_timeseries(df, 'rsi', 'timeseries_rsi.png', 'RSI')
        # sentiment_score may be present
        if 'sentiment_score' in df.columns:
            plot_timeseries(df, 'sentiment_score', 'timeseries_sentiment.png', 'Sentiment Score')
    else:
        logger.warning('final_multimodal_dataset.csv not found: %s', data_path)

    # Re-save LSTM training loss plot into visualizations if exists
    lstm_loss_src = os.path.join(OUT_DIR, 'lstm_training_loss.png')
    lstm_loss_dst = os.path.join(VIS_DIR, 'lstm_training_loss.png')
    if os.path.exists(lstm_loss_src):
        import shutil
        shutil.copy(lstm_loss_src, lstm_loss_dst)
        logger.info('Copied %s to %s', lstm_loss_src, lstm_loss_dst)
    else:
        logger.warning('LSTM loss plot not found: %s', lstm_loss_src)

    # List created files
    files = glob.glob(os.path.join(VIS_DIR, '*'))
    print('Created visualization files:')
    for f in files:
        print('-', os.path.relpath(f, ROOT))


if __name__ == '__main__':
    main()
