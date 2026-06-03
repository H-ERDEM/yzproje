#!/usr/bin/env python3

import os
os.environ['USE_TF'] = '0'
os.environ['USE_TORCH'] = '1'
import re
import sys
import logging
from html import unescape
from typing import List

import pandas as pd
import numpy as np
import torch
from transformers import pipeline

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


SAMPLE_SIZE = 200000
BATCH_SIZE = 32
CHUNKSIZE = 2000
CSV_PATH = os.path.expanduser('~/Desktop/yzproje/data/raw/tweets/tweets.csv')
OUT_DIR = os.path.expanduser('~/Desktop/yzproje/data/processed')
OUT_HOURLY = os.path.join(OUT_DIR, 'btc_sentiment_hourly.csv')
OUT_RAW = os.path.join(OUT_DIR, 'btc_sentiment_raw_sample.csv')


USE_COLS = ['timestamp', 'text', 'replies', 'likes', 'retweets']


def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ''

    text = unescape(text)

    text = re.sub(r'http\S+|https?://\S+', '', text)

    text = re.sub(r'@\w+', '', text)

    text = re.sub(r'\bRT\b', '', text)

    text = text.replace('#', '')

    text = re.sub(r'\s+', ' ', text).strip()
    return text


def map_label_to_score(label: str, score: float) -> float:
    lab = label.lower()
    if 'pos' in lab or 'positive' in lab:
        return float(score)
    if 'neg' in lab or 'negative' in lab:
        return -float(score)
    return 0.0


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    if not os.path.exists(CSV_PATH):
        logger.error('CSV file not found: %s', CSV_PATH)
        sys.exit(1)


    if torch.backends.mps.is_available():
        device = 'mps'
    elif torch.cuda.is_available():
        device = 0
    else:
        device = -1
    logger.info('Device set to use %s', device)

    try:
        sentiment_pipe = pipeline('sentiment-analysis', model='ProsusAI/finbert', tokenizer='ProsusAI/finbert', device=device)
    except Exception as e:
        logger.exception('Failed to load FinBERT pipeline: %s', e)
        sys.exit(1)

    processed_rows = []
    total_read = 0


    usecols_present = None
    for chunk in pd.read_csv(CSV_PATH, sep=';', usecols=lambda c: c in USE_COLS or c in ['timestamp','text','replies','likes','retweets'], chunksize=CHUNKSIZE, iterator=True, encoding='utf-8', dtype=str):

        chunk.columns = [c.strip() for c in chunk.columns]

        cols = [c for c in USE_COLS if c in chunk.columns]
        if usecols_present is None:
            usecols_present = cols
            logger.info('Columns present in CSV used: %s', usecols_present)
        dfc = chunk[cols].copy()


        if 'timestamp' in dfc.columns:
            dfc['timestamp'] = pd.to_datetime(dfc['timestamp'], errors='coerce')


        if 'text' in dfc.columns:
            dfc['text'] = dfc['text'].map(clean_text)

            dfc = dfc[dfc['text'].str.len() >= 3]
        else:
            logger.error('No text column in chunk; aborting')
            break


        for col in ['replies', 'likes', 'retweets']:
            if col in dfc.columns:
                dfc[col] = pd.to_numeric(dfc[col], errors='coerce').fillna(0).astype(int)
            else:
                dfc[col] = 0

        processed_rows.append(dfc)
        total_read += len(dfc)
        logger.info('Accumulated %d processed tweets', total_read)

        if total_read >= SAMPLE_SIZE:
            logger.info('Reached SAMPLE_SIZE=%d, stopping read.', SAMPLE_SIZE)
            break

    if len(processed_rows) == 0:
        logger.error('No rows processed from CSV.')
        sys.exit(1)

    df_all = pd.concat(processed_rows, ignore_index=True).head(SAMPLE_SIZE)
    logger.info('Total tweets prepared for sentiment inference: %d', len(df_all))

    texts = df_all['text'].fillna('').tolist()
    sentiment_scores: List[float] = []


    try:
        for i in range(0, len(texts), BATCH_SIZE):
            batch = texts[i:i+BATCH_SIZE]
            preds = sentiment_pipe(batch)
            for p in preds:
                sentiment_scores.append(map_label_to_score(p.get('label',''), p.get('score', 0.0)))
    except Exception as e:
        logger.exception('Error during sentiment inference: %s', e)
        sys.exit(1)

    df_all['sentiment_score'] = sentiment_scores


    df_all.to_csv(OUT_RAW, index=False)


    if 'timestamp' not in df_all.columns or df_all['timestamp'].isna().all():
        logger.error('No valid timestamp column available; cannot compute hourly aggregates.')
        print('Processed tweets:', len(df_all))
        print('Raw sample saved to:', OUT_RAW)
        print('btc_sentiment_hourly.csv oluşturulamadı (timestamp yok)')
        sys.exit(0)


    df_all['timestamp'] = pd.to_datetime(df_all['timestamp'], errors='coerce')
    df_all = df_all.dropna(subset=['timestamp']).copy()


    df_all['timestamp_hour'] = df_all['timestamp'].dt.floor('h')

    hourly = df_all.groupby('timestamp_hour').agg({
        'sentiment_score': 'mean',
        'text': 'count',
        'likes': 'mean',
        'retweets': 'mean'
    }).rename(columns={'text': 'tweet_count'})


    hourly = hourly.fillna(0)


    price_path = os.path.expanduser('~/Desktop/yzproje/data/processed/bitcoin_hourly_features.csv')
    try:
        if os.path.exists(price_path):

            price_df = pd.read_csv(price_path, parse_dates=[0], index_col=0)

            price_idx = pd.to_datetime(price_df.index)
            if price_idx.tz is None:
                price_idx = price_idx.tz_localize('UTC')
            else:
                price_idx = price_idx.tz_convert('UTC')


            h_idx = pd.to_datetime(hourly.index)
            if h_idx.tz is None:
                h_idx = h_idx.tz_localize('UTC')
            else:
                h_idx = h_idx.tz_convert('UTC')


            hourly.index = h_idx
            price_min = price_idx.min()
            price_max = price_idx.max()

            hourly = hourly[(hourly.index >= price_min) & (hourly.index <= price_max)]
        else:
            logger.warning('Price file not found, skipping date-range filter: %s', price_path)
    except Exception:
        logger.exception('Error while reading price file for date range; skipping filter.')


    hourly.to_csv(OUT_HOURLY)


    try:
        raw_shape = pd.read_csv(OUT_RAW).shape
        hourly_df = pd.read_csv(OUT_HOURLY, parse_dates=[0], index_col=0)

        print('\nraw sample shape:', raw_shape)
        print('\nhourly shape:', hourly_df.shape)

        print('\nhourly tarih min:', hourly_df.index.min())
        print('\nhourly tarih max:', hourly_df.index.max())

        total_tweets = int(hourly_df['tweet_count'].sum()) if 'tweet_count' in hourly_df.columns else 0
        print('\ntweet_count toplamı:', total_tweets)
        print('\nilk 5 satır:')
        print(hourly_df.head().to_string())
        print('\ndosya var mı:', os.path.exists(OUT_HOURLY))
    except Exception as e:
        logger.exception('Error while reporting outputs: %s', e)


if __name__ == '__main__':
    main()
