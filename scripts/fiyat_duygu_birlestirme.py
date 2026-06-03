#!/usr/bin/env python3
"""Merge hourly Bitcoin price features with hourly tweet sentiment.

Produces data/processed/final_multimodal_dataset.csv
"""
import os
import sys
import logging
import pandas as pd
import numpy as np

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)

ROOT = os.path.expanduser('~/Desktop/yzproje')
PRICE_PATH = os.path.join(ROOT, 'data', 'processed', 'bitcoin_hourly_features.csv')
SENT_PATH = os.path.join(ROOT, 'data', 'processed', 'btc_sentiment_hourly.csv')
OUT_PATH = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')


def to_datetime_index(df, guess_col=None):
    """Ensure the dataframe has a datetime index.
    If the index looks like a timestamp, convert it. Otherwise, try guess_col.
    Returns dataframe with datetime index (not tz-converted).
    """

    try:
        if not pd.api.types.is_datetime64_any_dtype(df.index):

            if guess_col and guess_col in df.columns:
                df[guess_col] = pd.to_datetime(df[guess_col], errors='coerce')
                df = df.set_index(guess_col)
            else:

                df.index = pd.to_datetime(df.index, errors='coerce')
    except Exception:
        logger.exception('Error converting to datetime index')


    df = df[~df.index.isna()].copy()
    return df


def normalize_timezone_to_utc(idx):

    idx = pd.to_datetime(idx)
    if idx.tz is None:
        idx = idx.tz_localize('UTC')
    else:
        idx = idx.tz_convert('UTC')
    return idx


def main():
    if not os.path.exists(PRICE_PATH):
        logger.error('Price file not found: %s', PRICE_PATH)
        sys.exit(1)
    if not os.path.exists(SENT_PATH):
        logger.error('Sentiment file not found: %s', SENT_PATH)
        sys.exit(1)


    price = pd.read_csv(PRICE_PATH, parse_dates=[0], index_col=0)
    sentiment = pd.read_csv(SENT_PATH, parse_dates=[0], index_col=0)

    logger.info('price shape: %s', price.shape)
    logger.info('sentiment shape: %s', sentiment.shape)


    price = to_datetime_index(price)
    sentiment = to_datetime_index(sentiment)


    try:
        price.index = normalize_timezone_to_utc(price.index)
    except Exception:
        price.index = pd.to_datetime(price.index)
        price.index = price.index.tz_localize('UTC')

    try:
        sentiment.index = normalize_timezone_to_utc(sentiment.index)
    except Exception:
        sentiment.index = pd.to_datetime(sentiment.index)
        sentiment.index = sentiment.index.tz_localize('UTC')


    sentiment['weighted_sentiment'] = sentiment['sentiment_score'] * np.log1p(sentiment['likes'] + sentiment['retweets'] + 1.0)


    merged = price.join(sentiment, how='left', rsuffix='_sent')


    if 'sentiment_score' not in merged.columns:
        merged['sentiment_score'] = 0.0
    merged['sentiment_score'] = merged['sentiment_score'].ffill().fillna(0.0)

    if 'weighted_sentiment' not in merged.columns:
        merged['weighted_sentiment'] = 0.0
    merged['weighted_sentiment'] = merged['weighted_sentiment'].ffill().fillna(0.0)

    for col in ['tweet_count', 'likes', 'retweets']:
        if col not in merged.columns:
            merged[col] = 0
        merged[col] = merged[col].fillna(0)


    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    merged.to_csv(OUT_PATH)


    print('price shape', price.shape)
    print('sentiment shape', sentiment.shape)
    print('merged shape', merged.shape)
    print('\ncolumns:')
    print(merged.columns.tolist())

    matched = int((merged['tweet_count'] > 0).sum()) if 'tweet_count' in merged.columns else 0
    print('\nsentiment eşleşen satır sayısı:', matched)
    print('\nilk 5 satır:')
    print(merged.head().to_string())
    print('\ndosya var mı?:', os.path.exists(OUT_PATH))


if __name__ == '__main__':
    main()
