import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

ROOT = os.path.expanduser('~/Desktop/yzproje')
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')
OUT_DIR = os.path.join(ROOT, 'outputs')

st.set_page_config(layout='wide', page_title='Bitcoin Volatilite Tahmini')

st.title('Bitcoin Volatilite Tahmini - Multimodal Yapay Zeka Projesi')

with st.sidebar:
    st.header('Proje Özeti')
    st.write('Bu proje Bitcoin fiyat ve sosyal medya verilerini kullanarak volatilite tahmini yapar.')
    st.header('Veri Kaynakları')
    st.write('- Kaggle tweet dataset (alaix14)\n- Bitcoin 1-min price dataset (mczielinski)')
    st.header('Kullanılan Modeller')
    st.write('- Linear Regression (baseline)\n- RandomForest (baseline)\n- LSTM (PyTorch)\n- CNN + BiLSTM (PyTorch)')

st.markdown('---')

col1, col2 = st.columns(2)

if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])
    with col1:
        st.subheader('Bitcoin Close Price')
        fig, ax = plt.subplots(figsize=(8,3))
        ax.plot(df.index, df['close'])
        ax.set_xlabel('Date')
        ax.set_ylabel('Close')
        st.pyplot(fig)

    with col2:
        st.subheader('Sentiment Score')
        fig, ax = plt.subplots(figsize=(8,3))
        if 'sentiment_score' in df.columns:
            ax.plot(df.index, df['sentiment_score'])
        st.pyplot(fig)

    st.subheader('Tweet Count')
    fig, ax = plt.subplots(figsize=(12,3))
    if 'tweet_count' in df.columns:
        ax.plot(df.index, df['tweet_count'])
    st.pyplot(fig)
else:
    st.warning(f'Data file not found: {DATA_PATH}')

st.markdown('---')

# Read results
def load_results(name):
    p = os.path.join(OUT_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None

baseline = load_results('baseline_results.csv')
lstm = load_results('lstm_results.csv')
cnn = load_results('cnn_bilstm_results.csv')

results = []
for dfr in [baseline, lstm, cnn]:
    if dfr is not None:
        for _, row in dfr.iterrows():
            results.append(row.to_dict())

if results:
    res_df = pd.DataFrame(results)[['model','MAE','RMSE','R2']]
    st.subheader('Model Results Comparison')
    st.dataframe(res_df)

    # plots
    st.subheader('Metric Comparisons')
    fig, axes = plt.subplots(1,3, figsize=(15,4))
    res_df.plot(kind='bar', x='model', y='RMSE', ax=axes[0], legend=False)
    axes[0].set_title('RMSE')
    res_df.plot(kind='bar', x='model', y='MAE', ax=axes[1], legend=False)
    axes[1].set_title('MAE')
    res_df.plot(kind='bar', x='model', y='R2', ax=axes[2], legend=False)
    axes[2].set_title('R2')
    st.pyplot(fig)

    # Best model by RMSE
    best = res_df.loc[res_df['RMSE'].idxmin()]
    st.success(f"Best model by RMSE: {best['model']} (RMSE={best['RMSE']:.6f})")
else:
    st.info('No model results found in outputs/')

st.markdown('---')

# Display loss images if present
st.subheader('Training Loss Plots')
loss_imgs = [os.path.join(OUT_DIR,'lstm_training_loss.png'), os.path.join(OUT_DIR,'cnn_bilstm_training_loss.png')]
for img in loss_imgs:
    if os.path.exists(img):
        st.image(img, caption=os.path.basename(img))

st.markdown('---')
st.subheader('Kısa Yorum')
st.write('Baseline modellere göre LSTM daha düşük RMSE üretmiştir.')
st.write('CNN + BiLSTM daha karmaşık olmasına rağmen küçük veri ve düşük sentiment eşleşmesi nedeniyle daha iyi performans göstermemiş olabilir.')

st.markdown('---')
st.write('Çalıştırmak için:')
st.code('streamlit run app.py')
