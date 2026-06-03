import os
import sys
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import torch
import torch.nn as nn
import joblib


ROOT = os.path.expanduser('~/Desktop/yzproje')
DATA_PATH = os.path.join(ROOT, 'data', 'processed', 'final_multimodal_dataset.csv')
MODEL_DIR = os.path.join(ROOT, 'models')
OUT_DIR = os.path.join(ROOT, 'outputs')


st.set_page_config(
    layout='wide',
    page_title='Kripto Para Volatilite ve Risk Analiz Paneli',
    page_icon=None,
    initial_sidebar_state='expanded'
)


st.markdown("""
    <style>
    /* Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=Outfit:wght@300;400;600;700;800&display=swap');
    
    /* Ana arka plan */
    .stApp {
        background-color: #0c0d12;
        color: #d1d5db;
        font-family: 'Outfit', sans-serif;
    }
    
    /* Kenar çubuğu (Sidebar) */
    [data-testid="stSidebar"] {
        background-color: #0c0d12;
        border-right: 1px solid rgba(0, 240, 255, 0.1);
    }
    
    /* Glassmorphic kartlar */
    div[data-testid="stVerticalBlockBorderDiv"] {
        display: flex !important;
        flex-direction: column !important;
        height: auto !important;
        background: linear-gradient(135deg, rgba(20, 24, 33, 0.65) 0%, rgba(8, 9, 13, 0.9) 100%) !important;
        border: 1px solid rgba(0, 240, 255, 0.12) !important;
        border-radius: 16px !important;
        padding: 24px !important;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.5) !important;
        backdrop-filter: blur(8px) !important;
        -webkit-backdrop-filter: blur(8px) !important;
        transition: all 0.3s ease !important;
        margin-bottom: 20px !important;
    }
    div[data-testid="stVerticalBlockBorderDiv"]:hover {
        transform: translateY(-2px) !important;
        border-color: rgba(255, 0, 127, 0.45) !important;
        box-shadow: 0 12px 40px rgba(0, 240, 255, 0.1) !important;
    }
    
    /* Başlıklar */
    h1, h2, h3 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700 !important;
        letter-spacing: -0.02em;
        color: #00f0ff !important;
    }
    
    /* Renk geçişli metinler (Gradient) */
    .gradient-text {
        background: linear-gradient(90deg, #00f0ff 0%, #ff007f 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-weight: 800;
    }

    /* KPI Göstergeleri */
    .kpi-container {
        display: flex;
        justify-content: space-between;
        gap: 15px;
        margin-bottom: 25px;
    }
    .kpi-card {
        flex: 1;
        background: rgba(20, 24, 33, 0.4);
        border: 1px solid rgba(0, 240, 255, 0.08);
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .kpi-val {
        font-size: 1.8rem;
        font-weight: 800;
        color: #00f0ff;
    }
    .kpi-lbl {
        font-size: 0.75rem;
        color: #8a99ad;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        margin-top: 5px;
    }
    
    /* Başarı kutuları */
    .success-box {
        background: rgba(0, 255, 135, 0.06);
        border: 1px solid rgba(0, 255, 135, 0.2);
        border-radius: 8px;
        padding: 15px;
        color: #00ff87;
        font-weight: 500;
        margin-bottom: 15px;
    }
    </style>
    """, unsafe_allow_html=True)


def load_results(name):
    p = os.path.join(OUT_DIR, name)
    if os.path.exists(p):
        return pd.read_csv(p)
    return None


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


with st.sidebar:
    st.markdown('<h2 style="color: #00f0ff; text-align: center;">Çok Modlu Volatilite</h2>', unsafe_allow_html=True)
    st.write('---')
    st.subheader('Proje Özeti')
    st.markdown("""
    Bu platform, FinBERT NLP modeliyle çıkarılan **çok modlu Twitter duygu analizi** ile **tarihsel fiyat teknik göstergelerini** birleştirerek Bitcoin volatilitesini tahmin eder.
    """)
    st.write('---')
    st.subheader('Veri Detayları')
    st.markdown("""
    - **Taranan Tweet Sayısı:** 200.000 aktif tweet (FinBERT NLP)
    - **Tarih Aralığı:** 2017-01-27 ile 2019-05-27
    """)
    st.write('---')
    st.markdown('<div style="text-align: center; color:#ff007f; font-weight: 600; font-size: 0.8rem;">Hazırlayan: Hayrunnisa Büşra Erdem</div>', unsafe_allow_html=True)


st.markdown('<h1 class="gradient-text" style="font-size: 2.2rem; margin-bottom: 5px; line-height: 1.3;">Kripto Para Piyasalarında Çok Kanallı Veri Analizi ile Kısa Vadeli Volatilite Tahmini ve Risk Analizi</h1>', unsafe_allow_html=True)
st.markdown('<p style="color:#94a3b8; font-size:1.1rem; margin-bottom: 25px;">Piyasa Fiyat Hareketleri ve Sosyal Duygunun Çok Modlu Yapay Zeka Entegrasyonu</p>', unsafe_allow_html=True)


if os.path.exists(DATA_PATH):
    df = pd.read_csv(DATA_PATH, index_col=0, parse_dates=[0])
    

    df_active = df[(df.index >= '2017-01-27') & (df.index <= '2019-05-27')].copy()
    

    st.markdown(f"""
    <div class="kpi-container">
        <div class="kpi-card">
            <div class="kpi-val">{len(df_active):,}</div>
            <div class="kpi-lbl">Analiz Edilen Aktif Saat</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{df_active['tweet_count'].sum():,.0f}</div>
            <div class="kpi-lbl">Toplam Örneklenen Tweet</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{df_active['sentiment_score'].mean():.4f}</div>
            <div class="kpi-lbl">Ortalama Duygu Skoru</div>
        </div>
        <div class="kpi-card">
            <div class="kpi-val">{df_active['future_volatility'].mean() * 100:.3f}%</div>
            <div class="kpi-lbl">Ortalama Saatlik Volatilite</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    

    tab_analysis, tab_models, tab_prediction = st.tabs([
        'Fiyat & Duygu Analizi', 
        'Model Kıyaslama & Kayıp', 
        'Canlı Volatilite Tahmini'
    ])
    
    with tab_analysis:
        st.markdown('### İnteraktif Tarihsel Trendler')
        st.write('Piyasa fiyatı, sosyal duygu ve gelecek volatilite arasındaki ilişkileri keşfedin.')
        
        with st.expander("Teknik Göstergeler ve Özellik Mühendisliği (Feature Engineering) Açıklamaları"):
            st.markdown("""
            Bu çalışmada tahmin kalitesini artırmak için aşağıdaki ileri düzey finansal ve sosyal metrikler üretilmiştir:
            - **ATR (Average True Range - Ortalama Gerçek Aralık):** Fiyatın belirli bir zaman dilimindeki oynaklığını (volatilitesini) gösteren temel teknik indikatör.
            - **VWAP (Volume Weighted Average Price - Hacim Ağırlıklı Ortalama Fiyat):** Fiyatın hacimle ağırlıklandırılmış ortalamasıdır. Fiyat hareketinin gücünü ve kurumsal işlem seviyelerini yansıtır.
            - **RSI & MACD:** Trend yönü, momentumu ve aşırı alım/satım durumlarını ölçen standart teknik göstergelerdir.
            - **Ağırlıklı Duygu Skoru (Weighted Sentiment):** Ham duygu skorunun, tweet'in sosyal etkileşim (Beğeni ve Retweet) gücüne göre ağırlıklandırılmış halidir. Denklemi: $\\text{Duygu} \\times \\log(\\text{Beğeni} + \\text{Retweet} + 2)$ şeklindedir. Böylece yüksek etkileşimli influencer tweetlerinin etkisi artırılmıştır.
            """)
        

        min_date = df_active.index.min().to_pydatetime()
        max_date = df_active.index.max().to_pydatetime()
        
        selected_range = st.slider(
            "Analiz için Zaman Aralığını Seçin:",
            min_value=min_date,
            max_value=max_date,
            value=(min_date, max_date),
            format="YYYY-MM-DD"
        )
        
        df_filtered = df_active[(df_active.index >= selected_range[0]) & (df_active.index <= selected_range[1])]
        

        df_daily = df_filtered.resample('D').mean()
        
        col_chart1, col_chart2 = st.columns(2)
        
        with col_chart1:
            with st.container(border=True):
                st.markdown('<h4>Bitcoin Fiyatı & Bollinger Bantları</h4>', unsafe_allow_html=True)
                fig_price = go.Figure()
                fig_price.add_trace(go.Scatter(x=df_daily.index, y=df_daily['close'], name='Kapanış Fiyatı', line=dict(color='#00f0ff', width=2)))
                if 'bollinger_h' in df_daily.columns:
                    fig_price.add_trace(go.Scatter(x=df_daily.index, y=df_daily['bollinger_h'], name='Bollinger Üst', line=dict(color='rgba(0, 240, 255, 0.15)', width=1, dash='dash')))
                    fig_price.add_trace(go.Scatter(x=df_daily.index, y=df_daily['bollinger_l'], name='Bollinger Alt', line=dict(color='rgba(0, 240, 255, 0.15)', width=1, dash='dash'), fill='tonexty', fillcolor='rgba(0, 240, 255, 0.02)'))
                
                fig_price.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#d1d5db',
                    font_family='Outfit',
                    xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Tarih'),
                    yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Fiyat (USD)'),
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=320
                )
                st.plotly_chart(fig_price, use_container_width=True)
            
        with col_chart2:
            with st.container(border=True):
                st.markdown('<h4>Volatilite ve Twitter Duygusu</h4>', unsafe_allow_html=True)
                fig_vol = go.Figure()
                fig_vol.add_trace(go.Scatter(x=df_daily.index, y=df_daily['future_volatility'], name='Gelecek Volatilite', line=dict(color='#ff007f', width=2)))
                fig_vol.add_trace(go.Scatter(x=df_daily.index, y=df_daily['sentiment_score'], name='Duygu Skoru', line=dict(color='#00f0ff', width=1.5), yaxis='y2'))
                
                fig_vol.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#d1d5db',
                    font_family='Outfit',
                    xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Tarih'),
                    yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Volatilite', color='#ff007f'),
                    yaxis2=dict(title='Duygu Skoru', color='#00f0ff', overlaying='y', side='right'),
                    hovermode='x unified',
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=20, r=20, t=10, b=10),
                    height=320
                )
                st.plotly_chart(fig_vol, use_container_width=True)

        with st.container(border=True):
            st.markdown('<h4>Toplam Tweet Hacmi & Duygu Dalgalanmaları</h4>', unsafe_allow_html=True)
            fig_tweets = go.Figure()
            fig_tweets.add_trace(go.Bar(x=df_daily.index, y=df_daily['tweet_count'], name='Günlük Ortalama Tweet Sayısı', marker_color='rgba(0, 240, 255, 0.3)'))
            fig_tweets.add_trace(go.Scatter(x=df_daily.index, y=df_daily['sentiment_score'].rolling(7).mean(), name='7 Günlük Duygu HO', line=dict(color='#ff007f', width=2), yaxis='y2'))
            
            fig_tweets.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#d1d5db',
                font_family='Outfit',
                xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)'),
                yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Ortalama Tweet / Saat'),
                yaxis2=dict(title='Duygu (7G HO)', overlaying='y', side='right', color='#ff007f'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=20, r=20, t=10, b=10),
                height=280
            )
            st.plotly_chart(fig_tweets, use_container_width=True)


        with st.container(border=True):
            st.markdown('<h4>Zaman Serisi Bölümlemesi (Eğitim / Test Ayrımı)</h4>', unsafe_allow_html=True)
            st.write('Modellerimizin zaman serisi verilerinde geleceği görerek sızıntı yapmasını (Data Leakage) önlemek için veriyi sıralı (sequential) olarak böldük: İlk %80 eğitim seti (mavi) ve son %20 bağımsız test seti (pembe) olarak ayrılmıştır.')
            
            split_idx = int(len(df_daily) * 0.8)
            train_slice = df_daily.iloc[:split_idx]
            test_slice = df_daily.iloc[split_idx:]
            
            fig_split = go.Figure()
            fig_split.add_trace(go.Scatter(x=train_slice.index, y=train_slice['close'], name='Eğitim Seti (%80)', line=dict(color='#00f0ff', width=2)))
            fig_split.add_trace(go.Scatter(x=test_slice.index, y=test_slice['close'], name='Test Seti (%20)', line=dict(color='#ff007f', width=2)))
            

            split_date = df_daily.index[split_idx]
            fig_split.add_vline(x=split_date, line_width=2, line_dash="dash", line_color="#ff007f")
            
            fig_split.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='#d1d5db',
                font_family='Outfit',
                xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Tarih'),
                yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Bitcoin Kapanış Fiyatı (USD)'),
                hovermode='x unified',
                legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                margin=dict(l=20, r=20, t=10, b=10),
                height=300
            )
            st.plotly_chart(fig_split, use_container_width=True)

    with tab_models:
        st.markdown('### Model Karşılaştırma Paneli')
        st.write('Temel matematiksel/topluluk modellerini (Linear Regression, Random Forest) Derin Öğrenme zaman serisi modelleriyle karşılaştırın.')
        

        baseline_res = load_results('baseline_results.csv')
        lstm_res = load_results('lstm_results.csv')
        cnn_res = load_results('cnn_bilstm_results.csv')
        ensemble_res = load_results('ensemble_results.csv')
        
        results_list = []
        for dfr in [baseline_res, lstm_res, cnn_res, ensemble_res]:
            if dfr is not None:
                for _, row in dfr.iterrows():
                    results_list.append(row.to_dict())
                    
        if results_list:
            res_df = pd.DataFrame(results_list)[['model', 'MAE', 'RMSE', 'R2']]
            display_df = res_df.copy()
            for metric in ['MAE', 'RMSE', 'R2']:
                display_df[metric] = display_df[metric].apply(lambda x: f"{x:.6f}")
            

            st.dataframe(display_df, use_container_width=True)
            

            st.markdown('---')
            st.markdown('<h4>Performans Metriği Karşılaştırma Grafiği</h4>', unsafe_allow_html=True)
            
            selected_metric = st.selectbox(
                "Görselleştirmek istediğiniz performans metriğini seçin:",
                options=['RMSE', 'MAE', 'R2'],
                format_func=lambda x: {
                    'RMSE': 'RMSE (Kök Ortalama Kare Hata - Düşük Olması İyidir)',
                    'MAE': 'MAE (Ortalama Mutlak Hata - Düşük Olması İyidir)',
                    'R2': 'R² (Belirleyicilik Katsayısı - 1\'e Yakın Olması İyidir)'
                }[x],
                help="RMSE ve MAE tahmin hatalarının büyüklüğünü ölçer. R² (R-kare) ise modelin volatilite varyansını açıklama oranını gösterir."
            )
            
            with st.container(border=True):
                fig_metric = px.bar(
                    res_df, 
                    x='model', 
                    y=selected_metric, 
                    color='model',
                    color_discrete_sequence=['#1b1c22', '#00f0ff', '#ff007f', '#9d4edd', '#00ff87'],
                    text_auto='.6f' if selected_metric != 'R2' else '.4f'
                )
                fig_metric.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#d1d5db',
                    font_family='Outfit',
                    showlegend=False,
                    xaxis=dict(title='Model'),
                    yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title=selected_metric),
                    margin=dict(l=20, r=20, t=20, b=10),
                    height=300
                )
                st.plotly_chart(fig_metric, use_container_width=True)
                

            best_idx = res_df['RMSE'].idxmin()
            best_model_name = res_df.loc[best_idx, 'model']
            best_rmse_val = res_df.loc[best_idx, 'RMSE']
            
            st.markdown(f"""
            <div class="success-box">
                <b>En İyi Model Önerisi:</b> En tutarlı ve düşük hatalı tahmini <b>{best_rmse_val:.6f}</b> RMSE değeri ile <b>{best_model_name}</b> modeli elde etmiştir.
            </div>
            """, unsafe_allow_html=True)
        else:
            st.info('outputs/ altında model eğitim sonuçları bulunamadı. Lütfen önce eğitim betiklerini çalıştırın.')


        st.markdown('### Eğitim Kayıp Eğrileri')
        col_loss1, col_loss2 = st.columns(2)
        with col_loss1:
            lstm_loss_path = os.path.join(OUT_DIR, 'lstm_training_loss.png')
            if os.path.exists(lstm_loss_path):
                st.image(lstm_loss_path, caption='LSTM PyTorch Kayıp Geçmişi', use_container_width=True)
            else:
                st.info('LSTM Eğitim Kayıp eğrisi henüz oluşturulmadı.')
        with col_loss2:
            cnn_loss_path = os.path.join(OUT_DIR, 'cnn_bilstm_training_loss.png')
            if os.path.exists(cnn_loss_path):
                st.image(cnn_loss_path, caption='CNN-BiLSTM PyTorch Kayıp Geçmişi', use_container_width=True)
            else:
                st.info('CNN-BiLSTM Eğitim Kayıp eğrisi henüz oluşturulmadı.')


        feat_imp_path = os.path.join(OUT_DIR, 'feature_importances.csv')
        if os.path.exists(feat_imp_path):
            st.markdown('---')
            st.markdown('### Açıklanabilir Yapay Zeka (XAI) & Özellik Önem Derecesi')
            st.write('Random Forest modelinden elde edilen özellik önem dereceleri, volatilite tahmininde en belirleyici göstergeleri listeler.')
            
            with st.container(border=True):
                imp_df = pd.read_csv(feat_imp_path)

                fig_imp = px.bar(
                    imp_df.head(10),
                    x='importance',
                    y='feature',
                    orientation='h',
                    color='importance',
                    color_continuous_scale=['rgba(20, 24, 33, 0.4)', '#00f0ff'],
                    title='En Baskın 10 Girdi Parametresi'
                )
                fig_imp.update_layout(
                    plot_bgcolor='rgba(0,0,0,0)',
                    paper_bgcolor='rgba(0,0,0,0)',
                    font_color='#d1d5db',
                    font_family='Outfit',
                    xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Önem Katsayısı'),
                    yaxis=dict(title='Özellikler', categoryorder='total ascending'),
                    margin=dict(l=20, r=20, t=40, b=10),
                    coloraxis_showscale=False,
                    height=320
                )
                st.plotly_chart(fig_imp, use_container_width=True)


        test_preds_path = os.path.join(OUT_DIR, 'test_predictions.csv')
        if os.path.exists(test_preds_path):
            preds_df = pd.read_csv(test_preds_path, index_col=0, parse_dates=[0])
            st.markdown('---')
            st.markdown('### Test Seti Tahmin Doğruluğu & Dağılım Analizi')
            
            col_desc, col_dl = st.columns([3, 1])
            with col_desc:
                st.write('Modellerin bağımsız test verilerindeki tahminlerinin gerçek değerler ile kıyaslanması.')
            with col_dl:
                st.download_button(
                    label="Tahminleri İndir (CSV)",
                    data=preds_df.to_csv(),
                    file_name="btc_test_predictions.csv",
                    mime="text/csv",
                    help="Test setindeki tüm modellerin tahminlerini ve gerçek zaman serisi değerlerini CSV formatında bilgisayarınıza indirin."
                )
            
            col_preds_chart, col_scatter_chart = st.columns(2)
            
            with col_preds_chart:
                with st.container(border=True):
                    st.markdown('<h4>Gerçek vs Tahmin Edilen Değerler (Son 150 Saat)</h4>', unsafe_allow_html=True)
                    df_preds_sub = preds_df.tail(150)
                    
                    fig_preds_ts = go.Figure()
                    fig_preds_ts.add_trace(go.Scatter(x=df_preds_sub.index, y=df_preds_sub['Actual'], name='Gerçek Volatilite', line=dict(color='#ffffff', width=2)))
                    fig_preds_ts.add_trace(go.Scatter(x=df_preds_sub.index, y=df_preds_sub['LinearRegression'], name='Linear Regression', line=dict(color='rgba(0,240,255,0.4)', width=1, dash='dot')))
                    fig_preds_ts.add_trace(go.Scatter(x=df_preds_sub.index, y=df_preds_sub['LSTM'], name='LSTM', line=dict(color='#3b82f6', width=1.5)))
                    fig_preds_ts.add_trace(go.Scatter(x=df_preds_sub.index, y=df_preds_sub['CNN_BiLSTM'], name='CNN-BiLSTM', line=dict(color='#ff007f', width=1.5)))
                    fig_preds_ts.add_trace(go.Scatter(x=df_preds_sub.index, y=df_preds_sub['Ensemble'], name='Ensemble (Harmanlanmış)', line=dict(color='#00ff87', width=2)))
                    
                    fig_preds_ts.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#d1d5db',
                        font_family='Outfit',
                        xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)'),
                        yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Volatilite'),
                        hovermode='x unified',
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        margin=dict(l=20, r=20, t=10, b=10),
                        height=320
                    )
                    st.plotly_chart(fig_preds_ts, use_container_width=True)
                
            with col_scatter_chart:
                with st.container(border=True):
                    st.markdown('<h4>Ensemble Tahmin Dağılımı ve y = x Doğruluk Grafiği</h4>', unsafe_allow_html=True)
                    
                    fig_scatter = go.Figure()
                    fig_scatter.add_trace(go.Scatter(
                        x=preds_df['Actual'], 
                        y=preds_df['Ensemble'], 
                        mode='markers', 
                        name='Tahminler',
                        marker=dict(color='#ff007f', size=4, opacity=0.4)
                    ))
                    
                    min_val = min(preds_df['Actual'].min(), preds_df['Ensemble'].min())
                    max_val = max(preds_df['Actual'].max(), preds_df['Ensemble'].max())
                    
                    fig_scatter.add_trace(go.Scatter(
                        x=[min_val, max_val], 
                        y=[min_val, max_val], 
                        name='Mükemmel Uyum (y=x)', 
                        line=dict(color='#00f0ff', width=1.5, dash='dash')
                    ))
                    
                    fig_scatter.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#d1d5db',
                        font_family='Outfit',
                        xaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Gerçek Volatilite'),
                        yaxis=dict(gridcolor='rgba(0, 240, 255, 0.04)', title='Tahmin Edilen Volatilite'),
                        legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                        margin=dict(l=20, r=20, t=10, b=10),
                        height=320
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)

    with tab_prediction:
        st.markdown('### Gerçek Zamanlı Volatilite Tahmin Simülatörü')
        st.write('Piyasa senaryolarını test edin veya sosyal medya dalgalanmalarının volatiliteye etkisini simüle edin. Aşağıdaki parametreleri değiştirerek PyTorch modellerinden gerçek zamanlı çıkarım alabilirsiniz.')
        

        feature_scaler_path = os.path.join(MODEL_DIR, 'feature_scaler.pkl')
        target_scaler_path = os.path.join(MODEL_DIR, 'target_scaler.pkl')
        lstm_weights_path = os.path.join(MODEL_DIR, 'lstm_volatility_model_pytorch.pt')
        cnn_weights_path = os.path.join(MODEL_DIR, 'cnn_bilstm_volatility_model_pytorch.pt')
        
        if os.path.exists(feature_scaler_path) and os.path.exists(target_scaler_path):

            feat_scaler = joblib.load(feature_scaler_path)
            targ_scaler = joblib.load(target_scaler_path)
            

            FEATURE_COLS = [
                'open','high','low','close','volume','rsi','macd','macd_signal','bollinger_h','bollinger_l','return',
                'sentiment_score','tweet_count','likes','retweets','weighted_sentiment','atr','vwap'
            ]
            latest_window_df = df_active[FEATURE_COLS].tail(12).copy()
            
            col_sim_inputs, col_sim_output = st.columns([1, 1.2])
            
            with col_sim_inputs:
                with st.container(border=True):
                    st.markdown('<h4>Senaryo Parametreleri (Son Saat)</h4>', unsafe_allow_html=True)
                    

                    presets = {
                        "Manuel Kontrol (Son Saat Değerleri)": {
                            "close": float(latest_window_df['close'].iloc[-1]),
                            "return": float(latest_window_df['return'].iloc[-1]),
                            "sentiment": float(latest_window_df['sentiment_score'].iloc[-1]),
                            "tweets": int(latest_window_df['tweet_count'].iloc[-1]),
                            "likes": int(latest_window_df['likes'].iloc[-1]),
                            "retweets": int(latest_window_df['retweets'].iloc[-1])
                        },
                        "Boğa Koşusu (FOMO)": {
                            "close": float(latest_window_df['close'].iloc[-1] * 1.08),
                            "return": 0.065,
                            "sentiment": 0.82,
                            "tweets": 5200,
                            "likes": 16000,
                            "retweets": 4500
                        },
                        "Sert Çöküş (Panik)": {
                            "close": float(latest_window_df['close'].iloc[-1] * 0.91),
                            "return": -0.09,
                            "sentiment": -0.78,
                            "tweets": 7800,
                            "likes": 22000,
                            "retweets": 8500
                        },
                        "Sessiz Dönem (Akümülasyon)": {
                            "close": float(latest_window_df['close'].iloc[-1]),
                            "return": 0.0,
                            "sentiment": 0.05,
                            "tweets": 280,
                            "likes": 650,
                            "retweets": 180
                        }
                    }
                    

                    if 'sim_close' not in st.session_state:
                        st.session_state.sim_close = presets["Manuel Kontrol (Son Saat Değerleri)"]["close"]
                    if 'sim_return' not in st.session_state:
                        st.session_state.sim_return = presets["Manuel Kontrol (Son Saat Değerleri)"]["return"]
                    if 'sim_sentiment' not in st.session_state:
                        st.session_state.sim_sentiment = presets["Manuel Kontrol (Son Saat Değerleri)"]["sentiment"]
                    if 'sim_tweets' not in st.session_state:
                        st.session_state.sim_tweets = presets["Manuel Kontrol (Son Saat Değerleri)"]["tweets"]
                    if 'sim_likes' not in st.session_state:
                        st.session_state.sim_likes = presets["Manuel Kontrol (Son Saat Değerleri)"]["likes"]
                    if 'sim_retweets' not in st.session_state:
                        st.session_state.sim_retweets = presets["Manuel Kontrol (Son Saat Değerleri)"]["retweets"]


                    def on_preset_change():
                        p_name = st.session_state.selected_preset_val
                        if p_name in presets:
                            vals = presets[p_name]
                            st.session_state.sim_close = vals["close"]
                            st.session_state.sim_return = vals["return"]
                            st.session_state.sim_sentiment = vals["sentiment"]
                            st.session_state.sim_tweets = vals["tweets"]
                            st.session_state.sim_likes = vals["likes"]
                            st.session_state.sim_retweets = vals["retweets"]
                    

                    selected_preset = st.selectbox(
                        "Hazır Senaryo Preseti Seçin:",
                        options=list(presets.keys()),
                        key="selected_preset_val",
                        on_change=on_preset_change,
                        help="Hızlı piyasa durumları oluşturmak için hazır şablonları seçebilirsiniz. Seçimden sonra kaydırıcıları oynatarak değerleri özelleştirmeye devam edebilirsiniz."
                    )
                    

                    sim_close = st.slider('BTC Fiyatı (USD)', 
                                          min_value=float(latest_window_df['close'].iloc[-1]*0.8), 
                                          max_value=float(latest_window_df['close'].iloc[-1]*1.2), 
                                          key='sim_close',
                                          step=10.0,
                                          format="$%.2f",
                                          help="Bitcoin'in simüle edilen son saatteki fiyatı. Fiyat değişimleri Bollinger Bantları ve VWAP hesaplamalarını etkiler.")
                    
                    sim_return = st.slider('Saatlik Getiri', 
                                           min_value=-0.1, 
                                           max_value=0.1, 
                                           key='sim_return',
                                           step=0.001,
                                           format="%.4f",
                                           help="Son bir saatteki fiyat yüzdesel değişimi. Örneğin: 0.05 = +%5 getiri.")
                    
                    sim_sentiment = st.slider('Twitter Duygu Skoru', 
                                              min_value=-1.0, 
                                              max_value=1.0, 
                                              key='sim_sentiment',
                                              step=0.01,
                                              help="FinBERT NLP modelinin çıkardığı genel Twitter duygu skoru (-1: Aşırı Negatif, 0: Nötr, 1: Aşırı Pozitif).")
                    
                    sim_tweets = st.number_input('Tweet Hacmi (saatlik)', 
                                                 min_value=0, 
                                                 max_value=10000, 
                                                 key='sim_tweets',
                                                 help="Bitcoin hakkında son bir saatte atılan örnek tweet sayısı. Model bu sayıyı logaritmik ölçekte işler.")
                    
                    sim_likes = st.number_input('Tahmini Beğeni Hacmi', 
                                                min_value=0, 
                                                max_value=50000, 
                                                key='sim_likes',
                                                help="İlgili tweetlerin aldığı toplam beğeni sayısı. Ağırlıklı duygu skorunu doğrudan belirler.")
                    
                    sim_retweets = st.number_input('Tahmini Retweet Hacmi', 
                                                   min_value=0, 
                                                   max_value=25000, 
                                                   key='sim_retweets',
                                                   help="İlgili tweetlerin aldığı toplam retweet sayısı. Sosyal medyadaki yayılım gücünü yansıtır.")
                    
                    selected_model_name = st.radio(
                        'Tahmin Alınacak Derin Öğrenme Mimarisi:', 
                        ('LSTM (PyTorch)', 'CNN-BiLSTM (PyTorch)', 'Ensemble (LSTM + CNN-BiLSTM)'),
                        help="Kullanılacak tahmin modelini seçin. Ensemble seçilirse, CNN-BiLSTM (%88) ve LSTM (%12) harmanlanır."
                    )
                
            with col_sim_output:
                with st.container(border=True):
                    st.markdown('<h4>Simüle Edilen Tahmini Volatilite</h4>', unsafe_allow_html=True)
                    

                    sim_data = latest_window_df.values.copy()
                    

                    price_ratio = sim_close / latest_window_df['close'].iloc[-1]
                    sim_data[-1, 0] = latest_window_df['open'].iloc[-1] * price_ratio
                    sim_data[-1, 1] = latest_window_df['high'].iloc[-1] * price_ratio
                    sim_data[-1, 2] = latest_window_df['low'].iloc[-1] * price_ratio
                    sim_data[-1, 3] = sim_close
                    sim_data[-1, 10] = sim_return
                    sim_data[-1, 11] = sim_sentiment
                    sim_data[-1, 12] = sim_tweets
                    sim_data[-1, 13] = sim_likes
                    sim_data[-1, 14] = sim_retweets
                    sim_data[-1, 15] = sim_sentiment * np.log1p(sim_likes + sim_retweets + 1.0)
                    sim_data[-1, 16] = latest_window_df['atr'].iloc[-1]
                    sim_data[-1, 17] = latest_window_df['vwap'].iloc[-1]
                    


                    sim_data[:, 4] = np.log1p(sim_data[:, 4])
                    sim_data[:, 12] = np.log1p(sim_data[:, 12])
                    sim_data[:, 13] = np.log1p(sim_data[:, 13])
                    sim_data[:, 14] = np.log1p(sim_data[:, 14])
                    

                    sim_scaled = feat_scaler.transform(sim_data)
                    sim_tensor = torch.from_numpy(sim_scaled).float().unsqueeze(0)
                    

                    predicted_val = 0.0
                    try:
                        if selected_model_name == 'LSTM (PyTorch)':
                            if os.path.exists(lstm_weights_path):
                                lstm_model = LSTMModel(input_size=18)
                                lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location='cpu'))
                                lstm_model.eval()
                                with torch.no_grad():
                                    pred_scaled = lstm_model(sim_tensor).numpy()
                                predicted_val = targ_scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]
                            else:
                                st.warning('LSTM model ağırlıkları bulunamadı. Önce modeli eğitin.')
                        elif selected_model_name == 'CNN-BiLSTM (PyTorch)':
                            if os.path.exists(cnn_weights_path):
                                cnn_model = CNNBiLSTM(input_features=18)
                                cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location='cpu'))
                                cnn_model.eval()
                                with torch.no_grad():
                                    pred_scaled = cnn_model(sim_tensor).numpy()
                                predicted_val = targ_scaler.inverse_transform(pred_scaled.reshape(-1,1)).flatten()[0]
                            else:
                                st.warning('CNN-BiLSTM model ağırlıkları bulunamadı. Önce modeli eğitin.')
                        else:
                            if os.path.exists(lstm_weights_path) and os.path.exists(cnn_weights_path):
                                lstm_model = LSTMModel(input_size=18)
                                lstm_model.load_state_dict(torch.load(lstm_weights_path, map_location='cpu'))
                                lstm_model.eval()
                                
                                cnn_model = CNNBiLSTM(input_features=18)
                                cnn_model.load_state_dict(torch.load(cnn_weights_path, map_location='cpu'))
                                cnn_model.eval()
                                
                                with torch.no_grad():
                                    pred_lstm_scaled = lstm_model(sim_tensor).numpy()
                                    pred_cnn_scaled = cnn_model(sim_tensor).numpy()
                                    
                                pred_lstm = targ_scaler.inverse_transform(pred_lstm_scaled.reshape(-1,1)).flatten()[0]
                                pred_cnn = targ_scaler.inverse_transform(pred_cnn_scaled.reshape(-1,1)).flatten()[0]
                                

                                predicted_val = 0.88 * pred_cnn + 0.12 * pred_lstm
                            else:
                                st.warning('Modellerin ağırlıkları bulunamadı. Önce modelleri eğitin.')
                    except Exception as e:
                        st.error(f'Model çıkarım simülasyonunda hata oluştu: {e}')
                    

                    fig_gauge = go.Figure(go.Indicator(
                        mode = "gauge+number",
                        value = float(predicted_val),
                        domain = {'x': [0, 1], 'y': [0, 1]},
                        title = {'text': f"{selected_model_name} Tahmini", 'font': {'size': 18, 'color': '#00f0ff'}},
                        number = {'valueformat': ".6f", 'font': {'color': '#00f0ff', 'size': 32}},
                        gauge = {
                            'axis': {'range': [0, 0.035], 'tickwidth': 1, 'tickcolor': "#d1d5db", 'tickformat': ".4f"},
                            'bar': {'color': "#00f0ff"},
                            'bgcolor': "rgba(20, 24, 33, 0.4)",
                            'borderwidth': 1,
                            'bordercolor': "rgba(0, 240, 255, 0.2)",
                            'steps': [
                                {'range': [0, 0.005], 'color': 'rgba(0, 255, 135, 0.15)'},
                                {'range': [0.005, 0.015], 'color': 'rgba(245, 158, 11, 0.15)'},
                                {'range': [0.015, 0.035], 'color': 'rgba(239, 68, 68, 0.15)'}
                            ],
                        }
                    ))
                    
                    fig_gauge.update_layout(
                        plot_bgcolor='rgba(0,0,0,0)',
                        paper_bgcolor='rgba(0,0,0,0)',
                        font_color='#d1d5db',
                        font_family='Outfit',
                        margin=dict(l=30, r=30, t=50, b=30),
                        height=280
                    )
                    
                    st.plotly_chart(fig_gauge, use_container_width=True)
                    

                    status_text = ""
                    if predicted_val < 0.005:
                        status_text = "**Düşük Volatilite Rejimi:** Model bu koşullar altında oldukça durağan ve konsolide (stabil) bir fiyat hareketi bekliyor."
                    elif predicted_val < 0.015:
                        status_text = "**Normal Volatilite Rejimi:** Standart piyasa dalgalanmaları bekleniyor; normal işlem ve trade koşulları öngörülmektedir."
                    else:
                        status_text = "**Yüksek Volatilite Rejimi:** Sert fiyat hareketleri öngörülüyor! Bu durum, yüksek sosyal medya hacmi ve ani duygu değişimleriyle ilişkili olarak sert kırılımlara (yukarı veya aşağı yönlü) işaret edebilir."
                    
                    st.markdown(status_text)
                
        else:
            st.info('Model özellik/hedef ölçekleyici dosyaları eksik. Model eğitim scriptlerinin başarıyla çalıştığından emin olun.')
            
else:
    st.warning(f'Birleştirilmiş çok modlu veri seti {DATA_PATH} yolunda bulunamadı. Lütfen önce veri birleştirme adımlarını çalıştırın.')
