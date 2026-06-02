# Multimodal Deep Learning Framework for Bitcoin Volatility Forecasting: Integrating Price Dynamics and Sentiment Analytics

Bu çalışma, Bitcoin fiyat verileri ve sosyal medya (Twitter) duyarlılık analizini birleştirerek saatlik bazda gelecek volatiliteyi tahmin eden multimodal (çok modlu) bir yapay zeka modelleme çerçevesi sunmaktadır. Çalışmada zaman serisi fiyat hareketleri, teknik analiz indikatörleri ve FinBERT tabanlı metinsel duygu skorları tek bir öznitelik uzayında birleştirilmiş; baseline regresyon modelleri (Linear Regression, Random Forest) ve derin öğrenme mimarileri (LSTM, CNN-BiLSTM) kullanılarak karşılaştırmalı performans analizleri gerçekleştirilmiştir.

---

## 🚀 Proje Genel Mimarisi ve Akış Şeması

Proje, heterojen veri kaynaklarının işlenmesi, zamansal hizalanması, multimodal füzyonu ve derin öğrenme modellerine beslenerek volatilite kestirimi yapılması sürecini kapsar.

```text
       ┌────────────────────────┐      ┌─────────────────────────┐
       │   Bitcoin Fiyat Verisi │      │ Twitter Tweet Veri Seti │
       │ (1-dakikalık Ham Veri) │      │   (Ham CSV, Semikolon)  │
       └───────────┬────────────┘      └────────────┬────────────┘
                   │                                │
                   ▼ (Saatlik Resample)             ▼ (Text Temizleme)
       ┌────────────────────────┐      ┌─────────────────────────┐
       │     OHLCV Serisi       │      │   Temizlenmiş Metinler  │
       └───────────┬────────────┘      └────────────┬────────────┘
                   │                                │
                   ▼ (Teknik Analiz)                ▼ (ProsusAI/FinBERT)
       ┌────────────────────────┐      ┌─────────────────────────┐
       │  RSI, MACD, Bollinger  │      │ Duygu Sınıflandırma     │
       │   Bantları & Getiri    │      │ (Pos/Neg/Neu Skorları)  │
       └───────────┬────────────┘      └────────────┬────────────┘
                   │                                │
                   │                                ▼ (Saatlik Agregasyon)
                   │                   ┌─────────────────────────┐
                   │                   │   Saatlik Ortalama      │
                   │                   │  Duygu, Etkileşim & Sayı│
                   │                   └────────────┬────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼ (Temporal Left-Join Füzyonu)
                     ┌───────────────────────────┐
                     │   Multimodal Veri Seti    │
                     │ (final_multimodal_dataset)│
                     └─────────────┬─────────────┘
                                   │
                   ┌───────────────┴───────────────┐
                   ▼ (Zaman Serisi Sekansları)     ▼ (Baseline Regresyon)
       ┌────────────────────────┐      ┌─────────────────────────┐
       │ PyTorch Derin Öğrenme  │      │   Linear Regression &   │
       │  (LSTM & CNN-BiLSTM)   │      │ Random Forest Regressor │
       └───────────┬────────────┘      └────────────┬────────────┘
                   │                                │
                   └───────────────┬────────────────┘
                                   │
                                   ▼ (Kestirim & Hata Analizi)
                     ┌───────────────────────────┐
                     │ Gelecek Volatilite Tahmini│
                     │ (Future Volatility Pred)  │
                     └───────────────────────────┘
```

---

## 1. Teorik Altyapı ve Matematiksel Formülasyon

### 1.1. Getiri ve Hedef Volatilite Hesaplaması
Zaman serisi fiyat kestiriminde doğrudan fiyat tahmini yerine volatilite tahmini risk yönetimi ve opsiyon fiyatlaması için kritik öneme sahiptir. Saatlik kapanış fiyat serisi $P_t$ üzerinden saatlik yüzde getiri $r_t$ aşağıdaki gibi tanımlanır:

$$r_t = \frac{P_t - P_{t-1}}{P_{t-1}}$$

Hedef değişken olarak kullanılan **Gelecek Volatilite** ($FutureVolatility_t$), cari $t$ anından sonraki 24 saatlik zaman penceresindeki ($t+1$ ila $t+24$) getirilerin standart sapması (rolling standard deviation) olarak tanımlanır ve $24$ adım geriye kaydırılarak (shift) hedef öznitelik haline getirilir:

$$\sigma_{t, t+24} = \sqrt{\frac{1}{24}\sum_{i=1}^{24}(r_{t+i} - \bar{r}_{t,t+24})^2}$$

Burada $\bar{r}_{t,t+24}$ ilgili 24 saatlik penceredeki ortalama getiridir.

### 1.2. Teknik Analiz Öznitelikleri (Feature Extraction)
Fiyat serisinin momentum, trend ve oynaklık özelliklerini yakalamak için aşağıdaki teknik indikatörler hesaplanmıştır:

1. **Relative Strength Index (RSI):** Finansal varlığın aşırı alım veya aşırı satım durumunu ölçen momentum osilatörüdür. 14 saatlik periyot için hesaplanır:
   $$RSI = 100 - \left(\frac{100}{1 + RS}\right)$$
   $$RS = \frac{\text{14 periyottaki ortalama kazanç}}{\text{14 periyottaki ortalama kayıp}}$$

2. **Moving Average Convergence Divergence (MACD):** Trend takip edici momentum göstergesidir. 12 ve 26 periyotluk Üstel Hareketli Ortalamalar (EMA) arasındaki farktan elde edilir:
   $$MACD_t = EMA_{12}(P_t) - EMA_{26}(P_t)$$
   $$Signal_t = EMA_{9}(MACD_t)$$

3. **Bollinger Bands:** Fiyatın oynaklık aralığını belirleyen zarflardır. 20 günlük basit hareketli ortalama ($SMA_{20}$) ve buna eklenen/çıkarılan 2 standart sapma ($\sigma$) değerinden türetilir:
   $$Bollinger_H = SMA_{20}(P_t) + 2\sigma(P_t)$$
   $$Bollinger_L = SMA_{20}(P_t) - 2\sigma(P_t)$$

### 1.3. Doğal Dil İşleme ve FinBERT Tabanlı Duygu Analizi
Metinsel veri kaynağından finansal duygu analizi yapabilmek için finansal korpuslar üzerinde özel olarak eğitilmiş **FinBERT** (`ProsusAI/finbert`) modeli tercih edilmiştir. Model, girdi olarak verilen tweet metnini üç sınıfa olasılıksal olarak atar: Pozitif ($p_{pos}$), Negatif ($p_{neg}$) ve Nötr ($p_{neu}$). 

Tekil bir tweet için nihai duygu skoru ($s_{tweet}$) şu şekilde formüle edilmiştir:

$$s_{tweet} = p_{pos} \cdot (Score_{pos}) - p_{neg} \cdot (Score_{neg})$$

Burada $Score_{pos}$ ve $Score_{neg}$ ilgili sınıflara ait softmax olasılıklarıdır.

Saatlik agregasyon aşamasında, $t$ saat dilimi içerisine düşen tüm tweet'lerin duygu skorları ağırlıklı ortalaması alınarak saatlik duygu endeksi ($S_t$) hesaplanır:

$$S_t = \frac{1}{N_t} \sum_{j=1}^{N_t} s_{tweet, j}$$

Ayrıca saatlik toplam tweet sayısı ($TweetCount_t$), ortalama beğeni ($Likes_t$) ve ortalama retweet ($Retweets_t$) miktarı da sisteme sosyal etki katsayıları olarak dahil edilir.

---

## 2. Derin Öğrenme Modelleri ve Katman Mimarileri

Projede zaman serisinin ardışık (sequential) yapısını ve öznitelikler arası uzamsal ilişkileri modellemek amacıyla iki farklı PyTorch tabanlı mimari tasarlanmıştır.

### 2.1. PyTorch LSTM Modeli
Uzun Kısa Vadeli Bellek (LSTM) ağları, zaman serisi verilerindeki uzun dönemli bağımlılıkları öğrenmede standart RNN'lerin yaşadığı gradyan kaybolması (vanishing gradient) problemini kapı (gate) mekanizmalarıyla aşar.

Girdi sekansı $X \in \mathbb{R}^{B \times W \times F}$ (burada $B$: batch size, $W$: pencere uzunluğu - 12 saat, $F$: öznitelik sayısı - 15) LSTM katmanına beslenir. Model yapısı şu şekildedir:
- **LSTM Katmanı:** Giriş Boyutu ($F=15$), Gizli Durum Boyutu ($HiddenSize=32$), Katman Sayısı ($NumLayers=1$).
- **Fully Connected (Dense) Katmanı:** $32 \rightarrow 1$ boyut dönüşümü ile doğrusal projeksiyon yaparak $\hat{y}_t$ volatilite değerini tahmin eder.

İleri besleme (forward pass) sırasında, sekansın en son zaman adımına ait gizli durumu ($h_W$) alınarak regresyon katmanına aktarılır:
$$\hat{y}_t = W_{fc} \cdot h_W + b_{fc}$$

### 2.2. PyTorch CNN-BiLSTM Hibrit Modeli
Bu mimari, girdi özelliklerinden yerel zamansal örüntüleri çıkarmak için 1-Boyutlu Evrişimli Sinir Ağlarını (1D-CNN) ve çift yönlü zamansal bağlamı öğrenmek için Bidirectional LSTM (BiLSTM) katmanını birleştirir.

Katman akışı ve tensör boyut değişimleri:
1. **Girdi Tensörü:** $X \in \mathbb{R}^{B \times W \times F}$
2. **Permütasyon:** Evrişim katmanı kanal bazlı çalıştığından öznitelikler kanal boyutuna çekilir: $X_{perm} \in \mathbb{R}^{B \times F \times W}$
3. **1D Convolution (Conv1D):** Giriş Kanalı ($F=15$), Çıkış Kanalı ($ConvOut=32$), Kernel Boyutu ($K=3$), Dolgu ($Padding=1$). Bu katman, zaman adımları boyunca öznitelik kombinasyonlarını filtreler.
   $$X_{conv} = \text{ReLU}(\text{Conv1d}(X_{perm})) \in \mathbb{R}^{B \times 32 \times W}$$
4. **1D Max Pooling (MaxPool1d):** Kernel Boyutu = 2. Zamansal çözünürlüğü yarıya indirerek en baskın özellikleri öne çıkarır ve aşırı öğrenmeyi (overfitting) engeller.
   $$X_{pool} = \text{MaxPool1d}(X_{conv}) \in \mathbb{R}^{B \times 32 \times \frac{W}{2}}$$
5. **Permütasyon:** BiLSTM beslemesi için tensör tekrar zamansal sıraya getirilir: $X_{lstm\_in} \in \mathbb{R}^{B \times \frac{W}{2} \times 32}$
6. **Çift Yönlü LSTM (BiLSTM):** Giriş Boyutu = 32, Gizli Durum = 32. Çift yönlü olduğu için ileri ($h_{forward}$) ve geri ($h_{backward}$) yönlü gizli durumlar birleştirilir (concatenate).
   $$h_t = [h_{forward, t} \,;\, h_{backward, t}] \in \mathbb{R}^{B \times 64}$$
7. **Lineer Katman:** Birleştirilmiş 64 boyutlu vektör tek bir regresyon çıkışına indirgenir:
   $$\hat{y}_t = W_{fc} \cdot h_{last} + b_{fc} \quad (\text{burada } W_{fc} \in \mathbb{R}^{64 \times 1})$$

---

## 3. Deneysel Bulgular ve Performans Analizi

Modellerin değerlendirilmesinde Ortalama Mutlak Hata (MAE), Kök Ortalama Kare Hata (RMSE) ve Belirleyicilik Katsayısı ($R^2$) metrikleri kullanılmıştır. 

### 3.1. Karşılaştırmalı Sonuçlar Tablosu

| Model | MAE | RMSE | $R^2$ Skoru |
| :--- | :---: | :---: | :---: |
| **Linear Regression** | 0.002855 | 0.003621 | -1.402972 |
| **Random Forest Regressor** | 0.003030 | 0.003552 | -1.312252 |
| **PyTorch LSTM** (En Başarılı) | **0.001488** | **0.002048** | **0.010044** |
| **PyTorch CNN-BiLSTM** | 0.001604 | 0.002201 | -0.143415 |

### 3.2. Sonuçların Akademik Değerlendirmesi
- **Zaman Bağımlılığının Önemi:** PyTorch LSTM modeli, RMSE (0.002048) ve MAE (0.001488) kriterlerine göre baseline modellere (Linear Regression ve Random Forest) kıyasla yaklaşık **%40-45 oranında daha düşük hata** üretmiştir. Bu durum, finansal volatilitenin anlık özniteliklerden ziyade geçmişe dönük zamansal sekans bağımlılıkları (temporal dependency) barındırdığını doğrulamaktadır.
- **CNN-BiLSTM vs. Standart LSTM:** CNN-BiLSTM mimarisinin daha karmaşık ve parametrik olmasına rağmen LSTM'den daha zayıf performans göstermesi, veri setindeki sınırlı sentiment eşleşmesi ve veri hacminden kaynaklanmaktadır. Derin ve hibrit mimariler, aşırı öğrenmeye girmeden optimize olabilmek için daha geniş veri kümelerine gereksinim duyar.
- **Düşük / Negatif $R^2$ Değerlerinin Analizi:** Finansal volatilite tahmin problemlerinde negatif veya çok düşük pozitif $R^2$ değerleri literatürde sıkça karşılaşılan bir durumdur. Finansal zaman serilerinin içerdiği yüksek gürültü (noise) ve rassal yürüyüş (random walk) bileşenleri, modellerin varyansın tamamını açıklamasını zorlaştırır. LSTM'in $R^2$ skorunu pozitif bölgede tutabilmesi ($0.0100$), multimodal yaklaşımın gürültüyü bir miktar bastırabildiğinin göstergesidir.

---

## 4. Proje Dizini ve Yazılım Mimarisi

```text
yzproje/
├── data/
│   ├── raw/                             # Ham veri kaynakları (Değiştirilmez)
│   │   ├── bitcoin/
│   │   │   └── btcusd_1-min_data.csv    # 1-dakikalık BTC/USD fiyat verisi
│   │   └── tweets/
│   │       └── tweets.csv               # Ham Twitter veri seti (semi-colon sep)
│   └── processed/                       # Önişlemden geçmiş ara dosyalar
│       ├── bitcoin_hourly_features.csv  # Saatlik fiyat & teknik analiz tabloları
│       ├── btc_sentiment_hourly.csv     # FinBERT ile hesaplanmış saatlik sentiment
│       ├── btc_sentiment_raw_sample.csv # Sınıflandırılmış örnek tweet kayıtları
│       └── final_multimodal_dataset.csv # Birleştirilmiş nihai model girdi tablosu
├── models/                              # Kaydedilmiş model parametreleri ve scaler'lar
│   ├── feature_scaler.pkl               # MinMaxScaler nesnesi (Öznitelikler için)
│   ├── target_scaler.pkl                # MinMaxScaler nesnesi (Hedef değişken için)
│   ├── random_forest_baseline.pkl       # Eğitilmiş Random Forest modeli
│   ├── lstm_volatility_model_pytorch.pt # PyTorch LSTM model ağırlıkları
│   └── cnn_bilstm_volatility_model_pytorch.pt # PyTorch CNN-BiLSTM model ağırlıkları
├── scripts/                             # Modüler python iş akışları
│   ├── price_preprocess.py              # Fiyat önişleme ve ta kütüphanesi hesapları
│   ├── tweet_sentiment_preprocess.py    # Tweet temizliği, FinBERT batch inference ve gruplama
│   ├── merge_price_sentiment.py         # Zaman damgalı multimodal veri birleştirme
│   ├── train_baseline_models.py         # Sklearn modellerinin eğitimi ve metrik kaydı
│   ├── train_lstm_model.py              # PyTorch LSTM eğitim döngüsü ve değerlendirme
│   ├── train_cnn_bilstm_model.py        # PyTorch CNN-BiLSTM eğitim döngüsü ve değerlendirme
│   └── create_project_visualizations.py # Grafik çıktılarının PNG formatında üretimi
├── outputs/                             # Model çıktıları ve performans görselleri
│   ├── baseline_results.csv
│   ├── lstm_results.csv
│   ├── cnn_bilstm_results.csv
│   ├── lstm_training_loss.png
│   ├── cnn_bilstm_training_loss.png
│   └── visualizations/                   # Karşılaştırma ve zaman serisi grafikleri
│       ├── comparison_rmse.png
│       ├── comparison_mae.png
│       ├── comparison_r2.png
│       ├── timeseries_close.png
│       ├── timeseries_rsi.png
│       └── timeseries_sentiment.png
├── app.py                               # Streamlit Web Dashboard Uygulaması
├── requirements.txt                     # Kütüphane bağımlılık listesi
└── README.md                            # Akademik dökümantasyon (Bu dosya)
```

---

## 5. Kurulum ve Model Çalıştırma Pipeline'ı

Proje bağımlılıklarını izole bir ortamda kurup veri hattını baştan uca çalıştırmak için aşağıdaki adımları sırasıyla uygulayınız.

### 5.1. Sanal Ortam Kurulumu ve Bağımlılıklar (macOS / Linux)
```bash
# Proje dizinine geçiş yapın
cd ~/Desktop/yzproje

# Python sanal ortamı oluşturun
python3 -m venv .venv
source .venv/bin/activate

# Paketleri yükleyin
pip install -r requirements.txt
```
*Not: Eğer `requirements.txt` bulunmuyorsa, temel kütüphaneleri manuel olarak yükleyebilirsiniz:*
```bash
pip install pandas numpy scikit-learn torch transformers matplotlib streamlit plotly joblib ta
```

### 5.2. Veri İşleme ve Model Eğitimi Sıralaması
Veri hattının tutarlı çalışması için scriptlerin aşağıdaki sırada çalıştırılması zorunludur:

1. **Fiyat Verisi Hazırlama ve İndikatör Hesaplama:**
   ```bash
   python scripts/price_preprocess.py
   ```
   *Çıktı:* `data/processed/bitcoin_hourly_features.csv`

2. **Tweet Sentiment Analizi (FinBERT Inference):**
   *(Büyük veri setleri için GPU veya CPU batch işleme devrededir)*
   ```bash
   python scripts/tweet_sentiment_preprocess.py
   ```
   *Çıktı:* `data/processed/btc_sentiment_hourly.csv`

3. **Multimodal Veri Füzyonu:**
   ```bash
   python scripts/merge_price_sentiment.py
   ```
   *Çıktı:* `data/processed/final_multimodal_dataset.csv`

4. **Baseline Modellerin Eğitimi (Linear Reg & Random Forest):**
   ```bash
   python scripts/train_baseline_models.py
   ```
   *Çıktı:* `models/random_forest_baseline.pkl`, `outputs/baseline_results.csv`

5. **LSTM Derin Öğrenme Modelinin Eğitimi:**
   ```bash
   python scripts/train_lstm_model.py
   ```
   *Çıktı:* `models/lstm_volatility_model_pytorch.pt`, `outputs/lstm_results.csv`, `outputs/lstm_training_loss.png`

6. **CNN-BiLSTM Derin Öğrenme Modelinin Eğitimi:**
   ```bash
   python scripts/train_cnn_bilstm_model.py
   ```
   *Çıktı:* `models/cnn_bilstm_volatility_model_pytorch.pt`, `outputs/cnn_bilstm_results.csv`, `outputs/cnn_bilstm_training_loss.png`

7. **Proje Görselleştirmelerinin Oluşturulması:**
   ```bash
   python scripts/create_project_visualizations.py
   ```

---

## 6. Streamlit İnteraktif Analiz Paneli (Dashboard)

Model sonuçlarını interaktif olarak incelemek, zaman serisi grafiklerini gözlemlemek ve performans karşılaştırması yapmak için Streamlit tabanlı bir web arayüzü kodlanmıştır.

Arayüzü ayağa kaldırmak için:
```bash
streamlit run app.py
```

### Dashboard Yetenekleri:
- **Veri Dağılım Grafikleri:** Bitcoin kapanış fiyat trendi, saatlik tweet yoğunluğu ve ortalama sentiment skoru dinamik zaman serisi üzerinde gösterilir.
- **Model Karşılaştırma Matrisi:** MAE, RMSE ve $R^2$ metrikleri otomatik olarak tablolandırılır ve RMSE metriğine göre **En İyi Model** vurgulanır.
- **Eğitim İlerlemesi:** Eğitilen derin öğrenme modellerinin loss (kayıp) gelişim grafikleri görsel olarak sunulur.

---

## 7. Proje Sınırlılıkları ve Gelecek Çalışmalar

- **Veri Örnekleme Kısıtı:** Twitter veri setinin büyüklüğünden ötürü FinBERT çıkarımı deneysel amaçla 20.000 adet tweet ile sınırlandırılmıştır. Verinin tamamının (özellikle yüksek etkileşimli dönemlerin) işlenmesi model başarısını artıracaktır.
- **Duygu Seviyesi Seyrekliği (Sparsity):** Fiyat verisi kesintisiz 24 saat akarken, tweet verisi belirli saatlerde yoğunlaşmakta, bazı saatlerde ise boş kalmaktadır. Gelecek çalışmalarda bu boş saatlerin doldurulması için enterpolasyon veya ileriye doğru taşıma (propagation) teknikleri araştırılabilir.
- **Alternatif Veri Modaliteleri:** Modele sadece Twitter verisi değil, Reddit tartışmaları, Google Trends verileri ve blokzincir on-chain metrikleri (örneğin hash rate, aktif cüzdan sayıları) entegre edilerek çok modlu veri çeşitliliği genişletilebilir.
- **İleri Modeller:** Transformer tabanlı zaman serisi mimarileri (Temporal Fusion Transformer - TFT, Informer vb.) volatilite tahmininde uzun dönemli bağımlılıkları öğrenmede alternatif olarak denenebilir.

---

## 📑 Akademik Kaynakça

1. Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). *BERT: Pre-training of deep bidirectional transformers for language understanding*. arXiv preprint arXiv:1810.04805.
2. Araci, D. (2019). *FinBERT: Financial sentiment analysis with pre-trained language models*. arXiv preprint arXiv:1908.10063.
3. Hochreiter, S., & Schmidhuber, J. (1997). *Long short-term memory*. Neural computation, 9(8), 1735-1780.
4. Kim, Y. (2014). *Convolutional neural networks for sentence classification*. arXiv preprint arXiv:1408.5882.
5. Taylor, S. J. (2007). *Asset price dynamics, volatility, and prediction*. Princeton University Press.
