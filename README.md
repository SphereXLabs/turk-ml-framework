# TURK Dili: Türkçe Yapay Zeka Programlama Dili

<img src="docs/images/turk-logo.png" alt="TURK Dil Logo" width="300"/>

TURK, Türkçe sözdizimi kullanarak yapay zeka modelleri geliştirmenizi sağlayan özgün bir programlama dilidir. Türkçe anahtar kelimeler ve doğal dil yapısıyla, yapay zeka projelerini ana dilimizde geliştirmeyi kolaylaştırır.

## 📋 İçindekiler

- [Genel Bakış](#genel-bakış)
- [Özellikler](#özellikler)
- [Kurulum](#kurulum)
- [Kullanım](#kullanım)
- [Örnekler](#örnekler)
- [Belgelendirme](#belgelendirme)
- [Katkıda Bulunma](#katkıda-bulunma)
- [Lisans](#lisans)

## 🔍 Genel Bakış

TURK dili, yapay zeka modellemesi için tamamen Türkçe dilinde tasarlanmış bir programlama dilidir. Python veya diğer dillerde yazılmış yapay zeka kütüphanelerine alternatif olarak, makine öğrenmesi ve derin öğrenme projelerini Türkçe sözdizimi ile geliştirebilmenizi sağlar.

TURK dilinin temel amacı:
- Türkçe doğal dilinde programlama yapabilmek
- Yapay zeka kavramlarını anadilimizde öğretmek
- Türkçe eğitim materyalleri ile bilgi paylaşımını artırmak
- Türkçe yapay zeka ekosistemini geliştirmek

## ✨ Özellikler

TURK dili aşağıdaki temel özellikleri içerir:

### 1. Tamamen Türkçe Sözdizimi
```turk
Model Siniflandirici {
    Katman(784, 128, aktivasyon=Relu)
    Katman(128, 10, aktivasyon=Softmax)
    Optimizer = SGD(ogrenme_orani=0.01)
    Kayip = CrossEntropy
}

Egit(Siniflandirici, veri_x, veri_y, epoch=10)
```

### 2. Zengin Yapay Zeka Bileşenleri
- **Sinir Ağı Modelleri**: DNN, CNN, RNN, LSTM, GRU, Transformer
- **Öğrenme Algoritmaları**: SGD, Adam, AdamW, RMSProp
- **Kayıp Fonksiyonları**: MSE, CrossEntropy, BinaryCrossEntropy
- **Metrikler**: Doğruluk, Hassasiyet, Duyarlılık, F1-Skoru

### 3. Veri İşleme Özellikleri
- Veri yükleme ve ön işleme
- Görselleştirme araçları
- İstatistiksel fonksiyonlar
- Metin ve görüntü işleme

### 4. Ekosistem
- Model kaydetme ve yükleme
- Eğitim izleme ve raporlama
- Hazır model kütüphanesi
- Eğitim ve test araçları

## 🚀 Kurulum

### 1. Bağımlılıklar

TURK dilini kullanmak için aşağıdaki bağımlılıklara ihtiyacınız vardır:
- Rust 1.50 veya üzeri
- Cargo (Rust Paket Yöneticisi)

### 2. Kurulum Adımları

```bash
# TURK derleyicisini ve kütüphanelerini yükleyin
git clone https://github.com/NeonChipher/turk-ml-framework.git
cd turk-ml-framework

# Derleyin
cargo build --release

# Komut satırı aracını sistem yoluna ekleyin
cargo install --path .
```

### 3. Kullanım Doğrulama

```bash
turk --version
# Çıktı: TURK v0.1.0
```

## 💻 Kullanım

### 1. Temel Kod Yapısı

Bir TURK dosyası (.turk uzantılı) aşağıdaki temel yapıya sahiptir:

```turk
// Veri yükleme ve hazırlama
Veri egitim = Veri_Yukle("egitim_verisi.csv")
Veri test = Veri_Yukle("test_verisi.csv")

// Model tanımlama
Model YeniModel {
    // Katmanlar ve parametreler
    Katman(100, 50, aktivasyon=Relu)
    Katman(50, 10, aktivasyon=Softmax)
    
    // Hiperparametreler
    Optimizer = Adam(ogrenme_orani=0.001)
    Kayip = CrossEntropy
    Batch = 32
    Epoch = 5
}

// Eğitim
Egit(YeniModel, egitim.x, egitim.y)

// Değerlendirme
sonuclar = Degerlendir(YeniModel, test.x, test.y)
Yazdir("Doğruluk:", sonuclar.dogruluk)

// Tahmin
tahminler = Tahmin(YeniModel, yeni_veri)

// Kaydetme
Kaydet(YeniModel, "modeller/yeni_model.turk")
```

### 2. Komut Satırı Arayüzü

```bash
# Bir TURK programını çalıştırma
turk calistir ornekler/mnist_classifier.turk

# Interaktif kabuk başlatma
turk kabuk

# Yardım bilgisi alma
turk yardim
```

## 📚 Örnekler

TURK dilinde yazılmış örnek uygulamalar:

### 1. MNIST Rakam Sınıflandırıcı
[`ornekler/turk_mnist.turk`](ornekler/turk_mnist.turk)
```turk
// MNIST veri kümesi üzerinde derin sinir ağı
Veri mnist = Veri_Yukle("mnist.csv")
Veri egitim, test = Veri_Bol(mnist, oran=0.8)

Model RakamSiniflandirici {
    Katman(784, 128, aktivasyon=Relu)
    Katman(128, 64, aktivasyon=Relu)
    Katman(64, 10, aktivasyon=Softmax)
    
    Optimizer = Adam(ogrenme_orani=0.001)
    Kayip = CrossEntropy
}

Egit(RakamSiniflandirici, egitim.x, egitim.y)
```

### 2. Duygu Analizi
[`ornekler/turk_duygu_analizi.turk`](ornekler/turk_duygu_analizi.turk)
```turk
// Türkçe film yorumlarında duygu analizi
Veri yorumlar = Veri_Yukle("turk_film_yorumlari.csv")
egitim.x = MetinVektorlestir(egitim.x, yontem="tfidf")

Model DuyguAnalizi {
    LSTM { gizli_boyut = 64, bidirectional = Dogru }
    Dropout(0.3)
    Katman(64, 2, aktivasyon=Softmax)
    
    Optimizer = Adam(ogrenme_orani=0.001)
    Kayip = BinaryCrossEntropy
}
```

### 3. CNN ile Görüntü Sınıflandırma
[`ornekler/turk_cnn_goruntu.turk`](ornekler/turk_cnn_goruntu.turk)
```turk
// CIFAR-10 veri kümesi üzerinde CNN
Model CIFAR_Siniflandirici {
    // Evrişimli katmanlar
    Katman(tip = "Evrisimli", filtre_sayisi = 32, ...)
    Katman(tip = "Havuzlama", boyut = (2, 2), ...)
    
    // Tam bağlantılı katmanlar
    Katman(512, aktivasyon=Relu)
    Katman(10, aktivasyon=Softmax)
}
```

### 4. Zaman Serisi Tahmini
[`ornekler/turk_zaman_serisi.turk`](ornekler/turk_zaman_serisi.turk)
```turk
// Borsa verisi tahmini
Model BorsaTahmin {
    LSTM { gizli_boyut = 50, katman_sayisi = 2 }
    Katman(50, 1, aktivasyon=Dogrusal)
    
    Optimizer = Adam(ogrenme_orani=0.001)
    Kayip = MSE
}
```

## 📖 Belgelendirme

Detaylı belgelendirme için [docs/](docs/) dizinine bakın:

- [Dil Referansı](docs/dil_referansi.md)
- [API Belgeleri](docs/api.md)
- [Eğitimler](docs/egitimler/)
- [En İyi Uygulamalar](docs/en_iyi_uygulamalar.md)

## 👥 Katkıda Bulunma

TURK diline katkıda bulunmak isterseniz [CONTRIBUTING.md](CONTRIBUTING.md) belgesini okuyun. Katkılarınızı bekliyoruz!

## 📄 Lisans

Bu proje MIT lisansı altında lisanslanmıştır. Detaylar için [LICENSE](LICENSE) dosyasına bakın.


""""""""""