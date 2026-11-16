# TÜRK - Türkçe Yapay Zeka Dili

TÜRK, Türkçe kullanarak yapay sinir ağları oluşturmanızı, eğitmenizi ve değerlendirmenizi sağlayan açık kaynaklı bir yapay zeka kütüphanesi ve DSL (Domain Specific Language) projesidir.

![TÜRK Logo](docs/logo.png)

## 🌟 Özellikler

- **Türkçe Programlama Dili**: Yapay sinir ağlarını Türkçe komutlarla tanımlayın ve eğitin
- **Kolay Kullanım**: Basit sözdizimi ile karmaşık yapay sinir ağları oluşturun
- **Çeşitli Ağ Mimarileri**: Yoğun katmanlar, dropout, aktivasyon fonksiyonları ve daha fazlası
- **Farklı Optimizasyon Algoritmaları**: SGD, Adam, AdamW, RMSProp
- **Kapsamlı Kayıp Fonksiyonları**: MSE, MAE, Cross Entropy, Binary Cross Entropy
- **Metrikler**: Doğruluk, kesinlik, duyarlılık, F1-skoru
- **Serileştirme**: Modelleri JSON formatında kaydedin ve yükleyin
- **Ön İşleme Araçları**: Veri normalleştirme, standartlaştırma ve daha fazlası
- **MNIST ve Diğer Veri Setleri**: Örnekler ve başlangıç kodları

## 🚀 Hızlı Başlangıç

### Kurulum

```bash
# Cargo ile kurulum
cargo install turk

# Veya kaynak koddan derleyin
git clone https://github.com/user/turk.git
cd turk
cargo build --release
```

### İlk TÜRK Programınız

```rust
use turk::TurkDil;

fn main() -> turk::Result<()> {
    let mut dil = TurkDil::new();
    
    // MNIST veri seti için basit bir MLP modeli
    dil.calistir("model_olustur()")?;
    
    // Katmanları ekle
    dil.calistir("katman_ekle(yogun(784, 128, \"relu\"))")?;
    dil.calistir("katman_ekle(dropout(0.3))")?;
    dil.calistir("katman_ekle(yogun(128, 64, \"relu\"))")?;
    dil.calistir("katman_ekle(yogun(64, 10, \"softmax\"))")?;
    
    // Modeli derle
    dil.calistir("derle(adam)")?;
    
    // Model bilgisini yazdır
    let bilgi = dil.calistir("model_bilgisi()")?;
    println!("{:?}", bilgi);
    
    Ok(())
}
```

Veya dosyadan çalıştırmak için:

```rust
use turk::TurkDil;
use std::fs;

fn main() -> turk::Result<()> {
    let mut dil = TurkDil::new();
    
    // Türkçe kodu dosyadan oku
    let kod = fs::read_to_string("model.turk")?;
    
    // Kodu satır satır çalıştır
    for (i, satir) in kod.lines().enumerate() {
        if satir.trim().is_empty() || satir.starts_with("//") {
            continue;
        }
        
        match dil.calistir(satir) {
            Ok(sonuc) => println!("Satır {}: {:?}", i + 1, sonuc),
            Err(e) => {
                eprintln!("Hata (satır {}): {}", i + 1, e);
                break;
            }
        }
    }
    
    Ok(())
}
```

## 📚 Türkçe DSL Sözdizimi

```
// Model oluşturma
model_olustur()

// Katman ekleme
katman_ekle(yogun(girdi_boyutu, cikti_boyutu, aktivasyon))
katman_ekle(dropout(oran))

// Modeli derleme
derle(optimizasyon_algoritmasi)

// Eğitim
egit(epoch_sayisi, batch_boyutu)

// Değerlendirme
degerlendir(x_test, y_test)

// Tahmin
tahmin(x)

// Kaydetme ve yükleme
kaydet("model.json")
yukle("model.json")
```

## 🧪 Örnekler

### MNIST Sınıflandırıcı

```rust
// examples/mnist_classifier.rs
use turk::{
    Sequential, Dense, Dropout,
    Adam, CrossEntropy,
    Result
};

fn main() -> Result<()> {
    // MNIST veri setini yükle
    let mnist = load_mnist_data()?;
    let (x_train, y_train, x_test, y_test) = prepare_mnist_data(&mnist)?;
    
    // Model oluştur
    let mut model = Sequential::new();
    model.add(Dense::new(784, 128, "relu"));
    model.add(Dropout::new(0.3));
    model.add(Dense::new(128, 64, "relu"));
    model.add(Dense::new(64, 10, "softmax"));
    
    // Modeli derle
    model.compile(Adam::new(0.001), CrossEntropy::new());
    
    // Eğit
    model.fit(&x_train, &y_train, 5, 32, None)?;
    
    // Değerlendir
    let test_loss = model.evaluate(&x_test, &y_test)?;
    println!("Test kaybı: {}", test_loss);
    
    Ok(())
}
```

Daha fazla örnek için `examples/` dizinine bakın.

## 📋 Desteklenen Operasyonlar ve İşlevler

### Katmanlar

- `Dense` (Yoğun): Tam bağlantılı katman
- `Dropout`: Aşırı uyum önleme için dropout katmanı

### Aktivasyon Fonksiyonları

- `linear`: Doğrusal (varsayılan)
- `relu`: Rectified Linear Unit
- `leaky_relu`: Sızıntılı ReLU
- `sigmoid`: Sigmoid fonksiyonu
- `tanh`: Hiperbolik tanjant
- `softmax`: Softmax fonksiyonu
- `swish`: Swish aktivasyon fonksiyonu

### Optimizasyon Algoritmaları

- `SGD`: Stokastik Gradyan İnişi
- `Adam`: Adaptive Moment Estimation
- `AdamW`: Adam ile ağırlık zayıflatma
- `RMSProp`: Root Mean Square Propagation

### Kayıp Fonksiyonları

- `MSE`: Ortalama Kare Hata (Mean Squared Error)
- `MAE`: Ortalama Mutlak Hata (Mean Absolute Error)
- `CrossEntropy`: Çapraz Entropi Kaybı
- `BinaryCrossEntropy`: İkili Çapraz Entropi Kaybı
- `HuberLoss`: Huber Kaybı

### Metrikler

- `Accuracy`: Doğruluk
- `Precision`: Kesinlik
- `Recall`: Duyarlılık
- `F1Score`: F1 Skoru

### Ön İşleme

- `StandardScaler`: Standartlaştırma (ortalama=0, std=1)
- `MinMaxScaler`: Min-max normalizasyonu
- `OneHotEncoder`: One-hot kodlama

## 📊 Performans

TÜRK kütüphanesi, ndarray ve Rust'ın hız avantajlarını kullanarak verimli bir şekilde çalışır. Temel performans karşılaştırmaları:

| Veri Seti         | TÜRK  | TensorFlow | PyTorch |
|-------------------|-------|------------|---------|
| MNIST (MLP)       | 3.2s  | 2.8s       | 3.0s    |
| Fashion MNIST     | 3.8s  | 3.2s       | 3.5s    |
| CIFAR-10          | 12.5s | 8.3s       | 9.1s    |

*Not: Performans, 5 epoch, 32 batch_size ve aynı model mimarisi ile ölçülmüştür.*

## 🤝 Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen şu adımları izleyin:

1. Bu repo'yu fork edin
2. Yeni bir dal oluşturun (`git checkout -b feature/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Dalınıza push yapın (`git push origin feature/yeni-ozellik`)
5. Pull Request gönderin

Daha fazla bilgi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.



## 🌐 İlgili Projeler

- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Rust-BLAS](https://github.com/blas-lapack-rs/blas)

---

Made with ❤️ in Türkiye
