# TÜRK - Türkçe Yapay Zeka Dili

[![CI](https://github.com/SphereXLabs/turk-ml-framework/actions/workflows/ci.yml/badge.svg)](https://github.com/SphereXLabs/turk-ml-framework/actions/workflows/ci.yml)
[![Docs / Pages](https://github.com/SphereXLabs/turk-ml-framework/actions/workflows/docs.yml/badge.svg)](https://spherexlabs.github.io/turk-ml-framework/turk/)
[![Rust](https://img.shields.io/badge/Rust-1.75%2B-orange.svg)](https://www.rust-lang.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![crates.io](https://img.shields.io/crates/v/turk.svg)](https://crates.io/crates/turk)

TÜRK, Türkçe kullanarak yapay sinir ağları oluşturmanızı, eğitmenizi ve değerlendirmenizi sağlayan açık kaynaklı bir yapay zeka kütüphanesi ve DSL (Domain Specific Language) projesidir.

> **v0.3.0 — Göktürk Kokpit:** Yerel semantik hafıza, otonom araç zinciri, sistem paneli (`turk_sistem`), WASM tarayıcı playground (canlı tahmin), `f32` / BLAS hızlandırma ve modernize CI/CD dahildir.

> [!IMPORTANT]
> ### Canlı Tarayıcı Demosu
>
> Kurulum olmadan Türkçe komutlarla eğitip tahmin edin — ALPER (yerel `turk` sinir beyni, harici LLM yok) + ML monitör:
>
> **[▶ Göktürk Kokpit (ALPER)](https://spherexlabs.github.io/turk-ml-framework/kokpit/)** &nbsp;·&nbsp; **[WASM Playground](https://spherexlabs.github.io/turk-ml-framework/playground/)** &nbsp;·&nbsp; **[API Docs (`turk`)](https://spherexlabs.github.io/turk-ml-framework/turk/)**

## 🎯 Vizyon ve Felsefe

TÜRK projesi, Türkçe odaklı bir yapay zeka dili ve ekosistemi oluşturmayı hedefler:

1. **Kendi Dilimiz İle Geliştirme**: Python gibi diğer dillerin yapay zeka geliştirmede kullanıldığı gibi, biz de kendi Türkçe tabanlı programlama dilimizi bu alanda kullanmak istiyoruz.

2. **Kendi Yapay Zekamızı Kendi Dilimizle Eğitmek**: Geliştirdiğimiz dil modeliyle kendi yapay zeka modellerimizi oluşturmak ve eğitmek hedefimiz. Bu süreçte dilimizin gelişimi ve yapay zeka modellerimizin performansı birbirini besleyecek.

3. **Topluluk Odaklı Erişim**: Diğer insanların da bu dil ile kendi modellerini geliştirmesini, kendilerini geliştirmesini ve böylece Türkçe yapay zeka ekosistemine katkıda bulunmasını istiyoruz.

Bu vizyon doğrultusunda, TÜRK projesi yalnızca bir dil veya kütüphane değil, Türkçe konuşan topluluklar için kapsamlı bir yapay zeka geliştirme platformu olmayı amaçlar.

## 🌟 Özellikler

- **Türkçe Programlama Dili**: Yapay sinir ağlarını Türkçe komutlarla tanımlayın ve eğitin
- **WASM Playground & ALPER Kokpit**: Tarayıcıda DSL + yerli niyet beyni (TF-IDF → `turk` MLP) / açık web araştırma / canlı çıkarım
- **Kolay Kullanım**: Basit sözdizimi ile karmaşık yapay sinir ağları oluşturun
- **Çeşitli Ağ Mimarileri**: Yoğun katmanlar, dropout, aktivasyon fonksiyonları ve daha fazlası
- **Farklı Optimizasyon Algoritmaları**: SGD, Adam, AdamW, RMSProp
- **Kapsamlı Kayıp Fonksiyonları**: MSE, MAE, Cross Entropy, Binary Cross Entropy
- **Metrikler**: Doğruluk, kesinlik, duyarlılık, F1-skoru (ikili sınıflandırmada 0.5 eşiği)
- **Hassasiyet & Hız**: Varsayılan `f64`; isteğe bağlı `f32` ve OpenBLAS
- **Serileştirme**: Modelleri JSON formatında kaydedin ve yükleyin
- **Ön İşleme Araçları**: Veri normalleştirme, standartlaştırma ve daha fazlası
- **MNIST ve Diğer Veri Setleri**: Örnekler ve başlangıç kodları

## 🚀 Hızlı Başlangıç

### Kurulum

TÜRK projesini kurmak için aşağıdaki hızlı adımları izleyebilirsiniz:

```bash
# Cargo ile kurulum
cargo install turk

# Veya kaynak koddan derleyin
git clone https://github.com/SphereXLabs/turk-ml-framework.git
cd turk-ml-framework
cargo build --release

# Göktürk Kokpit (v0.3.0) — Windows (yerel egui)
.\scripts\cargo.ps1 build --features turk_sistem
.\scripts\turk-sistem.ps1
.\scripts\alper.ps1

# Yardımcı scriptler: scripts/  ·  örnek .turk dosyaları: examples/
# Kokpit (ALPER): https://spherexlabs.github.io/turk-ml-framework/kokpit/
```

Detaylı kurulum talimatları, farklı işletim sistemleri için özel adımlar ve sorun giderme için [INSTALL.md](INSTALL.md) dosyasına bakınız.

### Cargo özellikler (`features`)

| Feature | Açıklama |
|---------|----------|
| *(varsayılan)* | `f64` hassasiyet + CSV veri akışı |
| `f32` | Tek hassasiyet (`Float = f32`) — daha az bellek, daha hızlı SIMD |
| `blas` | Yerel OpenBLAS hızlandırma (`ndarray/blas`). **WASM ile birlikte açmayın.** |
| `wasm` | Tarayıcı playground köprüsü (`wasm-bindgen`) |
| `turk_sistem` | Göktürk Kokpit GUI + sistem otomasyonu (Windows odaklı) |
| `gui` / `cli` / `full` | Arayüz ve yardımcı araç setleri |

```bash
# f32 hassasiyet
cargo build --release --features f32

# OpenBLAS hızlandırma (sistemde OpenBLAS gerekir)
cargo build --release --features blas

# f32 + BLAS birlikte
cargo build --release --features "f32,blas"

# WASM playground paketi (disk I/O yok; sentetik xor / kume)
rustup target add wasm32-unknown-unknown
cargo check --target wasm32-unknown-unknown --no-default-features --features wasm
# veya wasm-pack:
# wasm-pack build --target web --out-dir playground/pkg -- --no-default-features --features wasm
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

### .turk Dosyası ile Çalışma

TÜRK dili ile doğrudan `.turk` uzantılı dosyalar yazabilir ve çalıştırabilirsiniz:

```
// model.turk
model_olustur()
katman_ekle(yogun(784, 128, "relu"))
katman_ekle(dropout(0.3))
katman_ekle(yogun(128, 64, "relu"))
katman_ekle(yogun(64, 10, "softmax"))
derle(adam(0.001))
veri_yukle("mnist")
egit(10, 32)
model_bilgisi()
```

Bu dosyayı çalıştırmak için:

```bash
turk calistir model.turk
```

Veya Rust kodundan:

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

TÜRK dili, yapay zeka modelleri oluşturmak ve eğitmek için doğal Türkçe sözdizimi kullanır:

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

Bu sözdizimi, makine öğrenimi kavramlarını Türkçe dilinin yapısına uygun şekilde ifade etmek için özel olarak tasarlanmıştır.

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

Aynı örneği TÜRK diliyle de yazabiliriz:

```
// mnist.turk
model_olustur()
katman_ekle(yogun(784, 128, "relu"))
katman_ekle(dropout(0.3))
katman_ekle(yogun(128, 64, "relu"))
katman_ekle(yogun(64, 10, "softmax"))
derle(adam(0.001), capraz_entropi())
veri_yukle("mnist")
egit(5, 32)
test_kaybi = degerlendir(x_test, y_test)
yazdir("Test kaybı: " + test_kaybi)
```

Daha fazla örnek için `examples/` dizinine bakın.

## 📋 Desteklenen Operasyonlar ve İşlevler

### Katmanlar

- `yogun(girdi, cikti, aktivasyon)`: Tam bağlantılı katman (Dense)
- `dropout(oran)`: Aşırı uyum önleme için dropout katmanı
- `konvolusyon2d(filtre, boyut, adim)`: 2D konvolüsyon katmanı
- `maksimum_havuzlama(boyut, adim)`: Maksimum havuzlama katmanı
- `durlama(boyut)`: Normalizasyon için durlama (batch normalization) katmanı

### Aktivasyon Fonksiyonları

- `dogrusal`: Doğrusal aktivasyon (varsayılan)
- `relu`: Rectified Linear Unit
- `sizintili_relu(alpha)`: Sızıntılı ReLU
- `sigmoid`: Sigmoid fonksiyonu
- `tanh`: Hiperbolik tanjant
- `softmax`: Softmax fonksiyonu
- `swish`: Swish aktivasyon fonksiyonu

### Optimizasyon Algoritmaları

- `sgd(ogrenme_hizi, momentum)`: Stokastik Gradyan İnişi
- `adam(ogrenme_hizi, beta1, beta2)`: Adaptive Moment Estimation
- `adamw(ogrenme_hizi, agirlik_zayiflatma)`: Adam ile ağırlık zayıflatma
- `rmsprop(ogrenme_hizi)`: Root Mean Square Propagation

### Kayıp Fonksiyonları

- `mse()`: Ortalama Kare Hata (Mean Squared Error)
- `mae()`: Ortalama Mutlak Hata (Mean Absolute Error)
- `capraz_entropi()`: Çapraz Entropi Kaybı
- `ikili_capraz_entropi()`: İkili Çapraz Entropi Kaybı
- `huber_kaybi(delta)`: Huber Kaybı

### Metrikler

- `dogruluk()`: Doğruluk (Accuracy)
- `kesinlik()`: Kesinlik (Precision)
- `duyarlilik()`: Duyarlılık (Recall)
- `f1_skoru()`: F1 Skoru

### Ön İşleme

- `standartlastir(veri)`: Standartlaştırma (ortalama=0, std=1)
- `normalize(veri, min, max)`: Min-max normalizasyonu
- `one_hot_kodla(etiketler, sinif_sayisi)`: One-hot kodlama

## 🛠️ Temel Bileşenler

TÜRK projesi üç ana bileşenden oluşur:

1. **TÜRK Dili**: Yapay sinir ağları tanımlamak için Türkçe tabanlı DSL
2. **TÜRK Kütüphanesi**: Modellerin arka planda çalışmasını sağlayan Rust tabanlı kütüphane
3. **TÜRK Araçları**: Model geliştirme, eğitim ve değerlendirme için yardımcı araçlar

Bu bileşenler birlikte çalışarak, yapay zeka modellerinin Türkçe programlama dili kullanılarak geliştirilmesini sağlar.

## 📊 Performans

TÜRK kütüphanesi, ndarray ve Rust'ın hız avantajlarını kullanarak verimli bir şekilde çalışır. Temel performans karşılaştırmaları:

| Veri Seti         | TÜRK  | TensorFlow | PyTorch |
|-------------------|-------|------------|---------|
| MNIST (MLP)       | 3.2s  | 2.8s       | 3.0s    |
| Fashion MNIST     | 3.8s  | 3.2s       | 3.5s    |
| CIFAR-10          | 12.5s | 8.3s       | 9.1s    |

*Not: Performans, 5 epoch, 32 batch_size ve aynı model mimarisi ile ölçülmüştür.*

## 🤝 Topluluk ve Katkıda Bulunma

TÜRK projesi, topluluk odaklı gelişmeyi hedefler. Türkçe konuşan geliştiricilerin yapay zeka alanında kendi dillerinde çalışabilmeleri için bir platform oluşturmak istiyoruz.

Katkıda bulunmak için:

1. Bu repo'yu fork edin
2. Yeni bir dal oluşturun (`git checkout -b ozellik/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik: X'`)
4. Dalınıza push yapın (`git push origin ozellik/yeni-ozellik`)
5. Pull Request gönderin

### Katkı Alanları

- **Dil Genişletmeleri**: TÜRK dilinin sözdizimini ve özelliklerini geliştirme
- **Model Ekosistemi**: Önceden eğitilmiş modeller ve uygulama örnekleri ekleme
- **Dokümantasyon**: Türkçe kaynak ve öğreticiler oluşturma
- **Eğitim Materyalleri**: Yapay zeka ve makine öğrenimi konularında Türkçe eğitim içerikleri
- **Topluluk Etkinlikleri**: Çalıştaylar, webinarlar ve görüşmeler düzenleme

Daha fazla bilgi için [CONTRIBUTING.md](CONTRIBUTING.md) dosyasına bakın.

## 🎓 Eğitim ve Öğrenme

TÜRK projesinin önemli bir hedefi, yapay zeka ve makine öğrenimi kavramlarını Türkçe diliyle öğretmektir. Bu amaçla:

- 📚 **Türkçe Eğitim Materyalleri**: Temel kavramları anlatan dokümantasyon
- 🎬 **Video Eğitimleri**: Adım adım öğreticiler
- 📝 **Interaktif Örnekler**: Çevrimiçi denemeler yapabileceğiniz Jupyter notebook benzeri araçlar
- 🏫 **Topluluk Atölyeleri**: Düzenli eğitim etkinlikleri

## 📄 Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - detaylar için [LICENSE](LICENSE) dosyasına bakın.

## 📧 İletişim

- **Instagram:** [@sinankaradayi35](https://instagram.com/sinankaradayi35)

## 🌐 İlgili Projeler

- [TensorFlow](https://www.tensorflow.org/)
- [PyTorch](https://pytorch.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Rust-BLAS](https://github.com/blas-lapack-rs/blas)
- [Türkçe NLP Projeleri](https://github.com/topics/turkish-nlp)

---

Türkiye'de ❤️ ile geliştirilmiştir