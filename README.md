# TURK Dili

![CI Status](https://github.com/snnkr/turk-lang/actions/workflows/ci.yml/badge.svg)
[![Crates.io](https://img.shields.io/crates/v/turk-lang.svg)](https://crates.io/crates/turk-lang)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

TURK, Türkçe diline özgü makine öğrenimi programlama dilidir. Yapay zeka ve derin öğrenme modellerini Türkçe sözdizimi kullanarak oluşturmanıza, eğitmenize ve değerlendirmenize olanak tanır.

## Özellikler

- 🧠 Kolay anlaşılabilir Türkçe sözdizimi ile yapay sinir ağları oluşturma
- 📊 Çeşitli katman tipleri (Dense, Dropout, BatchNorm, vb.)
- 🔄 Farklı aktivasyon fonksiyonları (ReLU, Sigmoid, Tanh, Softmax)
- 📈 Çeşitli optimizasyon algoritmaları (SGD, Adam, AdamW, RMSProp)
- 🎯 Kapsamlı kayıp fonksiyonları (MSE, CrossEntropy)
- 📏 Performans metrikleri (Accuracy, Precision, Recall, F1Score)
- 💾 Model kaydetme ve yükleme desteği
- 🚀 Rayon ile çoklu çekirdek paralelleştirme desteği

## Kurulum

Cargo.toml dosyanıza ekleyin:

```toml
[dependencies]
turk-lang = "0.1.0"
```

## Hızlı Başlangıç

### Temel Kullanım

```rust
use turk_lang::{Sequential, Dense, Activation, Adam, Result};
use ndarray::Array2;

fn main() -> Result<()> {
    // Model oluştur
    let mut model = Sequential::new();
    model.add(Dense::new(784, 128, Activation::Relu))
         .add(Dense::new(128, 10, Activation::Softmax))
         .compile(Adam::new(0.001));
    
    // Model özetini göster
    println!("{}", model.summary());
    
    // Veri hazırla (örnek)
    let x_train = Array2::zeros((100, 784));
    let y_train = Array2::zeros((100, 10));
    
    // Modeli eğit
    model.fit(&x_train, &y_train, 10, 32, None)?;
    
    // Tahmin yap
    let x_test = Array2::zeros((10, 784));
    let predictions = model.predict(&x_test)?;
    
    // Modeli kaydet
    model.save("model.json")?;
    
    Ok(())
}
```

### TURK DSL Sözdizimi ile Model Oluşturma

```rust
use turk_lang::{Parser, ModelParser, Result};

fn main() -> Result<()> {
    let kod = r#"
    Model mnist_siniflandirici {
        Katman(784, 128, ReLU),
        Katman(128, 64, ReLU),
        Dropout(0.2),
        Katman(64, 10, Softmax)
    }
    
    Veriler mnist_veri {
        Egitim: "data/mnist_train.csv",
        Test: "data/mnist_test.csv"
    }
    
    Egit mnist_siniflandirici {
        Optimizer: Adam(ogrenme_orani: 0.001),
        KayipFonksiyonu: CrossEntropy,
        EpochSayisi: 10,
        BatchBoyutu: 64
    }
    "#;
    
    let mut parser = ModelParser::new(kod);
    let program = parser.parse()?;
    
    // Program AST'sini kullanarak modeli oluştur ve eğit
    // ...
    
    Ok(())
}
```

## MNIST Örneği

TURK Dili ile MNIST veri setini sınıflandıran bir örnek:

```bash
# Basit model ile MNIST sınıflandırıcı çalıştır
$ cargo run --example mnist_classifier simple

# Orta düzey model ile MNIST sınıflandırıcı çalıştır
$ cargo run --example mnist_classifier medium

# Gelişmiş model ile MNIST sınıflandırıcı çalıştır
$ cargo run --example mnist_classifier advanced
```

## Katmanlar

TURK Dili şu katmanları destekler:

- **Dense**: Tam bağlantılı katman
- **Dropout**: Aşırı öğrenmeyi önlemek için dropout katmanı
- **BatchNorm**: Batch normalizasyon katmanı

## Optimizerlar

Şu optimizasyon algoritmaları mevcuttur:

- **SGD**: Stokastik Gradient İniş
- **Adam**: Adaptive Moment Estimation
- **AdamW**: Ağırlık aşınmalı Adam
- **RMSProp**: Root Mean Square Propagation

## Performans

TURK Dili, ndarray ve rayon kütüphanelerini kullanarak yüksek performanslı hesaplamalar sağlar. Çoklu çekirdek işlemcilerde paralelleştirme ile eğitim süreleri önemli ölçüde azaltılabilir.

Performans değerlendirmeleri için:

```bash
$ cargo bench
```

## Katkıda Bulunma

1. Bu depoyu fork edin
2. Yeni bir özellik dalı oluşturun (`git checkout -b ozellik/yeni-ozellik`)
3. Değişikliklerinizi commit edin (`git commit -am 'Yeni özellik ekle'`)
4. Dalınızı push edin (`git push origin ozellik/yeni-ozellik`)
5. Bir Pull Request oluşturun

## Lisans

Bu proje MIT Lisansı altında lisanslanmıştır - Detaylar için [LICENSE](LICENSE) dosyasına bakınız.

## İletişim

