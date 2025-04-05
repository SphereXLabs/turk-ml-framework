# TURK Dil Kütüphanesi 🇹🇷

Türkçe yapay zeka programlama dili ve makine öğrenmesi kütüphanesi.

[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-blue.svg)](LICENSE)
[![Cargo: turk-lang](https://img.shields.io/badge/crates.io-turk--lang-orange)](https://crates.io/crates/turk-lang)
[![Dokümanlar](https://img.shields.io/badge/docs.rs-turk--lang-blue)](https://docs.rs/turk-lang)
[![CI/CD](https://github.com/snnkr/turk-lang/workflows/TURK%20Dil%20Kütüphanesi%20CI/CD/badge.svg)](https://github.com/snnkr/turk-lang/actions)
[![Python Paketi](https://img.shields.io/badge/PyPI-turk--lang-blue)](https://pypi.org/project/turk-lang/)

## Nedir?

TURK, Türkçe komutlar kullanarak makine öğrenimi modelleri oluşturmanızı, eğitmenizi ve değerlendirmenizi sağlayan bir programlama dili ve kütüphanesidir. Rust programlama dili ile geliştirilmiş olup, yüksek performans ve güvenlik sunar.

## Özellikler

- 🧠 **Türkçe DSL**: Türkçe komutlarla makine öğrenimi modelleri tanımlama
- 🚀 **Yüksek Performans**: Rust ile yazılmış optimizasyonlar ve paralel işleme yeteneği
- 🛡️ **Güvenlik**: Bellek güvenliği ve kapsamlı hata yönetimi
- 📚 **Kapsamlı API**: Çeşitli model mimarileri, katmanlar ve optimizasyon algoritmaları
- 🔌 **Entegrasyon**: Python ve JavaScript (WASM) desteği 
- 🧩 **Genişletilebilirlik**: Kolay şekilde yeni bileşenler ekleyebilme

## Kurulum

### Rust ile Kullanım

Projenize Cargo.toml dosyanıza ekleyin:

```toml
[dependencies]
turk-lang = "0.1.0"
logos = "0.13.0"
```

### Python ile Kullanım

```bash
pip install turk-lang
```

### WASM ile Web Uygulamalarında Kullanım

```bash
npm install turk-lang
```

## Hızlı Başlangıç

### Rust ile Basit Model Örneği

```rust
use turk_lang::{Sequential, Dense, Activation, Adam};

fn main() -> turk_lang::Result<()> {
    // Veriyi yükle ve hazırla
    let (x_train, y_train) = veri_yukle()?;
    
    // Model oluştur
    let mut model = Sequential::new();
    model.add(Dense::new(784, 128, Activation::Relu))
         .add(Dense::new(128, 10, Activation::Softmax))
         .compile(Adam::new(0.001));
    
    // Eğit
    model.fit(&x_train, &y_train, 10, 32, None)?;
    
    // Kaydet
    model.save("model.json")?;
    
    Ok(())
}
```

### Python ile Kullanım

```python
import numpy as np
from turk_lang import Sequential, Dense, Activation, Adam

# Veriyi yükle
x_train = np.random.rand(1000, 10)
y_train = np.random.rand(1000, 1)

# Model oluştur
model = Sequential()
model.add(Dense(10, 5, Activation.relu()))
model.add(Dense(5, 1, Activation.sigmoid()))
model.compile(Adam(0.001))

# Eğit ve değerlendir
history = model.fit(x_train, y_train, epochs=10, batch_size=32)
print(f"Son kayıp: {history[-1]['loss']}")
```

### TURK Dil Sözdizimi

TURK dilini kullanarak yapay zeka modelini Türkçe sözdizimi ile şu şekilde tanımlayabilirsiniz:

```
model MnistSiniflandirici {
    katman Dense(girdi=784, cikti=128, aktivasyon=relu)
    katman Dropout(oran=0.2)
    katman Dense(girdi=128, cikti=10, aktivasyon=softmax)
    
    optimizer Adam(ogrenme_orani=0.001)
    kayip KategorikCaprazEntropi()
}

egit MnistSiniflandirici {
    veri = "mnist.csv"
    epok = 10
    yigin_boyutu = 32
}
```

## Mimari

TURK, modüler bir yapıda tasarlanmıştır:

- **Parser**: Türkçe DSL kodunu işleyen ayrıştırıcı
- **AST**: Ayrıştırılmış kodun soyut sözdizim ağacı
- **Model**: Eğitim ve tahmin işlemleri için temel API
- **Katmanlar**: Dense, Dropout, BatchNorm gibi çeşitli ağ katmanları
- **Optimizerlar**: SGD, Adam, AdamW, RMSProp gibi optimizasyon algoritmaları
- **Metrikler**: Doğruluk, kesinlik, geri çağırma gibi değerlendirme metrikleri

Detaylı mimari açıklaması için [ARCHITECTURE.md](ARCHITECTURE.md) dosyasını inceleyebilirsiniz.

## Örnek Uygulamalar

Örnek uygulamalar ve model zincirleri için [examples](examples) dizinini inceleyebilirsiniz:

- [MNIST Sınıflandırıcı](examples/mnist_classifier.rs)
- [Film Yorumları Duygu Analizi](examples/model_zoo/duygu_analizi.turk)
- [CIFAR-10 Görüntü Sınıflandırma](examples/model_zoo/cifar10_siniflandirici.turk)

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını inceleyerek katkıda bulunma sürecini öğrenin.

## Lisans

Bu proje MIT Lisansı altında dağıtılmaktadır. Detaylar için [LICENSE](LICENSE) dosyasını inceleyebilirsiniz.


