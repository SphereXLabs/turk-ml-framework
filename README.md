# TURK Dil Kütüphanesi 🇹🇷

Türkçe yapay zeka programlama dili ve makine öğrenmesi kütüphanesi.

[![Lisans: MIT](https://img.shields.io/badge/Lisans-MIT-blue.svg)](LICENSE)
[![Cargo: turk-lang](https://img.shields.io/badge/crates.io-turk--lang-orange)](https://crates.io/crates/turk-lang)
[![Dokümanlar](https://img.shields.io/badge/docs.rs-turk--lang-blue)](https://docs.rs/turk-lang)
[![CI/CD](https://github.com/NeonChipher/turk-ml-framework/workflows/TURK%20Dil%20Kütüphanesi%20CI/CD/badge.svg)](https://github.com/NeonChipher/turk-ml-framework/actions)
[![Python Paketi](https://img.shields.io/badge/PyPI-turk--lang-blue)](https://pypi.org/project/turk-lang/)

## Nedir?

TURK, Türkçe komutlar kullanarak makine öğrenimi modelleri oluşturmanızı, eğitmenizi ve değerlendirmenizi sağlayan bir programlama dili ve kütüphanesidir. Rust programlama dili ile geliştirilmiş olup, yüksek performans ve güvenlik sunar.

## Özellikler

- 🧠 **Türkçe DSL**: Türkçe komutlarla makine öğrenimi modelleri tanımlama
- 🚀 **Yüksek Performans**: Rust ile yazılmış optimizasyonlar ve paralel işleme yeteneği
- 🛡️ **Güvenlik**: Bellek güvenliği ve kapsamlı hata yönetimi
- 📚 **Kapsamlı API**: Çeşitli model mimarileri, katmanlar ve optimizasyon algoritmaları
- 🎨 **Görsel Arayüz**: Modelleri görselleştiren ve test etmeyi kolaylaştıran GUI aracı
- 🔌 **Entegrasyon**: Python ve JavaScript (WASM) desteği 
- 🧩 **Genişletilebilirlik**: Kolay şekilde yeni bileşenler ekleyebilme

## Kurulum

### Rust ile Kullanım

Projenize Cargo.toml dosyanıza ekleyin:

```toml
[dependencies]
turk-lang = "0.1.0"
```

### Python ile Kullanım

```bash
pip install turk-lang
```

### GUI Aracını Kurma

GUI aracını çalıştırmak için:

```bash
# Rust kurulu olmalıdır
git clone https://github.com/NeonChipher/turk-ml-framework.git
cd turk-ml-framework
cargo run --bin turk-gui
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

### TURK Dil Sözdizimi

TURK dilini kullanarak yapay zeka modelini Türkçe sözdizimi ile şu şekilde tanımlayabilirsiniz:

```
Model mnist_model {
    Katman(784, Aktivasyon=Relu)
    Katman(128, Aktivasyon=Relu)
    Katman(64, Aktivasyon=Relu)
    Katman(10, Aktivasyon=Softmax)
    OgrenmeOrani=0,001
    Optimizer="Adam"
    Kayip="categorical_crossentropy"
    Batch=64
    Epoch=10
}

VeriSeti mnist {
    Kaynak="mnist"
    EgitimOran=0.8
    TestOran=0.2
}

Egit(mnist_model, mnist.egitim_x, mnist.egitim_y)
Degerlendir(mnist_model, mnist.test_x, mnist.test_y)

Kaydet(mnist_model, "mnist_model.json")
```

## GUI Arayüzü

TURK dili için geliştirdiğimiz grafik arayüz aracı şunları sağlar:

- 📝 Türkçe makine öğrenimi kodu yazma ve düzenleme
- 🔄 Anında kod ayrıştırma ve hata kontrolü
- 📊 Model mimarilerini görsel olarak inceleme
- 🚀 Modelleri doğrudan arayüzden eğitme ve test etme
- 📋 Örnek kodlara kolay erişim

GUI aracının özellikleri:

- Basit ve kullanıcı dostu arayüz
- Koyu ve açık tema desteği
- Kodun gerçek zamanlı ayrıştırılması
- Ağ mimarisinin ağaç diyagramı olarak görselleştirilmesi
- Katman detaylarının ve model parametrelerinin gösterimi

![TURK GUI](docs/images/turk_gui_screenshot.png)

## Mimari

TURK, modüler bir yapıda tasarlanmıştır:

- **Parser**: Türkçe DSL kodunu işleyen ayrıştırıcı
- **AST**: Ayrıştırılmış kodun soyut sözdizim ağacı
- **Model**: Eğitim ve tahmin işlemleri için temel API
- **Katmanlar**: Dense, Dropout, BatchNorm gibi çeşitli ağ katmanları
- **Optimizerlar**: SGD, Adam, AdamW, RMSProp gibi optimizasyon algoritmaları
- **Metrikler**: Doğruluk, kesinlik, geri çağırma gibi değerlendirme metrikleri

Detaylı mimari açıklaması için [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) dosyasını inceleyebilirsiniz.

## Örnek Uygulamalar

Örnek uygulamalar ve model zincirleri için [examples](examples) dizinini inceleyebilirsiniz:

- [MNIST Sınıflandırıcı](examples/mnist_classifier.rs)
- [Film Yorumları Duygu Analizi](examples/model_zoo/duygu_analizi.turk)
- [CIFAR-10 Görüntü Sınıflandırma](examples/model_zoo/cifar10_siniflandirici.turk)

## Destek ve İletişim

- 🌐 [Resmi Web Sitesi](https://turk-lang.org)
- 📝 [GitHub Issues](https://github.com/NeonChipher/turk-ml-framework/issues)
- 📧 [E-posta İletişim](mailto:TurkDilModeli@gmail.com)

## Yol Haritası

- [x] Temel makine öğrenimi fonksiyonları
- [x] Grafik arayüz geliştirme
- [ ] Derin öğrenme modellerinin genişletilmesi (CNN, RNN)
- [ ] Dil tasarımının iyileştirilmesi
- [ ] Web tabanlı eğitme ve test etme ortamı
- [ ] Doğal dil işleme için özel kütüphaneler
- [ ] Çoklu GPU ve dağıtık eğitim desteği

## Katkıda Bulunma

Katkılarınızı bekliyoruz! Lütfen [CONTRIBUTING.md](CONTRIBUTING.md) dosyasını inceleyerek katkıda bulunma sürecini öğrenin.

## Lisans

Bu proje MIT Lisansı altında dağıtılmaktadır. Detaylar için [LICENSE](LICENSE) dosyasını inceleyebilirsiniz.


""""""""""