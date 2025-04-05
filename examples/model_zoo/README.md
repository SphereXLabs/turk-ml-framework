# TURK Model Kütüphanesi (Model Zoo)

Bu dizin, TURK dil kütüphanesi için hazırlanmış örnek modelleri içermektedir. Bu modeller, farklı problemler için referans olarak kullanılabilir veya kendi projelerinize uyarlanabilir.

## Modeller

### Görüntü Sınıflandırma

- [mnist_siniflandirici.turk](mnist_siniflandirici.turk) - MNIST rakam sınıflandırma için basit bir sinir ağı
- [fashion_mnist_siniflandirici.turk](fashion_mnist_siniflandirici.turk) - Moda-MNIST veri seti için sınıflandırıcı
- [cifar10_siniflandirici.turk](cifar10_siniflandirici.turk) - CIFAR-10 görüntüleri için CNN sınıflandırıcı

### Doğal Dil İşleme

- [duygu_analizi.turk](duygu_analizi.turk) - Türkçe film yorumları üzerinde duygu analizi
- [kelime_vektoru.turk](kelime_vektoru.turk) - Türkçe kelime vektörü eğitimi

### Regresyon

- [ev_fiyat_tahmini.turk](ev_fiyat_tahmini.turk) - Ev fiyatları tahmin modeli
- [zaman_serisi.turk](zaman_serisi.turk) - Zaman serisi tahmini için LSTM modeli

## Kullanım

Model dosyalarını doğrudan TURK DSL yorumlayıcısı ile çalıştırabilirsiniz:

```bash
cargo run --bin turk-interpreter örnek/yol/model_adi.turk
```

Alternatif olarak, modelleri programınızda şu şekilde yükleyebilirsiniz:

```rust
use turk_lang::Sequential;

fn main() -> turk_lang::Result<()> {
    let mut model = Sequential::load("örnek/yol/model_adi.turk")?;
    
    // Modeli kullan
    let tahminler = model.predict(&giris_verisi)?;
    
    Ok(())
}
```

## Model Yapıları

Her model dosyası için, ilgili yapı ve parametreler aşağıdaki gibi belirtilmektedir:

### MNIST Sınıflandırıcı

```
model MnistSiniflandirici {
    katman Dense(girdi=784, cikti=128, aktivasyon=relu)
    katman Dropout(oran=0.2)
    katman Dense(girdi=128, cikti=10, aktivasyon=softmax)
    
    optimizer Adam(ogrenme_orani=0.001)
    kayip KategorikCaprazEntropi()
}
```

### Duygu Analizi

```
model DuyguAnalizi {
    katman LSTM(girdi=100, cikti=64, aktivasyon=tanh)
    katman Dense(girdi=64, cikti=1, aktivasyon=sigmoid)
    
    optimizer RMSProp(ogrenme_orani=0.001)
    kayip IkiliCaprazEntropi()
}
```

## Katkıda Bulunma

Model kütüphanesine katkıda bulunmak için:

1. Yeni model dosyanızı oluşturun
2. Modeli test edin ve performans metriklerini belgelendirin
3. Bu README dosyasını güncelleyin ve modelinizi ekleyin
4. Pull request açın

Yeni modellere katkıda bulunurken lütfen şu kurallara uyun:

- Model dosya adı açıklayıcı olmalı
- Dosya içinde yorum satırları ile modelin amacını ve kullanım bilgilerini ekleyin
- Modelin gerektirdiği veri ön işleme adımlarını belirtin
- Varsa test sonuçlarını ve performans metriklerini ekleyin 