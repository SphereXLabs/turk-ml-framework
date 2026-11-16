# TURK Model Kütüphanesi (Model Zoo)

Bu dizin, TURK dil kütüphanesi için hazırlanmış örnek modelleri içermektedir. Bu modeller, farklı problemler için referans olarak kullanılabilir veya kendi projelerinize uyarlanabilir.

## Modeller

### Görüntü Sınıflandırma

- [mnist_siniflandirici.turk](mnist_siniflandirici.turk) - MNIST rakam sınıflandırma için basit bir sinir ağı
- [fashion_mnist_siniflandirici.turk](fashion_mnist_siniflandirici.turk) - Moda-MNIST veri seti için sınıflandırıcı
- [cifar10_siniflandirici.turk](cifar10_siniflandirici.turk) - CIFAR-10 görüntüleri için CNN sınıflandırıcı
- [mobilenet_transfer.turk](mobilenet_transfer.turk) - Transfer learning ile MobileNet tabanlı görüntü sınıflandırıcı

### Doğal Dil İşleme

- [duygu_analizi.turk](duygu_analizi.turk) - Türkçe film yorumları üzerinde duygu analizi
- [kelime_vektoru.turk](kelime_vektoru.turk) - Türkçe kelime vektörü eğitimi
- [transformer_siniflandirici.turk](transformer_siniflandirici.turk) - Basit Transformer tabanlı metin sınıflandırıcı

### Regresyon

- [ev_fiyat_tahmini.turk](ev_fiyat_tahmini.turk) - Ev fiyatları tahmin modeli
- [zaman_serisi.turk](zaman_serisi.turk) - Zaman serisi tahmini için LSTM modeli

## Kullanım

Model dosyalarını doğrudan TURK DSL yorumlayıcısı ile çalıştırabilirsiniz:

```bash
cargo run --bin turk-cli örnek/yol/model_adi.turk
```

Alternatif olarak, GUI uygulamasını kullanabilirsiniz:

```bash
cargo run --bin turk-gui
```

Modelleri programınızda şu şekilde yükleyebilirsiniz:

```rust
use turk_lang::Sequential;

fn main() -> turk_lang::Result<()> {
    let mut model = Sequential::load("örnek/yol/model_adi.turk")?;
    
    // Modeli kullan
    let tahminler = model.predict(&giris_verisi)?;
    
    Ok(())
}
```

## Python ile Kullanım

TURK modellerini Python'da da kullanabilirsiniz:

```python
import turk_lang as tl

# Model yükleme
model = tl.Sequential.load("örnek/yol/model_adi.turk")

# Tahmin yapma
predictions = model.predict(input_data)

# Veya yeni model oluşturma
model = tl.Sequential()
model.add(tl.Dense(784, 128, tl.Activation.relu()))
model.add(tl.Dense(128, 10, tl.Activation.softmax()))
model.compile(tl.Adam(0.001))

# Eğitim
model.fit(x_train, y_train, epochs=10, batch_size=32)
```

## ONNX Dönüşümü

TURK modellerini ONNX formatına dönüştürebilirsiniz, böylece diğer framework'lerde (TensorFlow, PyTorch, ONNX Runtime vb.) kullanabilirsiniz:

```rust
use turk_lang::{Sequential, export_onnx};

fn main() -> turk_lang::Result<()> {
    let model = Sequential::load("örnek/yol/model_adi.turk")?;
    
    // Modeli ONNX'e dönüştür
    export_onnx(&model, "model.onnx", &[1, 784])?;
    
    Ok(())
}
```

Python ile ONNX dönüşümü:

```python
import turk_lang as tl

# Model yükleme
model = tl.Sequential.load("örnek/yol/model_adi.turk")

# ONNX'e dönüştürme
model.export_onnx("model.onnx", input_shape=(1, 784))
```

## Model Yapıları

Her model dosyası için, ilgili yapı ve parametreler aşağıdaki gibi belirtilmektedir:

### MNIST Sınıflandırıcı

```
Model MnistSiniflandirici {
    Katman(784, Aktivasyon=Relu)
    Katman(128, Aktivasyon=Relu)
    Katman(10, Aktivasyon=Softmax)
    
    Optimizer="Adam"
    Kayip="cross_entropy"
    Batch=64
    Epoch=10
}
```

### Duygu Analizi

```
Model DuyguAnalizi {
    Katman(100, Aktivasyon=Relu)
    Katman(64, Aktivasyon=Relu)
    Katman(1, Aktivasyon=Sigmoid)
    
    Optimizer=Adam(ogrenme_orani=0.001, beta1=0.9)
    Kayip="binary_cross_entropy"
    Batch=32
    Epoch=5
}
```

### MobileNet Transfer Learning

```
Model MobileNetTransfer {
    Kaynagı="pretrained_mobilenet"
    DondurKatman=15
    
    Katman(1024, Aktivasyon=Relu)
    Katman(10, Aktivasyon=Softmax)
    
    Optimizer=AdamW(ogrenme_orani=0.0001, weight_decay=0.01)
    Kayip="categorical_crossentropy"
    Batch=16
    Epoch=3
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
- Eğer mümkünse, modelin ONNX dönüşümünü test edin ve doğrulayın 