# ARCHITECTURE.md Taşındı

Bu dokümantasyon dosyası `docs/ARCHITECTURE.md` konumuna taşınmıştır. Lütfen güncel mimari dökümantasyonuna aşağıdaki linkten erişin:

[TURK Dil Kütüphanesi Mimari Dökümantasyonu](docs/ARCHITECTURE.md)

Bu dosya yalnızca geri uyumluluk için korunmaktadır.

# TURK Dil Kütüphanesi Mimari Dokümantasyonu

Bu doküman, TURK dil kütüphanesinin genel mimari yapısını, bileşenlerini ve veri akışını açıklamaktadır.

## 1. Genel Mimari Bakış

TURK, Türkçe makine öğrenimi komutlarını yorumlayıp çalıştırabilen özel bir dil kütüphanesidir. Bu kütüphane, aşağıdaki ana bileşenlerden oluşmaktadır:

```
┌─────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│ TURK Kodu   │────▶│  Parser    │────▶│    AST     │────▶│  Model     │
└─────────────┘     └────────────┘     └────────────┘     └────────────┘
                                                                │
                           ┌────────────────────────────────────┘
                           ▼
┌─────────────┐     ┌────────────┐     ┌────────────┐     ┌────────────┐
│   Output    │◀────│ Execution  │◀────│ Optimizer  │◀────│   Layers   │
└─────────────┘     └────────────┘     └────────────┘     └────────────┘
```

### 1.1 Ana Bileşenler

1. **Parser**: Türkçe DSL (Domain Specific Language) kodunu ayrıştırır ve sözdizimi analizini gerçekleştirir
2. **AST (Abstract Syntax Tree)**: Ayrıştırılmış kodun ağaç yapısındaki gösterimidir
3. **Model**: AST'den bir makine öğrenimi modeli yapısı oluşturur
4. **Layers**: Modelin katmanlarını (Dense, Dropout, vb.) tanımlar
5. **Optimizer**: Model eğitimi için optimizasyon algoritmalarını uygular
6. **Execution**: Modelin eğitimi ve tahmin işlemlerini yürütür

## 2. Veri Akışı

### 2.1 DSL'den AST'ye

1. Kullanıcı, Türkçe DSL kullanarak model tanımını girer
2. Lexer, girdideki token'ları ayrıştırır
3. Parser, token'ları kullanarak AST oluşturur
4. AST doğrulaması yapılır

### 2.2 AST'den Modele

1. AST ayrıştırıcısı, AST'yi bir model yapısına dönüştürür
2. Katmanlar, optimizerlar ve hiperparametreler yapılandırılır
3. Model, eğitim için hazırlanır

### 2.3 Model Eğitimi ve Tahmin

1. Veri ön işleme adımları uygulanır
2. Model eğitimi başlatılır
3. Optimizer, ağırlıkları günceller
4. Metrikler hesaplanır
5. Model, tahmin için kullanılır

## 3. Kod Organizasyonu

### 3.1 Ana Modüller

| Dosya              | Açıklama                                             |
|--------------------|----------------------------------------------------|
| `lib.rs`           | Kütüphane giriş noktası ve modül dışa aktarımları     |
| `ast.rs`           | Soyut Sözdizimi Ağacı yapıları                     |
| `token.rs`         | Token tanımları ve token işlemleri                  |
| `parser.rs`        | TURK dilini ayrıştırma işlemleri                    |
| `model.rs`         | Model oluşturma ve yönetme                          |
| `layer.rs`         | Katman tanımları ve işlemleri                       |
| `loss.rs`          | Kayıp fonksiyonları                                 |
| `optimizer.rs`     | Optimizasyon algoritmaları                          |
| `metrics.rs`       | Değerlendirme metrikleri                            |
| `error.rs`         | Hata tipleri ve hata yönetimi                       |
| `initializers.rs`  | Ağırlık başlatıcıları                               |
| `preprocessing.rs` | Veri ön işleme araçları                             |
| `utils.rs`         | Yardımcı fonksiyonlar                               |

### 3.2 Alt Modüller

| Klasör             | Açıklama                                             |
|--------------------|----------------------------------------------------|
| `parser/`          | Parser bileşenleri                                  |
| `parser/mod.rs`    | Parser API tanımları                                |
| `parser/model_parser.rs` | Model tanımları için özel parser              |
| `parser/expr_parser.rs`  | İfade ayrıştırma                              |

## 4. Önemli Veri Yapıları

### 4.1 AST Yapıları

```rust
pub struct ModelDef {
    pub name: String,
    pub layers: Vec<LayerDef>,
    pub optimizer: OptimizerDef,
    pub loss: LossDef,
}

pub enum LayerDef {
    Dense { input: usize, output: usize, activation: Activation },
    Dropout { rate: f32 },
    BatchNorm { features: usize },
}

pub enum OptimizerDef {
    SGD { learning_rate: f32, momentum: Option<f32> },
    Adam { learning_rate: f32 },
    AdamW { learning_rate: f32, weight_decay: f32 },
    RMSProp { learning_rate: f32 },
}
```

### 4.2 Model Yapıları

```rust
pub struct Sequential {
    layers: Vec<Box<dyn Layer>>,
    optimizer: Option<Box<dyn Optimizer>>,
    loss: Option<Box<dyn Loss>>,
}

pub trait Layer {
    fn forward(&self, input: &Array2<f32>) -> Result<Array2<f32>>;
    fn backward(&self, grad: &Array2<f32>) -> Result<Array2<f32>>;
    fn update(&mut self, optimizer: &dyn Optimizer) -> Result<()>;
}
```

### 4.3 Optimizer Yapıları

```rust
pub trait Optimizer {
    fn update(&self, param: &mut Array2<f32>, grad: &Array2<f32>) -> Result<()>;
    fn update_with_state(&mut self, param: &mut Array2<f32>, grad: &Array2<f32>) -> Result<()>;
}

pub struct SGD {
    learning_rate: f32,
    momentum: Option<f32>,
    velocity: HashMap<usize, Array2<f32>>,
}
```

## 5. Feature Bayrakları

TURK kütüphanesi, farklı özellikler için feature bayrakları kullanmaktadır:

- `python`: Python bağlantıları için destek
- `wasm`: WebAssembly desteği
- `simd`: SIMD vektör optimizasyonları
- `blas`: BLAS/LAPACK entegrasyonu

## 6. Genişletme Noktaları

TURK kütüphanesi, aşağıdaki alanlarda genişletilebilir:

1. **Yeni Katmanlar**: `Layer` trait'ini uygulayarak yeni katman tipleri eklenebilir
2. **Yeni Optimizerlar**: `Optimizer` trait'ini uygulayarak yeni optimizasyon algoritmaları eklenebilir
3. **Yeni Kayıp Fonksiyonları**: `Loss` trait'ini uygulayarak yeni kayıp fonksiyonları eklenebilir
4. **Yeni Metrikler**: `Metric` trait'ini uygulayarak yeni değerlendirme metrikleri eklenebilir
5. **Yeni Veri Ön İşleme Adımları**: `Preprocessor` trait'ini uygulayarak yeni veri hazırlama adımları eklenebilir

## 7. Performans Optimizasyonları

TURK, aşağıdaki mekanizmalar aracılığıyla yüksek performans sağlar:

1. **Paralel İşlem**: Rayon kütüphanesi kullanılarak çok çekirdekli işlem desteği
2. **SIMD Vektörizasyonu**: CPU vektör talimatlarının kullanımı
3. **BLAS Entegrasyonu**: Optimizasyon edilmiş lineer cebir kütüphaneleri
4. **Zero-Copy Parsing**: Veri kopyalamayı minimize eden ayrıştırma mekanizması
5. **Heap Alokasyon Optimizasyonu**: Gereksiz heap alokasyonlarından kaçınma

## 8. Güvenlik Değerlendirmesi

TURK kütüphanesi, aşağıdaki güvenlik önlemlerini içerir:

1. **Giriş Doğrulama**: Tüm kullanıcı girdileri doğrulanır
2. **Bellek Güvenliği**: `unsafe` kod bloklarının kullanımı minimum düzeydedir
3. **Serileştirme Güvenliği**: Serde şema doğrulaması
4. **Hata Yönetimi**: Panik yerine `Result` tipi kullanımı
5. **Kaynak Sınırlaması**: Dosya boyutları ve işlem sınırlamaları

## 9. Diyagram: Eğitim İş Akışı

```
┌──────────────┐
│ Veri Yükleme │
└──────┬───────┘
       │
       ▼
┌──────────────┐    ┌──────────────┐
│ Ön İşleme    │───▶│ Veri Ayrımı  │
└──────┬───────┘    └──────┬───────┘
       │                   │
       ▼                   ▼
┌──────────────┐    ┌──────────────┐
│ Model Kurma  │    │ Doğrulama    │
└──────┬───────┘    │ Verisi       │
       │            └──────┬───────┘
       ▼                   │
┌──────────────┐           │
│ Forward Pass │◀──────────┘
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Kayıp Hesabı │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Backward Pass│
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Ağırlık      │
│ Güncelleme   │
└──────┬───────┘
       │
       ▼
┌──────────────┐
│ Ölçüm &      │
│ Kaydetme     │
└──────────────┘
```

## 10. Örnek TURK DSL Yapısı

Aşağıda, TURK DSL'nin temel sözdizimi örneği verilmiştir:

```
model SiniflandiriciOlustur {
    katman Dense(girdi=784, cikti=128, aktivasyon=relu)
    katman Dropout(oran=0.2)
    katman Dense(girdi=128, cikti=10, aktivasyon=softmax)
    
    optimizer Adam(ogrenme_orani=0.001)
    kayip KategorikCaprazEntropi()
}

egit SiniflandiriciOlustur {
    veri = MnistVeriYukle()
    epok = 10
    yigin_boyutu = 32
}

tahminEt SiniflandiriciOlustur {
    giris = TestVeriYukle()
    cikti = "tahminler.csv"
}
``` 