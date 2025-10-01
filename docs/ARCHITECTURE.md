# TURK Dil Kütüphanesi Mimari Dökümantasyonu

Bu belge, TURK dil kütüphanesinin mimari yapısını, bileşenlerini ve veri akışını açıklamaktadır.

## Genel Mimari

TURK, aşağıdaki temel bileşenlerden oluşan katmanlı bir mimariye sahiptir:

```
┌─────────────────────────────────────────────────────────────┐
│                        Kullanıcı Arayüzü                     │
│            (TURK GUI, CLI, Python/JS Bağlayıcıları)          │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                    TURK Programlama Dili                     │
│                                                              │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────┐  │
│  │  Lexer      │─▶│  Parser      │─▶│  AST Değerlendirici │  │
│  │  (Token)    │  │  (Sözdizimi) │  │  (Semantik Analiz)  │  │
│  └─────────────┘  └──────────────┘  └────────────────────┘  │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     TURK Makine Öğrenmesi                    │
│                                                              │
│  ┌─────────────┐  ┌────────────┐  ┌─────────────┐           │
│  │  Model      │  │  Katmanlar  │  │  Optimizer  │           │
│  └─────────────┘  └────────────┘  └─────────────┘           │
│                                                              │
│  ┌─────────────┐  ┌────────────┐  ┌─────────────┐           │
│  │  Kayıp      │  │  Metrikler  │  │  Veri İşleme │           │
│  │  Fonksiyonu │  │             │  │             │           │
│  └─────────────┘  └────────────┘  └─────────────┘           │
└───────────────────────────┬─────────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────────┐
│                     Temel Kütüphaneler                       │
│          (ndarray, Rayon, Serde, Logos, ndarray-stats)       │
└─────────────────────────────────────────────────────────────┘
```

## Ana Bileşenler

### 1. Sözdizimi ve Ayrıştırma

- **Token** (`src/token.rs`): Dil için temel sözdizimsel birimleri tanımlar.
- **Parser** (`src/parser/mod.rs`, `src/parser/model_parser.rs`, vb.): TURK kodunu anlamsal yapıya dönüştürür.
- **AST** (`src/ast.rs`): Ayrıştırılmış kodun soyut sözdizim ağacını temsil eder.

### 2. Makine Öğrenimi Bileşenleri

- **Model** (`src/model.rs`): Sinir ağı modellerini tanımlar ve yönetir.
- **Katmanlar** (`src/layer.rs`): Farklı sinir ağı katmanlarını (Dense, Dropout, vb.) içerir.
- **Optimizer** (`src/optimizer.rs`): Çeşitli optimizasyon algoritmalarını (SGD, Adam, AdamW, RMSProp) uygular.
- **Kayıp Fonksiyonları** (`src/loss.rs`): Model eğitimi için çeşitli kayıp fonksiyonları sağlar.
- **Metrikler** (`src/metrics.rs`): Model performansını değerlendirmek için ölçüm araçları sunar.
- **Veri İşleme** (`src/preprocessing.rs`): Veri normalizasyonu, özellik çıkarma ve veri zenginleştirme araçları.
- **Başlatıcılar** (`src/initializers.rs`): Ağırlık ve bias değerleri için başlangıç değerlerini oluşturur.

### 3. Yardımcı Bileşenler

- **Hata Yönetimi** (`src/error.rs`): Merkezi hata türleri ve hata işleme.
- **Yardımcı İşlevler** (`src/utils.rs`): Genel yardımcı fonksiyonlar ve araçlar.

### 4. Arayüz Bileşenleri

- **GUI** (`src/bin/turk_gui.rs`): Grafiksel kullanıcı arayüzü.
- **CLI** (`src/main.rs`): Komut satırı arayüzü.

## Veri Akışı

1. **Giriş**: Kullanıcı, TURK sözdizimi ile bir program yazar.
2. **Leksikal Analiz**: Program `Token` yapılarına bölünür.
3. **Sözdizimsel Analiz**: Token dizisi, AST'ye dönüştürülür.
4. **Anlamsal Analiz**: AST, tutarlılık için kontrol edilir ve değerlendirilir.
5. **Model Oluşturma**: AST'den bir sinir ağı modeli oluşturulur.
6. **Eğitim/Değerlendirme**: Model, kullanıcı tarafından belirtilen verilerle eğitilir ve değerlendirilir.
7. **Çıktı**: Sonuçlar kullanıcıya sunulur.

## Eklenti Mimarisi

TURK, genişletilebilir bir mimari üzerine kurulmuştur:

- Yeni katmanlar `Layer` trait'ini uygulayarak eklenebilir.
- Yeni optimizerlar `Optimizer` trait'ini uygulayarak eklenebilir.
- Yeni kayıp fonksiyonları `Loss` trait'ini uygulayarak eklenebilir.
- Yeni metrikler `Metric` trait'ini uygulayarak eklenebilir.

## Modüller Arası Etkileşim

```
          ┌──────────────────────┐
          │        Parser        │
          └──────────┬───────────┘
                     │ üretir
                     ▼
          ┌──────────────────────┐       kullanır     ┌──────────────────┐
          │        AST           │◄──────────────────▶│     Model        │
          └──────────┬───────────┘                    └─────────┬────────┘
                     │ yorumlanır                              │
                     ▼                                         │ içerir
          ┌──────────────────────┐                    ┌────────▼─────────┐
          │  Runtime Değerler    │                    │     Katmanlar    │
          └──────────┬───────────┘                    └─────────┬────────┘
                     │ kullanır                                 │
                     │                                          │ kullanır
                     │                                          ▼
                     │                                ┌──────────────────┐
                     └───────────────────────────────►    Optimizer     │
                                                     └──────────────────┘
```

## Performans Optimizasyonları

TURK, çeşitli performans optimizasyonları içerir:

1. **Paralel İşleme**: Rayon kütüphanesi ile çoklu iş parçacığı desteği.
2. **SIMD Vektörizasyonu**: ndarray aracılığıyla hızlı matris işlemleri.
3. **Önbelleğe Alma**: Hesaplamaları tekrarlamaktan kaçınmak için.
4. **Hafıza Yönetimi**: Verimli bellek kullanımı için yapılar.

## Güvenlik Özellikleri

1. **Tür Güvenliği**: Rust'un tür sistemi sayesinde derleme zamanında hata kontrolü.
2. **Bellek Güvenliği**: Rust'un ownership modeli ile bellek sızıntıları ve dangling pointer'lar önlenir.
3. **Hata Yönetimi**: Kapsamlı hata işleme ve raporlama.
4. **Input Validation**: Kullanıcı girdilerinin güvenli şekilde doğrulanması.

## Genişletme Kılavuzu

TURK'u genişletmek için şu adımları izleyin:

1. İlgili trait'i uygulayan yeni bir tür oluşturun.
2. Gerekli metodları uygulayın.
3. Birim testleri ekleyin.
4. Dokümantasyon ekleyin.
5. Yeni bileşeni ilgili registry'e kaydedin.

Örnek: Yeni bir optimizer ekleme:

```rust
use ndarray::Array2;
use turk::{Optimizer, Result};

#[derive(Clone, Debug)]
pub struct MyOptimizer {
    learning_rate: f32,
}

impl MyOptimizer {
    pub fn new(learning_rate: f32) -> Self {
        Self { learning_rate }
    }
}

impl Optimizer for MyOptimizer {
    fn step(&mut self, params: &mut Array2<f32>, grads: &Array2<f32>) -> Result<()> {
        // Optimizasyon algoritmasını uygula
        *params -= &(grads * self.learning_rate);
        Ok(())
    }

    fn learning_rate(&self) -> f32 {
        self.learning_rate
    }

    fn optimizer_type(&self) -> &'static str {
        "my_optimizer"
    }

    fn clone_box(&self) -> Box<dyn Optimizer> {
        Box::new(self.clone())
    }

    fn reset_state_for(&mut self, _param_name: &str) {}

    fn reset_state(&mut self) {}
}
```

## Gelecek İyileştirmeler

- GPU hızlandırma desteği (CUDA, OpenCL)
- Dağıtık eğitim desteği
- Daha fazla derin öğrenme mimarisi (CNN, RNN, Transformer)
- Web arayüzü ve bulut entegrasyonu
- AutoML desteği 