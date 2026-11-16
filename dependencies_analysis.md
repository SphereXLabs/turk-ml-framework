# TURK Projesi Bağımlılık Analizi

## Mevcut Durum

TURK projesinde kullanılan bağımlılıkları aşağıdaki kategorilere ayırabiliriz:

### Temel Veri İşleme ve Hesaplama Kütüphaneleri
- `ndarray` (0.15): Çok boyutlu diziler ve matris işlemleri (serde ve rayon özellikleri ile)
- `ndarray-rand` (0.14): Rastgele ndarray nesneleri oluşturmak için
- `ndarray-stats` (0.5): İstatistiksel işlemler için

### Rastgele Sayı Üretimi
- `rand` (0.8): Temel rastgele sayı üreteci
- `rand_distr` (0.4): Farklı dağılımlar için

### Serileştirme ve Veri İşleme
- `serde` (1.0): Serileştirme ve deserileştirme işlemleri (derive özelliği ile)
- `serde_json` (1.0): JSON formatında serileştirme
- `csv` (1.1): CSV dosyası işleme

### Loglama ve Hata Yönetimi
- `env_logger` (0.9): Çevresel değişkenlerden loglama ayarları
- `log` (0.4): Loglama altyapısı
- `thiserror` (1.0): Özel hata türleri tanımlama
- `anyhow` (1.0): Hata işleme için kullanışlı bir kütüphane

### Paralel İşleme
- `rayon` (1.5): Paralel işleme altyapısı

### Metin İşleme ve Ayrıştırma
- `regex` (1.5): Düzenli ifadeler
- `lazy_static` (1.4): Statik değerlerin tembel değerlendirilmesi
- `logos` (0.12): Sözcük ayrıştırma

### Veri Setleri
- `mnist` (0.5): MNIST veri setine erişim

### Görselleştirme ve Kullanıcı Arayüzü
- `plotters` (0.3): Veri görselleştirme için
- `indicatif` (0.16): İlerleme çubukları ve göstergeleri
- `colored` (2.0): Renkli terminal çıktısı
- `chrono` (0.4): Zaman ve tarih işlemleri
- `itertools` (0.14.0): İteratörler için ek işlevsellik

### GUI (İsteğe Bağlı)
- `eframe` (0.22.0): GUI çerçevesi (isteğe bağlı)
- `poll-promise` (0.3.0): Asenkron işlemler (isteğe bağlı)
- `rfd` (0.11.0): Dosya diyalogları (isteğe bağlı)

### Geliştirme Bağımlılıkları
- `criterion` (0.5): Performans testi
- `pprof` (0.12): Profilleme (flamegraph özelliği ile)
- `dhat` (0.3): Heap profilleme
- `tempfile` (3.2): Geçici dosya işlemleri
- `pretty_assertions` (0.7): Daha okunabilir test iddiaları

## İyileştirme Önerileri

### 1. Gereksiz Bağımlılıkları Temizleme

Aşağıdaki bağımlılıklar gözden geçirilmelidir:

- `env_logger` (0.9): Daha yeni sürümü (0.10+) mevcut
- `indicatif` (0.16): Daha yeni sürümü (0.17+) mevcut
- `lazy_static` (1.4): Yeni Rust sürümlerinde `once_cell` veya `std::lazy::OnceCell` kullanılabilir
- `mnist` (0.5): Sadece örnek ve testlerde kullanılıyorsa, `dev-dependencies` altına taşınabilir

### 2. Feature Flag Optimizasyonu

Mevcut feature flag sistemi iyileştirilebilir:

```toml
[features]
default = ["ndarray", "log"]
full = ["default", "plotters", "indicatif", "colored", "csv"]
cli = ["full", "clap"]
gui = ["full", "eframe", "poll-promise", "rfd"]
```

Bu şekilde:
- `default`: Temel işlevsellik
- `full`: Tüm standart özellikler
- `cli`: Komut satırı arabirimi
- `gui`: Grafik kullanıcı arabirimi

### 3. Sürüm Kısıtlamaları

Bağımlılıklarda daha spesifik sürüm kısıtlamaları belirlenmelidir:

```toml
ndarray = { version = "0.15.4", features = ["serde", "rayon"] }
serde = { version = "1.0.152", features = ["derive"] }
```

Bu şekilde, API uyumsuzluğu riski azaltılır.

### 4. Optimize Edilmiş Cargo.toml Örneği

```toml
[package]
name = "turk"
version = "0.1.0"
edition = "2021"
authors = ["SphereX <turkdilmodeli@gmail.com>"]
description = "Türkçe yapay sinir ağı kütüphanesi"
license = "MIT"
repository = "https://github.com/snnkr/turk"
default-run = "turk"
readme = "README.md"
categories = ["science", "machine-learning", "neural-networks"]
keywords = ["turk", "turkish", "neural-network", "deep-learning", "ai"]

[lib]
name = "turk"
path = "src/lib.rs"

[[bin]]
name = "turk"
path = "src/main.rs"
required-features = ["cli"]

[[bin]]
name = "turk_gui"
path = "src/bin/turk_gui.rs"
required-features = ["gui"]

[features]
default = ["ndarray", "log"]
full = ["default", "plotters", "indicatif", "colored", "csv"]
cli = ["full", "clap"]
gui = ["full", "eframe", "poll-promise", "rfd"]

[dependencies]
# Temel hesaplama kütüphaneleri
ndarray = { version = "0.15.4", features = ["serde", "rayon"] }
ndarray-rand = "0.14.0"
ndarray-stats = "0.5.1"

# Rastgele sayı üretimi
rand = "0.8.5"
rand_distr = "0.4.3"

# Serileştirme
serde = { version = "1.0.152", features = ["derive"] }
serde_json = "1.0.93"

# Loglama ve hata yönetimi
log = "0.4.17"
thiserror = "1.0.38"
anyhow = "1.0.69"

# Paralel işleme
rayon = "1.7.0"

# Metin işleme
regex = "1.7.1"
logos = "0.12.1"

# Seçimlik bağımlılıklar (feature flags ile kontrol edilir)
plotters = { version = "0.3.4", optional = true }
indicatif = { version = "0.17.3", optional = true }
colored = { version = "2.0.0", optional = true }
chrono = { version = "0.4.23", optional = true }
itertools = { version = "0.14.0", optional = true }
csv = { version = "1.2.1", optional = true }

# CLI için gerekli
clap = { version = "4.1.8", features = ["derive"], optional = true }

# GUI için gerekli
eframe = { version = "0.22.0", optional = true }
poll-promise = { version = "0.3.0", optional = true }
rfd = { version = "0.11.0", optional = true }

[dev-dependencies]
criterion = "0.5.1"
pprof = { version = "0.12.1", features = ["flamegraph", "criterion"] }
dhat = "0.3.2"
tempfile = "3.5.0"
pretty_assertions = "1.3.0"
mnist = "0.5.0"  # Sadece test ve örneklerde kullanılıyor

[profile.release]
opt-level = 3
lto = true
codegen-units = 1

[profile.dev]
opt-level = 0
debug = true

[profile.bench]
debug = true
```

## Güvenlik İncelemeleri

Güvenlik açısından aşağıdaki adımlar atılmalıdır:

1. Cargo audit ile bağımlılıklardaki bilinen güvenlik açıklarını kontrol etme
2. Sürüm kısıtlamalarını netleştirerek güvenlik yamalarını daha kolay uygulama
3. Kullanılmayan veya nadiren kullanılan bağımlılıkları kaldırarak saldırı yüzeyini azaltma

Bu analiz, bağımlılık yönetimi çalışması için bir başlangıç noktası olacaktır. 