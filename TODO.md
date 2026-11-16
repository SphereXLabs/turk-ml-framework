# TURK Projesi Yapılacaklar Listesi

## Vizyon ve Felsefe

TURK projesi, Türkçe odaklı bir yapay zeka dili ve ekosistemi oluşturmayı hedefler.

## Faz 0 — Sertleştirme (Tamamlandı ✓)

- [x] `derle()` loss seçimi (`loss=mse`, `loss=mae`, …) + `ogrenme_orani`
- [x] Eğitim döngüsünde seçilen loss fonksiyonunun kullanılması
- [x] `model_bilgisi()` gerçek mimari özeti (katman zinciri, param, optimizer, loss)
- [x] `veri_yukle(..., bolme=0.2)` train/val ayrımı + `val_loss` History JSON'a
- [x] Lazy stream epoch shuffle (satır ofset indeksi + karıştırılmış okuma)
- [x] `egit(..., sabir=10, min_delta=0.0001)` erken durdurma DSL parametreleri
- [x] `turk check` statik analiz komutu
- [x] Lazy CSV veri bandı (RAM'de yalnızca batch)

## Kod Temizleme ve Yeniden Düzenleme
- [X] `model.rs` — `Sequential::fit` alt fonksiyonlara ayrıldı
- [X] Kullanılmayan import/değişken temizliği
- [ ] Kodda tekrarlanan mantığı ortak yardımcı fonksiyonlara çıkarma
- [ ] Kod stilini standartlaştırma (rustfmt)

## Test Süreçleri
- [X] Birim testleri (forward, backward, optimizer, hata durumları)
- [X] Entegrasyon testleri (tam eğitim döngüsü, kaydet/yükle)
- [X] `check` komutu testleri
- [X] Lazy CSV + bolme + shuffle testleri
- [ ] Farklı veri formatları için testler (NPZ, IDX)

## Dokümantasyon
- [X] README, INSTALL, EXAMPLES, CONTRIBUTING
- [ ] DSL komut referansı (derle/egit/veri_yukle parametreleri)

## CI/CD
- [X] GitHub Actions CI, docs, release pipeline

## Performans
- [X] Criterion benchmark altyapısı
- [ ] Kritik yolları optimize etme
- [ ] f32 feature flag
- [ ] Paralel hesaplama (Rayon/BLAS)

## Sonraki Faz — Katman Genişletme (Tamamlandı ✓)
- [x] Conv2D + MaxPool2D + Flatten (`KatmanKayit` üzerinden)
- [x] MNIST CNN örneği (`examples/mnist_cnn.turk`)
- [x] Resmi DSL parser + satır/sütun hata konumları (`src/dsl_parser.rs`)
- [x] ONNX export (`turk export`, `onnx_disa_aktar(...)`)
- [x] Python paketi (`pip install -e python/`, `turk-py` CLI)

## Uzun Vade
- [x] Paralel CPU desteği (`cihaz=paralel`, varsayılan feature, `turk cihaz`)
- [ ] Sparse tensor desteği
- [x] Model Zoo (indirilebilir `.turk` + manifest, `turk zoo`, `model_zoo()`)
- [x] GUI (eframe) — canli egitim grafigi (`turk-gui.ps1`)
- [ ] WASM tarayıcı demosu

## Yapılanlar (Özet)
- [x] KatmanKayit modüler fabrika sistemi
- [x] BatchNormalization, EarlyStopping, History JSON export
- [x] Checkpoint + resume (`devam_et=true`)
- [x] CLI: `turk run`, `turk check`, `turk repl`
- [x] Gerçek CSV veri yükleme + lazy streaming
- [x] Conv2D, MaxPool2D, Flatten katmanları + model JSON metadata
- [x] Resmi DSL parser (`dsl_parser`) + satır/sütun hata mesajları
- [x] ONNX JSON export + Python `turk-py` sarmalayıcı
- [x] Siberpunk CLI log + `grafik_ciz.py`
