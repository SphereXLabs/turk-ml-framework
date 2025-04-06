# TURK Katkıda Bulunma Kılavuzu

TURK projesine katkıda bulunmak istediğiniz için teşekkür ederiz! Bu belge, katkıda bulunma sürecini kolaylaştırmak için gereken adımları ve kuralları anlatmaktadır.

## Katkıda Bulunma Süreci

1. Projeyi fork edin
2. Feature branch oluşturun (`git checkout -b feature/amazing-feature`)
3. Değişikliklerinizi commit edin (`git commit -m 'Add: some amazing feature'`)
4. Branch'inizi push edin (`git push origin feature/amazing-feature`)
5. Bir Pull Request açın

## Geliştirme Ortamı Kurulumu

### Gereksinimler

- Rust 1.55 veya daha yeni bir sürüm
- Cargo (Rust paket yöneticisi)
- Opsiyonel: Python 3.8+ (Python bağlantıları için)

### Kurulum

```bash
# Projeyi klonlayın
git clone https://github.com/NeonChipher/turk-ml-framework.git
cd turk-ml-framework

# Bağımlılıkları yükleyin ve projeyi derleyin
cargo build

# Testleri çalıştırın
cargo test

# Örnek uygulamayı çalıştırın
cargo run --example mnist_classifier
```

## Kod Standartları

### Kodlama Stili

- Rust'un resmi kod stilini takip edin (rustfmt ile formatlanmalıdır)
- Fonksiyonlar, türler ve modüller için belgelemeler ekleyin
- Karmaşık mantık için açıklayıcı yorumlar ekleyin
- Anlamlı değişken ve fonksiyon isimleri kullanın

### Commit Mesajları

Commit mesajları şu prefixlerden biriyle başlamalıdır:

- `Add:` Yeni özellik eklendiğinde
- `Fix:` Hata düzeltildiğinde
- `Update:` Mevcut kodda iyileştirme yapıldığında
- `Docs:` Dokümantasyon değişikliklerinde
- `Test:` Test kodu eklendiğinde veya değiştirildiğinde
- `Refactor:` Kod yeniden yapılandırıldığında
- `Perf:` Performans iyileştirmeleri yapıldığında

Örnek: `Add: LSTM katmanı desteği eklendi`

## Test

Her yeni özellik veya hata düzeltmesi için:

1. Yeni testler ekleyin
2. `cargo test` ile tüm testlerin çalıştığından emin olun

## Dokümantasyon

- Her public API için belgeleme ekleyin
- Belgelemeler Türkçe ve İngilizce olarak sağlanmalıdır
- Karmaşık algoritmalar için teknik notlar ekleyin
- Örnekler ve kullanım durumları ekleyin

## Geliştirebileceğiniz Alanlar

Aşağıdaki alanlarda katkılar özellikle değerlidir:

### Makine Öğrenimi Bileşenleri

- Yeni Katmanlar
- Yeni Optimizerlar
- Yeni Aktivasyon Fonksiyonları
- Transfer Öğrenme Desteği

### Dil Özellikleri

- Sözdizimi İyileştirmeleri
- Statik Tür Kontrolleri
- IDE Entegrasyonu ve Araçları

### Performans

- SIMD Optimizasyonları
- GPU Desteği
- Dağıtık Hesaplama

### Ekosistem

- Web Arayüzü
- Model Zoo ve Önceden Eğitilmiş Modeller
- Dokümantasyon ve Öğreticiler

## İletişim

- GitHub Issues: Hata raporlamaları ve özellik talepleri için
- E-posta: TurkDilModeli@gmail.com

## Lisans

Projeye katkıda bulunarak, katkılarınızın MIT Lisansı altında yayınlanacağını kabul etmiş olursunuz.

Tekrar katkılarınız için teşekkür ederiz! 