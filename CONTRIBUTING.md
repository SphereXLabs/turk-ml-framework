# TÜRK Projesine Katkıda Bulunma Rehberi

Türkçe yapay zeka dilinin gelişimine katkıda bulunmak istediğiniz için teşekkür ederiz! Bu rehber, TÜRK projesine nasıl katkıda bulunabileceğinizi ve projeye dahil olma süreçlerini detaylı olarak açıklar.

## 🎯 Katkı Vizyonumuz

TÜRK projesi, Türkçe konuşan geliştiricilerin yapay zeka alanında kendi dillerinde çalışabilmeleri için bir ekosistem oluşturmayı amaçlar. Katkılarınız, bu vizyonu gerçekleştirmemize yardımcı olacaktır:

1. **Kendi Dilimiz İle Geliştirme**: Türkçe programlama dili geliştirme sürecine katkıda bulunun
2. **Kendi Yapay Zekamızı Kendi Dilimizle Eğitmek**: Model geliştirme ve eğitim süreçlerine dahil olun
3. **Topluluk Odaklı Erişim**: Topluluk oluşturma ve erişim çalışmalarına katkı sağlayın

## 🚩 Başlangıç İçin Adımlar

1. **Projeyi Fork Edin**: GitHub'da projeyi fork ederek kendi kopyanızı oluşturun.
2. **Yerel Geliştirme Ortamını Kurun**:
   ```bash
   git clone https://github.com/kullaniciadi/turk.git
   cd turk
   cargo build
   ```
3. **Yeni Bir Dal Oluşturun**:
   ```bash
   git checkout -b ozellik/yeni-ozellik
   # veya
   git checkout -b duzeltme/hata-duzeltme
   ```
4. **Değişikliklerinizi Yapın**: Yeni özellik ekleyin veya hataları düzeltin.
5. **Testleri Çalıştırın**:
   ```bash
   cargo test
   ```
6. **Değişikliklerinizi Commit Edin**:
   ```bash
   git commit -am "Yeni özellik: XYZ eklendi"
   ```
7. **Dalınızı Push Edin**:
   ```bash
   git push origin ozellik/yeni-ozellik
   ```
8. **Pull Request Açın**: GitHub üzerinden bir Pull Request açın.

## 📂 Katkı Alanları

TÜRK projesine katkıda bulunmak için birçok alan vardır:

### 1. Dil Geliştirme

TÜRK dilinin sözdizimi ve yeteneklerinin geliştirilmesi:

- **DSL Geliştirme**: Türkçe programlama dilinin sözdizimi ve semantiğini geliştirin
- **Parser İyileştirmeleri**: Dil ayrıştırıcısının geliştirilmesi
- **Hata Mesajları**: Daha anlaşılır Türkçe hata mesajları ekleyin

### 2. Yapay Sinir Ağı Bileşenleri

Makine öğrenimi bileşenlerinin geliştirilmesi:

- **Katmanlar**: Yeni katman türleri ekleyin (CNN, RNN, Transformer vb.)
- **Optimizasyon Algoritmaları**: Yeni optimizasyon algoritmaları ekleyin
- **Aktivasyon Fonksiyonları**: Farklı aktivasyon fonksiyonları ekleyin
- **Kayıp Fonksiyonları**: Çeşitli kayıp fonksiyonları geliştirin
- **Metrikler**: Performans metrikleri ekleyin

### 3. Performans İyileştirmeleri

- **Algoritma Optimizasyonları**: Mevcut algoritmaların verimliliğini artırın
- **Paralelleştirme**: Çoklu iş parçacığı desteği ekleyin
- **SIMD Optimizasyonları**: Vektör işlemleri için optimizasyonlar
- **GPU Desteği**: CUDA veya OpenCL entegrasyonu ekleyin

### 4. Dokümantasyon ve Öğretici İçerikler

- **API Dokümantasyonu**: Fonksiyon ve modüllerin dokümantasyonunu geliştirin
- **Öğreticiler**: Adım adım rehberler ve eğitim materyalleri oluşturun
- **Örnek Projeler**: Yeni kullanım senaryoları için örnek projeler geliştirin
- **Türkçe Kaynaklar**: Yapay zeka kavramlarını Türkçe açıklayan kaynaklar ekleyin

### 5. Araçlar ve Ekosistem

- **Görselleştirme Araçları**: Modelleri ve eğitim sürecini görselleştirmek için araçlar
- **Model Dönüştürücüler**: Diğer frameworklerden model dönüştürme araçları
- **IDE Eklentileri**: Yaygın IDE'ler için eklentiler geliştirme
- **CLI İyileştirmeleri**: Komut satırı araçları geliştirme

### 6. Test ve Kalite

- **Birim Testleri**: Yeni özellikler için testler yazın
- **Entegrasyon Testleri**: Sistemin bütünsel testlerini ekleyin
- **Performans Testleri**: Benchmark testleri ekleyin
- **Belgelendirme Testleri**: Örnek kodların çalıştığından emin olun

## 📝 Kod Standartları

TÜRK projesinde kod kalitesini korumak için aşağıdaki standartlara uyun:

1. **Rust Stil Rehberi**: [Rust API Guidelines](https://rust-lang.github.io/api-guidelines/) dokümanını takip edin.
2. **Dokümantasyon Yorumları**:
   - Her fonksiyon ve struct için `///` dokümantasyon yorumları ekleyin
   - Örnekler ekleyin: ```rust ... ```
   - Parametreleri ve dönüş değerlerini açıklayın
3. **İsimlendirme Kuralları**:
   - Fonksiyonlar için `snake_case` kullanın
   - Türler ve Trait'ler için `PascalCase` kullanın
   - Modüller için `snake_case` kullanın
4. **Kod Organizasyonu**:
   - İlgili fonksiyonları aynı modülde tutun
   - Mantıksal birimleri uygun dosyalara ayırın
5. **Hata İşleme**:
   - Panik yerine `Result` türlerini kullanın
   - Anlamlı hata mesajları sağlayın

## 🧪 Test Etme

Her değişiklik için test eklenmeli ve mevcut testler başarıyla çalıştırılmalıdır:

```bash
# Tüm testleri çalıştır
cargo test

# Belirli bir test modülünü çalıştır
cargo test --package turk --lib tests::model_tests

# Belirli bir testi çalıştır
cargo test test_sequential_model

# Kod kapsamını kontrol et (cargo-tarpaulin gerektirir)
cargo install cargo-tarpaulin
cargo tarpaulin
```

## 🔍 Kod İnceleme Süreci

1. Pull Request'iniz en az bir katkıcı tarafından incelenecektir
2. CI testleri başarılı olmalıdır
3. Kod standardı uygunluğu kontrol edilecektir
4. Değişiklikleriniz onaylandıktan sonra ana dala birleştirilecektir

## 🐛 Hata Raporlama

Hataları raporlamak için GitHub Issues kullanın:

1. Benzer bir sorunun zaten bildirilip bildirilmediğini kontrol edin
2. [Bug Report] başlığıyla yeni bir issue açın
3. Aşağıdaki bilgileri ekleyin:
   - Hatanın nasıl oluştuğunu gösteren adımlar
   - Beklenen davranış
   - Gerçekleşen davranış
   - Çalışma ortamı (Rust versiyonu, işletim sistemi, vb.)
   - Varsa hata mesajları ve loglar
   - Mümkünse sorunu gösteren minimal bir kod örneği

## 💡 Özellik İstekleri

Yeni özellik önerileriniz için:

1. [Feature Request] başlığıyla yeni bir issue açın
2. Aşağıdaki bilgileri ekleyin:
   - Özelliğin amacı ve açıklaması
   - Kullanım senaryoları
   - Beklenen faydalar
   - Mümkünse örnek sözdizimi veya API kullanımı

## 🗣️ İletişim ve Topluluk

- **GitHub Discussions**: Projemiz hakkında tartışmalar için
- **Discord**: Gerçek zamanlı sohbet ve topluluk etkileşimi için [Discord Sunucumuz](https://discord.gg/turk-community)
- **E-posta**: Özel iletişim için turkdilmodeli@gmail.com
- **Haftalık Toplantılar**: Her Cuma 20:00'de (Türkiye saati) Discord üzerinden

## 📄 Lisans

Bu projeye yapılan tüm katkılar, projenin lisansı olan MIT lisansı altında lisanslanacaktır. Pull Request göndererek, bu koşulları kabul etmiş sayılırsınız.

---

TÜRK projesine olan ilginiz ve katkılarınız için tekrar teşekkür ederiz! Türkçe yapay zeka ekosistemini birlikte inşa edelim.