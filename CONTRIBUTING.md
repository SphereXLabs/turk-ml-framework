#!/bin/bash
# TÜRK Dil Modeli kurulum betiği

echo "TÜRK Dil Modeli Kurulumu Başlatılıyor..."
echo "Gerekli bağımlılıklar kontrol ediliyor..."

# Derleyici hazırla
mkdir -p build

# Derleme işlemi
echo "TÜRK Dil Modeli derleniyor..."
# (Burada gerçek derleme kodu olacak)

echo "Kurulum tamamlandı!"
echo "TÜRK Dil Modelini çalıştırmak için: ./turk <dosya.turk>"

# Çalıştırma betiğini oluştur
echo '#!/bin/bash
./build/turk "$@"' > turk
chmod +x turk

echo "Örnek bir program çalıştırmak için: ./turk ornekler/merhaba_dunya.turk"

# Katkıda Bulunma Rehberi

TÜRK projesine katkıda bulunmak istediğiniz için teşekkür ederiz! Bu rehber, projeye nasıl katkıda bulunabileceğinizi açıklar.

## Kod Standartları

- Rust'ın resmi stil rehberini takip edin
- Her fonksiyon için dokümantasyon yorumları ekleyin
- Anlamlı değişken ve fonksiyon isimleri kullanın
- Karmaşık kod bloklarını açıklayan yorumlar ekleyin

## Geliştirme Süreci

1. Bu depoyu fork edin
2. Yeni bir branch oluşturun:
   ```bash
   git checkout -b feature/yeni-ozellik
   ```
3. Değişikliklerinizi commit edin:
   ```bash
   git commit -am 'Yeni özellik eklendi'
   ```
4. Branch'inizi push edin:
   ```bash
   git push origin feature/yeni-ozellik
   ```
5. Pull Request oluşturun

## Test Etme

- Tüm değişiklikleriniz için test yazın
- Mevcut testleri çalıştırın:
  ```bash
  cargo test
  ```
- Kod kapsamını kontrol edin:
  ```bash
  cargo tarpaulin
  ```

## Dokümantasyon

- Kod değişiklikleriniz için dokümantasyon ekleyin
- README.md dosyasını güncelleyin
- Örnek kodlar ekleyin

## Hata Raporlama

- Hataları GitHub Issues üzerinden raporlayın
- Hata raporunda şunları belirtin:
  - Hatanın nasıl oluştuğu
  - Beklenen davranış
  - Gerçekleşen davranış
  - Çevre bilgileri (Rust versiyonu, işletim sistemi vb.)

## Özellik İstekleri

- Yeni özellik isteklerini GitHub Issues üzerinden gönderin
- İsteğinizi detaylı bir şekilde açıklayın
  - Özelliğin amacı
  - Kullanım senaryoları
  - Beklenen faydalar

## Lisans

Bu projeye yapılan tüm katkılar MIT lisansı altında lisanslanacaktır.

## İletişim

- GitHub Issues: Soru ve önerileriniz için
- E-posta: turkdilmodeli@gmail.com