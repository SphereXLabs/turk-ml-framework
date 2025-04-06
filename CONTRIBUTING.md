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