# Önerilen Görevler

## Yazım Hatası Düzeltme
- **Sorun:** `library_info` çıktısında "Optimizerlar" kelimesi hatalı yazılmış; doğru biçim "Optimizer'lar" olmalı.
- **Konum:** `src/lib.rs`, satır 42.
- **Öneri:** Dizedeki kelimeyi doğru yazımla değiştirerek kullanıcıya sunulan bilgiyi düzeltin.

## Hata Düzeltme
- **Sorun:** `utils::normalize` fonksiyonu standart sapması sıfır olan sütunlarda bölme sıfıra hatasına yol açarak `NaN` üretir.
- **Konum:** `src/utils.rs`, satırlar 15-17.
- **Öneri:** Standart sapması sıfır olan sütunlar için küçük bir epsilon ekleyerek veya değeri 1'e eşitleyerek bölmenin güvenli yapılmasını sağlayın.

## Dokümantasyon Tutarsızlığı
- **Sorun:** Mimari dokümantasyondaki optimizer örneği `Optimizer` trait'inde bulunmayan `new` ve `update` metodlarını gösteriyor.
- **Konum:** `docs/ARCHITECTURE.md`, satırlar 142-155.
- **Öneri:** Örneği güncelleyerek trait'in gerçek imzasıyla (`step`, `learning_rate`, vb.) uyumlu hale getirin.

## Test İyileştirmesi
- **Sorun:** `test_normalize` testi sütun ortalamalarını kontrol etmek istediğini belirtiyor ancak `outer_iter()` satırlar üzerinde döndüğü için gerçekten satırları test ediyor.
- **Konum:** `src/utils.rs`, satırlar 114-119.
- **Öneri:** Testi `Axis(0)` ve `Axis(1)` kontrollerini doğru eksenlerde yapacak şekilde güncelleyin veya iterasyon yönünü düzeltin.
