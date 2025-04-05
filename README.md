# 🚀 Türkçe Yapay Zeka Çerçevesi (TurkML)

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Rust Version](https://img.shields.io/badge/Rust-1.70%2B-blue)](https://www.rust-lang.org)
[![Build Status](https://github.com/NeonChipher/turk-ml-framework/actions/workflows/rust.yml/badge.svg)](https://github.com/NeonChipher/turk-ml-framework/actions)

Rust ile geliştirilen, Türkçe DSL (Domain-Specific Language) destekli makine öğrenimi kütüphanesi. MNIST sınıflandırma örneği dahildir.

<img src="https://example.com/project-preview.gif" width="600" alt="Proje Demo">

## 🌟 Öne Çıkan Özellikler
- Türkçe sözdizimi ile sinir ağı tanımlama
- Yerleşik optimizasyon algoritmaları (SGD, Adam)
- MNIST veri seti için hazır örnek
- Model serileştirme (JSON) desteği

## 📦 Kurulum

### Ön Koşullar
- Rust 1.70+ (`rustup update` ile güncelleyin)
- Cargo paket yöneticisi

```bash
# Repoyu klonla
git clone https://github.com/NeonChipher/turk-ml-framework.git
cd turk-ml-framework

# Bağımlılıklarla birlikte derle
cargo build --release
🧠 Örnek Kullanım
Türkçe DSL ile Model Tanımı
rust
Copy
Model sinir_agi {
    Katman(784, Aktivasyon = Relu)
    Katman(128, Aktivasyon = Sigmoid)
    Katman(10,  Aktivasyon = Softmax)
    OgrenmeOrani = 0.01
    Epoch = 30
}
MNIST Sınıflandırma Çalıştırma
bash
Copy
cargo run --example mnist_classifier
🛠 Mimarı
mermaid
Copy
graph TD
    A[Kullanıcı Kodu] --> B[TurkML DSL]
    B --> C[Parser]
    C --> D[Yorumlayıcı]
    D --> E[NDArray İşlemleri]
    E --> F[Sonuçlar]
📚 Dokümantasyon
Bileşen	Açıklama
src/model.rs	Sequential model implementasyonu
src/layer.rs	Dense katman ve aktivasyonlar
examples/	Kullanım örnekleri
🤝 Katkıda Bulunma
Fork'layın (https://github.com/NeonChipher/turk-ml-framework/fork)

Yeni branch oluşturun (git checkout -b feature/awesome-feature)

Commit'leyin (git commit -am 'Add awesome feature')

Push'layın (git push origin feature/awesome-feature)

Pull Request açın

📜 Lisans
MIT Lisansı - Detaylar için LICENSE dosyasına bakın.

✨ Teşekkürler
Rust ekibine

MNIST veri seti sağlayıcılarına

Copy

---

### 🎨 Ekstra Özellikler Eklemek İçin:
1. **Badge'ler**: 
   ```markdown
   [![Codecov](https://codecov.io/gh/NeonChipher/turk-ml-framework/branch/main/graph/badge.svg)](https://codecov.io/gh/NeonChipher/turk-ml-framework)
Demo Gif:

markdown
Copy
![Demo](https://github.com/NeonChipher/turk-ml-framework/blob/main/assets/demo.gif)
API Referansı (docs.rs bağlantısı):

markdown
Copy
[![docs.rs](https://docs.rs/turk-ml/badge.svg)](https://docs.rs/turk-ml)
