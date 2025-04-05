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
