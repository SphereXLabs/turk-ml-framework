# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [3.0.0] — 2026-11-17

### Added

- **WASM browser playground** — tarayıcıda `.turk` DSL çalıştırma (`--features wasm`), GitHub Pages demosu
- **`f32` precision feature** — varsayılan `f64` yerine tek hassasiyet (`--features f32`)
- **Optional BLAS acceleration** — OpenBLAS üzerinden `ndarray` hızlandırma (`--features blas`)
- Sentetik `veri_yukle("kume")` — deterministik 2-küme ikili sınıflandırma veri seti (WASM / playground)

### Changed

- İkili doğruluk metriği: tek çıkışlı modellerde 0.5 eşiği
- Doğrulama / `evaluate` sırasında Dropout ve BatchNorm **inference mode** (`set_training(false)`)
- CI/CD modernizasyonu (fmt, clippy `-D warnings`, çoklu OS test, Docs/Pages WASM yayınlama)

### Security

- PII sanitize, sıkılaştırılmış `.gitignore`, korumalı model/checkpoint yolları

[3.0.0]: https://github.com/SphereXLabs/turk-ml-framework/releases/tag/v3.0.0
