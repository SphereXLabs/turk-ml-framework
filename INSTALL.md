# TÜRK Projesi Kurulum Rehberi

Bu rehber, TÜRK (Türkçe Yapay Sinir Ağları) projesini farklı işletim sistemlerinde kurma ve çalıştırma adımlarını detaylı olarak açıklar.

## İçindekiler

- [Gereksinimler](#gereksinimler)
- [Hızlı Kurulum](#hızlı-kurulum)
- [Detaylı Kurulum Adımları](#detaylı-kurulum-adımları)
  - [Windows](#windows)
  - [macOS](#macos)
  - [Linux](#linux)
- [Bağımlılıklar](#bağımlılıklar)
- [Konfigürasyon](#konfigürasyon)
- [Sorun Giderme](#sorun-giderme)
- [Sık Sorulan Sorular](#sık-sorulan-sorular)

## Gereksinimler

TÜRK projesini kurmak için aşağıdaki yazılımların yüklü olması gerekmektedir:

- **Rust** (1.70.0 veya daha yeni): Rust dili ve Cargo paket yöneticisi
- **Git**: Kaynak kodu indirmek için
- **C/C++ derleyicisi**: Bazı bağımlılıkların derlenmesi için
  - Windows: MSVC veya MinGW
  - macOS: Xcode Command Line Tools
  - Linux: GCC veya Clang
- **OpenSSL**: Güvenli ağ iletişimi için (sistem kütüphanesi)
- **Python 3.8+**: Opsiyonel, veri işleme ve örnekler için

## Hızlı Kurulum

Hızlı kurulum için aşağıdaki adımları izleyebilirsiniz:

```bash
# Depoyu klonlayın
git clone https://github.com/kullaniciadi/turk.git
cd turk

# Projeyi derleyin
cargo build --release

# Binary'yi çalıştırın
./target/release/turk

# Veya doğrudan çalıştırın
cargo run --release
```

## Detaylı Kurulum Adımları

### Windows

1. **Rust'ı Yükleyin**:
   - [rustup.rs](https://rustup.rs/) adresinden rustup-init.exe'yi indirin ve çalıştırın
   - "1) Proceed with installation" seçeneğini seçin
   - Kurulum tamamlandığında PowerShell veya Command Prompt'u yeniden başlatın

2. **Git'i Yükleyin**:
   - [Git for Windows](https://gitforwindows.org/) adresinden Git'i indirin ve yükleyin
   - Standart seçeneklerle kurulumu tamamlayın

3. **Visual Studio Build Tools'u Yükleyin**:
   - [Visual Studio Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) indirin
   - "C++ ile Masaüstü Geliştirme" iş yükünü seçin
   - Kurulumu tamamlayın

4. **TÜRK Projesini İndirin ve Derleyin**:
   ```batch
   git clone https://github.com/kullaniciadi/turk.git
   cd turk
   cargo build --release
   ```

5. **Path'e Ekleyin** (Opsiyonel):
   - `target\release` dizinindeki `turk.exe` dosyasını sistem PATH'ine ekleyin veya
   - PowerShell ile kısayol oluşturun:
     ```powershell
     New-Item -ItemType SymbolicLink -Path "$Home\AppData\Local\Microsoft\WindowsApps\turk.exe" -Target "$(Get-Location)\target\release\turk.exe"
     ```

### macOS

1. **Rust'ı Yükleyin**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Xcode Command Line Tools'u Yükleyin**:
   ```bash
   xcode-select --install
   ```

3. **Homebrew ile Bağımlılıkları Yükleyin** (Opsiyonel):
   ```bash
   brew install openssl
   ```

4. **TÜRK Projesini İndirin ve Derleyin**:
   ```bash
   git clone https://github.com/kullaniciadi/turk.git
   cd turk
   cargo build --release
   ```

5. **Path'e Ekleyin** (Opsiyonel):
   ```bash
   echo 'export PATH="$PATH:$(pwd)/target/release"' >> ~/.zshrc
   # veya bash kullanıyorsanız
   echo 'export PATH="$PATH:$(pwd)/target/release"' >> ~/.bashrc
   source ~/.zshrc  # veya ~/.bashrc
   ```

### Linux

1. **Rust'ı Yükleyin**:
   ```bash
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source $HOME/.cargo/env
   ```

2. **Gerekli Paketleri Yükleyin**:
   - Debian/Ubuntu:
     ```bash
     sudo apt update
     sudo apt install build-essential pkg-config libssl-dev
     ```
   - Fedora:
     ```bash
     sudo dnf install gcc openssl-devel
     ```
   - Arch Linux:
     ```bash
     sudo pacman -S base-devel openssl
     ```

3. **TÜRK Projesini İndirin ve Derleyin**:
   ```bash
   git clone https://github.com/kullaniciadi/turk.git
   cd turk
   cargo build --release
   ```

4. **Path'e Ekleyin** (Opsiyonel):
   ```bash
   echo 'export PATH="$PATH:$(pwd)/target/release"' >> ~/.bashrc
   source ~/.bashrc
   ```

## Bağımlılıklar

TÜRK projesi aşağıdaki ana bağımlılıklara sahiptir:

| Bağımlılık | Sürüm | Açıklama |
|------------|-------|----------|
| ndarray | ^0.15 | Çok boyutlu dizi işlemleri |
| serde | ^1.0 | Serileştirme/deserileştirme |
| rand | ^0.8 | Rastgele sayı üretimi |
| log | ^0.4 | Loglama |
| clap | ^4.0 | Komut satırı argüman işleme |

Cargo.toml dosyasında tüm bağımlılıkların listesi bulunmaktadır. Cargo, bu bağımlılıkları otomatik olarak indirecek ve derleyecektir.

## Konfigürasyon

TÜRK projesini yapılandırmak için aşağıdaki seçenekleri kullanabilirsiniz:

### Cargo Features

Farklı özellikler için Cargo features kullanabilirsiniz:

```bash
# GUI desteği ile derleme
cargo build --release --features="gui"

# Yalnızca CLI sürümünü derleme
cargo build --release --features="cli"

# Paralel CPU (varsayilan — ek flag gerekmez)
cargo build --release

# Tek cekirdek (paralel kapali)
cargo build --release --no-default-features --features csv
```

### Ortam Değişkenleri

Çalışma zamanında davranışı değiştirmek için çeşitli ortam değişkenleri ayarlayabilirsiniz:

- `TURK_LOG_LEVEL`: Log seviyesini ayarlar (debug, info, warn, error)
- `TURK_DATA_DIR`: Veri dosyalarının konumunu belirtir
- `TURK_MODEL_PATH`: Varsayılan model yolunu belirtir

Örnek:
```bash
TURK_LOG_LEVEL=debug TURK_DATA_DIR=/home/user/turk_data ./target/release/turk
```

### Konfigürasyon Dosyası

TÜRK, aşağıdaki sırayla konfigürasyon dosyası arar:

1. Komut satırında belirtilen dosya: `turk --config /path/to/config.toml`
2. Çalışma dizinindeki `turk.toml` dosyası
3. Kullanıcı dizinindeki `.turk/config.toml` dosyası

Örnek konfigürasyon dosyası:

```toml
[general]
log_level = "info"
data_dir = "/usr/local/share/turk/data"

[model]
default_optimizer = "adam"
default_learning_rate = 0.001

[network]
threads = 4
```

## Sorun Giderme

### Derleme Hataları

1. **Bağımlılık Hatası**:
   ```
   error: failed to run custom build command for `openssl-sys`
   ```
   
   **Çözüm**: OpenSSL geliştirme paketlerini yükleyin:
   - Windows: `vcpkg install openssl:x64-windows-static`
   - macOS: `brew install openssl`
   - Linux: `sudo apt install libssl-dev` (Debian/Ubuntu)

2. **Rust Sürüm Hatası**:
   ```
   error: the trait bound `Foo: Bar` is not satisfied
   ```
   
   **Çözüm**: Rust'ı güncelleyin:
   ```bash
   rustup update stable
   ```

3. **Hedef Platformu Hatası**:
   ```
   error: failed to select a version for the requirement `foo = "^1.0"`
   ```
   
   **Çözüm**: Cargo.lock dosyasını silin ve yeniden deneyin:
   ```bash
   rm Cargo.lock
   cargo build
   ```

### Çalışma Zamanı Hataları

1. **Model Yükleme Hatası**:
   ```
   Error: Failed to load model from path: ./model.json
   ```
   
   **Çözüm**: Doğru model dosyasının mevcut olduğundan emin olun veya yeni bir model oluşturun:
   ```bash
   turk model_olustur --save model.json
   ```

2. **Bellek Hatası**:
   ```
   Error: Failed to allocate memory for tensor
   ```
   
   **Çözüm**: Daha küçük batch boyutu kullanın:
   ```bash
   turk egit --batch-size 32  # 64 veya 128 yerine
   ```

## Sık Sorulan Sorular

**S: TÜRK projesini Windows'ta WSL (Windows Subsystem for Linux) ile kullanabilir miyim?**  
C: Evet, WSL2 ile Linux kurulum adımlarını izleyerek TÜRK'ü kullanabilirsiniz. WSL2, daha iyi performans sağlar.

**S: TÜRK'ü Docker ile kullanabilir miyim?**  
C: Evet, projenin kök dizininde bir Dockerfile bulunmaktadır. Aşağıdaki komutlarla Docker üzerinde çalıştırabilirsiniz:
```bash
docker build -t turk .
docker run -it turk
```

**S: TÜRK için önerilen donanım gereksinimleri nelerdir?**  
C: Temel kullanım için minimum gereksinimler:
- 4 GB RAM
- Çift çekirdekli CPU
- 1 GB disk alanı

Daha büyük modeller için:
- 8+ GB RAM
- 4+ çekirdekli CPU
- GPU (isteğe bağlı, CUDA destekli)

**S: TÜRK'ü Python ile nasıl kullanabilirim?**  
C: TÜRK'ün Python bağlayıcısı şu anda geliştirme aşamasındadır. `/python` dizininde Python bağlayıcıları hakkında daha fazla bilgi bulabilirsiniz.

---

TÜRK projesinin kurulumu veya kullanımı hakkında başka sorularınız varsa, lütfen [GitHub Issues](https://github.com/kullaniciadi/turk/issues) üzerinden bir soru açın veya [Discord sunucumuza](https://discord.gg/turk-community) katılın. 