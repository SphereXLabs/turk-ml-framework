# TÜRK Python Paketi

Rust `turk.exe` binary'sini Python'dan çağırır.

## Kurulum

```powershell
cd <repo-root>
.\cargo.ps1 build          # veya tam yol ile cargo build

cd python
pip install -e .
```

## Önemli: TURK_BIN

`TURK_BIN` **cargo.exe değil**, **turk.exe** olmalı:

```powershell
# python/ klasöründeyken:
$env:TURK_BIN = "..\target\debug\turk.exe"

# Doğrulama:
python -c "from turk import TurkRunner; print(TurkRunner().binary)"
# Çıktı: ...\target\debug\turk.exe
```

TURK_BIN ayarlanmazsa paket otomatik olarak üst klasörlerde `target/debug/turk.exe` arar.

## turk-py PATH'te değilse

pip uyarısındaki Scripts klasörü PATH'te olmayabilir. Şunlardan birini kullanın:

```powershell
# Yöntem 1 — python -m (önerilen)
cd python
$env:TURK_BIN = "..\target\debug\turk.exe"
python -m cli check ..\proje.turk
python -m cli run ..\proje.turk
python -m cli export ..\xor_model.json ..\model.onnx.json

# Yöntem 2 — tam yol
& "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\turk-py.exe" check ..\proje.turk

# Yöntem 3 — Python API
python -c "from turk import TurkRunner; r=TurkRunner(); print(r.check('../proje.turk'))"
```

## Proje kökünden (PATH gerekmez)

```powershell
.\turk.ps1 check proje.turk
.\turk.ps1 run proje.turk
.\turk.ps1 export xor_model.json model.onnx.json
```
