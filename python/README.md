# TÜRK Python Paketi

Rust `turk.exe` binary'sini Python'dan çağırır.

## Kurulum

```powershell
cd <repo-root>
.\scripts\cargo.ps1 build  # veya tam yol ile cargo build

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
python -m cli check ..\examples\proje.turk
python -m cli run ..\examples\proje.turk
python -m cli export ..\xor_model.json ..\model.onnx.json

# Yöntem 2 — tam yol
& "$env:LOCALAPPDATA\Packages\PythonSoftwareFoundation.Python.3.11_qbz5n2kfra8p0\LocalCache\local-packages\Python311\Scripts\turk-py.exe" check ..\examples\proje.turk

# Yöntem 3 — Python API
python -c "from turk import TurkRunner; r=TurkRunner(); print(r.check('../examples/proje.turk'))"
```

## Proje kökünden (PATH gerekmez)

```powershell
.\scripts\turk.ps1 check examples/proje.turk
.\scripts\turk.ps1 run examples/proje.turk
.\scripts\turk.ps1 export xor_model.json model.onnx.json
```
