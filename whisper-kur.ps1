# TÜRK Whisper kurulumu — whisper-cli + ggml-small model
# Kullanim: .\whisper-kur.ps1

$ErrorActionPreference = "Stop"

$Kok = Join-Path $env:USERPROFILE ".turk\sistem"
$Bin = Join-Path $Kok "bin"
$Models = Join-Path $Kok "models"
$Tmp = Join-Path $env:TEMP "turk-whisper-kur"

New-Item -ItemType Directory -Force -Path $Bin, $Models, $Tmp | Out-Null

$CliHedef = Join-Path $Bin "whisper-cli.exe"
$ModelHedef = Join-Path $Models "ggml-small.bin"

Write-Host "TÜRK Whisper kurulumu" -ForegroundColor Cyan
Write-Host "  CLI  -> $CliHedef"
Write-Host "  Model -> $ModelHedef"
Write-Host ""

# --- whisper-cli ---
$ZipUrl = "https://github.com/ggml-org/whisper.cpp/releases/download/v1.9.1/whisper-bin-x64.zip"
$ZipPath = Join-Path $Tmp "whisper-bin-x64.zip"
$Cikart = Join-Path $Tmp "whisper"

if (-not (Test-Path $CliHedef) -or -not (Get-ChildItem $Bin -Filter "*.dll" -ErrorAction SilentlyContinue)) {
    Write-Host "[1/2] whisper.cpp indiriliyor..." -ForegroundColor Yellow
    if (-not (Test-Path $ZipPath)) {
        Invoke-WebRequest -Uri $ZipUrl -OutFile $ZipPath -UseBasicParsing
    }
    if (Test-Path $Cikart) { Remove-Item $Cikart -Recurse -Force }
    Expand-Archive -Path $ZipPath -DestinationPath $Cikart -Force

    $KaynakDir = Get-ChildItem -Path $Cikart -Recurse -Filter "whisper-cli.exe" -ErrorAction SilentlyContinue |
        Select-Object -First 1 | ForEach-Object { $_.DirectoryName }
    if (-not $KaynakDir) {
        $KaynakDir = Get-ChildItem -Path $Cikart -Recurse -Filter "main.exe" -ErrorAction SilentlyContinue |
            Select-Object -First 1 | ForEach-Object { $_.DirectoryName }
    }
    if (-not $KaynakDir) {
        Write-Error "whisper-cli.exe zip icinde bulunamadi."
    }

    # Tum DLL ve whisper-cli.exe kopyala (DLL eksikligi STATUS_DLL_NOT_FOUND verir)
    Copy-Item (Join-Path $KaynakDir "*.dll") $Bin -Force
    Copy-Item (Join-Path $KaynakDir "whisper-cli.exe") $Bin -Force
    Write-Host "[OK] whisper-cli + DLL'ler kuruldu ($(( Get-ChildItem $Bin -Filter '*.dll' ).Count) DLL)." -ForegroundColor Green
} else {
    Write-Host "[OK] whisper-cli.exe ve DLL'ler zaten var." -ForegroundColor Green
}

# --- model ---
if (Test-Path $ModelHedef) {
    $mb = [math]::Round((Get-Item $ModelHedef).Length / 1MB)
    Write-Host "[OK] ggml-small.bin zaten var ($mb MB)." -ForegroundColor Green
} else {
    $ModelUrl = "https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-small.bin"
    Write-Host "[2/2] ggml-small.bin indiriliyor (~466 MB, biraz surebilir)..." -ForegroundColor Yellow
    Invoke-WebRequest -Uri $ModelUrl -OutFile $ModelHedef -UseBasicParsing
    Write-Host "[OK] Model kuruldu." -ForegroundColor Green
}

Write-Host ""
Write-Host "Kurulum tamam. TÜRK sistemini yeniden baslatin:" -ForegroundColor Cyan
Write-Host "  .\turk-sistem.ps1"
