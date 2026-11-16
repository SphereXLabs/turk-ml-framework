# TÜRK Çekirdeği GUI kisayolu
# Kullanim: .\turk-sistem.ps1
# Her calistirmada kaynak degistiysa otomatik yeniden derler (cargo run).

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$CargoExe = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"

if (-not (Test-Path $CargoExe)) {
    Write-Error "cargo bulunamadi: $CargoExe"
    exit 1
}

Push-Location $Root
try {
    [Console]::OutputEncoding = [System.Text.UTF8Encoding]::new()
} catch {
    # Eski PowerShell surumleri
}
Write-Host "TURK Sistemi baslatiliyor (gerekirse derlenir)..." -ForegroundColor Cyan
& $CargoExe run --features turk_sistem --bin turk_sistem -- @args
$code = $LASTEXITCODE
Pop-Location
exit $code
