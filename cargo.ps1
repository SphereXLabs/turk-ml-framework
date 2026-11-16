# cargo kisayolu - Rust PATH'te degilse kullanin.
# Kullanim: .\cargo.ps1 build
#           .\cargo.ps1 build
#           .\cargo.ps1 build --no-default-features --features csv   # tek cekirdek
#           .\cargo.ps1 test
#           .\cargo.ps1 run -- check proje.turk

$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$CargoExe = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"

if (-not (Test-Path $CargoExe)) {
    Write-Error "cargo bulunamadi: $CargoExe (rustup kurulu mu?)"
    exit 1
}

Push-Location $Root
& $CargoExe @args
$code = $LASTEXITCODE
Pop-Location
exit $code
