# cargo kisayolu - Rust PATH'te degilse kullanin.
# Kullanim: .\scripts\cargo.ps1 build
#           .\scripts\cargo.ps1 test
#           .\scripts\cargo.ps1 run -- check examples/proje.turk

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
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
