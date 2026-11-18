# TURK GUI kisayolu
# Kullanim: .\scripts\turk-gui.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$CargoExe = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"
$GuiExe = Join-Path $Root "target\debug\turk_gui.exe"

if (-not (Test-Path $CargoExe)) {
    Write-Error "cargo bulunamadi: $CargoExe"
    exit 1
}

Push-Location $Root

if (-not (Test-Path $GuiExe)) {
    Write-Host "turk_gui derleniyor (--features gui)..." -ForegroundColor Yellow
    & $CargoExe build --features gui --bin turk_gui
    if ($LASTEXITCODE -ne 0) {
        Pop-Location
        exit $LASTEXITCODE
    }
}

& $GuiExe @args
$code = $LASTEXITCODE
Pop-Location
exit $code
