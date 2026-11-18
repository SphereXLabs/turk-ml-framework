# TURK CLI kisayolu - PATH'te cargo olmasa da calisir.
# Kullanim: .\scripts\turk.ps1 check examples/proje.turk
#           .\scripts\turk.ps1 run examples/proje.turk
#           .\scripts\turk.ps1 cihaz
#           .\scripts\turk.ps1 zoo list
#           .\scripts\turk.ps1 zoo calistir xor
# Tek cekirdek test: .\scripts\cargo.ps1 build --no-default-features --features csv

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
$Root = (Resolve-Path (Join-Path $ScriptDir "..")).Path
$TurkExe = Join-Path $Root "target\debug\turk.exe"
$CargoExe = Join-Path $env:USERPROFILE ".cargo\bin\cargo.exe"

function Test-NeedsRebuild {
    param([string]$ExePath, [string]$ProjectRoot)

    if (-not (Test-Path $ExePath)) { return $true }

    $exeTime = (Get-Item $ExePath).LastWriteTime
    $watch = @(
        (Join-Path $ProjectRoot "Cargo.toml"),
        (Join-Path $ProjectRoot "Cargo.lock")
    )
    $srcDir = Join-Path $ProjectRoot "src"
    if (Test-Path $srcDir) {
        $watch += Get-ChildItem -Path $srcDir -Recurse -Filter "*.rs" | ForEach-Object { $_.FullName }
    }

    foreach ($path in $watch) {
        if ((Test-Path $path) -and ((Get-Item $path).LastWriteTime -gt $exeTime)) {
            return $true
        }
    }
    return $false
}

if (Test-Path $CargoExe) {
    Push-Location $Root

    if (Test-NeedsRebuild -ExePath $TurkExe -ProjectRoot $Root) {
        Write-Host "turk yeniden derleniyor..." -ForegroundColor Yellow
        & $CargoExe build
        if ($LASTEXITCODE -ne 0) {
            Pop-Location
            exit $LASTEXITCODE
        }
    }

    if (Test-Path $TurkExe) {
        & $TurkExe @args
        $code = $LASTEXITCODE
        Pop-Location
        exit $code
    }

    & $CargoExe run -- @args
    $code = $LASTEXITCODE
    Pop-Location
    exit $code
}

if (Test-Path $TurkExe) {
    & $TurkExe @args
    exit $LASTEXITCODE
}

Write-Error "turk.exe bulunamadi. Once: .\scripts\cargo.ps1 build"
exit 1
