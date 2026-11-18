# Geriye uyumluluk — turk-sistem.ps1'e yonlendirir
# Kullanim: .\scripts\jarvis.ps1

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ScriptDir "turk-sistem.ps1") @args
exit $LASTEXITCODE
