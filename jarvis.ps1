# Geriye uyumluluk — turk-sistem.ps1'e yonlendirir
& (Join-Path (Split-Path -Parent $MyInvocation.MyCommand.Path) "turk-sistem.ps1") @args
exit $LASTEXITCODE
