# ALPER — Göktürk Kokpit kisayolu (turk-sistem'e yonlendirir)
# Kullanim: .\scripts\alper.ps1
# Harici LLM API yok; yerel turk motoru + kokpit WASM beyni.

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
& (Join-Path $ScriptDir "turk-sistem.ps1") @args
exit $LASTEXITCODE
