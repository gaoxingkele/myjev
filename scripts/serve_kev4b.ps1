$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $PSScriptRoot
$env:HF_HUB_CACHE = Join-Path $Root "hf_cache"
$env:HF_HOME = Join-Path $Root "hf_home"
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = "1"
$KevRepo = Join-Path $Root "repos\kev"
$Py = Join-Path $KevRepo ".venv\Scripts\python.exe"
# IMPORTANT: do not use `uv run` — it re-syncs CPU torch from the lockfile.
# Ensure Qwen3.5-4B-Base is present in HF_HUB_CACHE before starting.
Set-Location $KevRepo
Write-Host "HF_HUB_CACHE=$env:HF_HUB_CACHE"
Write-Host "Serving Kev-4B on :8009 ..."
& $Py -m kev.serve --run jaredpalmer/kev-4b --port 8009 --device cuda
