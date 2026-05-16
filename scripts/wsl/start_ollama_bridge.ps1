param(
    [string]$HostPort = "0.0.0.0:11435"
)

$ErrorActionPreference = "Stop"
$ollama = (Get-Command ollama -ErrorAction SilentlyContinue).Source
if (-not $ollama) {
    throw "ollama.exe was not found on PATH. Install Ollama or add it to PATH first."
}

$existing = Get-NetTCPConnection -LocalPort ($HostPort.Split(":")[-1]) -State Listen -ErrorAction SilentlyContinue
if ($existing) {
    Write-Host "Ollama bridge already listening on $HostPort"
    exit 0
}

Start-Process -FilePath "powershell" -ArgumentList @(
    "-NoProfile",
    "-Command",
    "`$env:OLLAMA_HOST='$HostPort'; & '$ollama' serve"
) -WindowStyle Hidden

Start-Sleep -Seconds 3
$port = $HostPort.Split(":")[-1]
Invoke-RestMethod -Uri "http://localhost:$port/v1/models" -TimeoutSec 5 | Out-Null
Write-Host "Ollama bridge started on $HostPort"
