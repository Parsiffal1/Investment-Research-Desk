$ErrorActionPreference = "Stop"

$principal = New-Object Security.Principal.WindowsPrincipal([Security.Principal.WindowsIdentity]::GetCurrent())
if (-not $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)) {
    throw "Run this script from an elevated PowerShell session. WSL optional features require administrator rights."
}

wsl --install -d Ubuntu
Write-Host "WSL installation command finished. Restart Windows if prompted, then launch Ubuntu once to create the Linux user."
