param(
    [string]$RepoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path,
    [switch]$IncludeUserCaches,
    [switch]$IncludeGitConfig
)

$ErrorActionPreference = "Stop"

function Test-IsAdministrator {
    $identity = [Security.Principal.WindowsIdentity]::GetCurrent()
    $principal = New-Object Security.Principal.WindowsPrincipal($identity)
    return $principal.IsInRole([Security.Principal.WindowsBuiltInRole]::Administrator)
}

function Repair-PathAcl {
    param(
        [Parameter(Mandatory = $true)]
        [string]$Target,
        [Parameter(Mandatory = $true)]
        [string]$UserName
    )

    if (-not (Test-Path -LiteralPath $Target)) {
        Write-Host "Skipping missing path: $Target"
        return
    }

    Write-Host ""
    Write-Host "Repairing: $Target"
    & takeown.exe /F $Target /R /D Y | Out-Null
    & icacls.exe $Target /inheritance:e /T /C | Out-Null
    & icacls.exe $Target /setowner $UserName /T /C | Out-Null
    & icacls.exe $Target /grant "${UserName}:(OI)(CI)F" /T /C | Out-Null

    $acl = Get-Acl -LiteralPath $Target
    Write-Host "Owner: $($acl.Owner)"
}

$userName = "$env:USERDOMAIN\$env:USERNAME"
$targets = @(
    $RepoRoot,
    (Join-Path $RepoRoot ".venv"),
    (Join-Path $RepoRoot "tmp")
)

if ($IncludeUserCaches) {
    $targets += (Join-Path $env:LOCALAPPDATA "uv\\cache")
}

if ($IncludeGitConfig) {
    $targets += (Join-Path $HOME ".config\\git")
}

$targets = $targets | Select-Object -Unique

Write-Host "User:    $userName"
Write-Host "Repo:    $RepoRoot"
Write-Host "Admin:   $(Test-IsAdministrator)"
Write-Host "Targets:"
$targets | ForEach-Object { Write-Host "  - $_" }

if (-not (Test-IsAdministrator)) {
    Write-Warning "Running without elevation may be enough for user-owned paths, but Administrator PowerShell is recommended for stubborn ACL issues."
}

foreach ($target in $targets) {
    Repair-PathAcl -Target $target -UserName $userName
}

Write-Host ""
Write-Host "ACL repair complete."
