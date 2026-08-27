$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$AlphaMotion = Join-Path $ProjectRoot ".venv\Scripts\alphamotion.exe"
if (-not (Test-Path $AlphaMotion)) {
    throw "AlphaMotion is not installed. Run .\install.ps1 first."
}
& $AlphaMotion serve @args
