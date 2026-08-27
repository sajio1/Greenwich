$ErrorActionPreference = "Stop"

$ProjectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
$PythonCommand = if (Get-Command py -ErrorAction SilentlyContinue) {
    "py"
} elseif (Get-Command python -ErrorAction SilentlyContinue) {
    "python"
} else {
    throw "Python 3.10 or newer is required. Install it from python.org first."
}
$PythonArgs = if ($PythonCommand -eq "py") { @("-3.11") } else { @() }

& $PythonCommand @PythonArgs -c "import sys; assert sys.version_info >= (3, 10), 'AlphaMotion requires Python 3.10 or newer'"

$VenvPython = Join-Path $ProjectRoot ".venv\Scripts\python.exe"
if (-not (Test-Path $VenvPython)) {
    & $PythonCommand @PythonArgs -m venv (Join-Path $ProjectRoot ".venv")
}

& $VenvPython -m pip install --upgrade pip
& $VenvPython -m pip install -e $ProjectRoot
& (Join-Path $ProjectRoot ".venv\Scripts\alphamotion.exe") setup @args

Write-Host ""
Write-Host "AlphaMotion is ready. Start it with: .\run.ps1"
