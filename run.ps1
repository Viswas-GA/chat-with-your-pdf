# Run Chat with Your PDF from the project root (double-click or: .\run.ps1)
$ProjectRoot = $PSScriptRoot
Set-Location $ProjectRoot

# Ollama installer may not be on PATH until you restart the terminal
$OllamaDir = "$env:LOCALAPPDATA\Programs\Ollama"
if ((Test-Path "$OllamaDir\ollama.exe") -and ($env:Path -notlike "*$OllamaDir*")) {
    $env:Path = "$OllamaDir;$env:Path"
}

if (-not (Test-Path "$ProjectRoot\.venv\Scripts\streamlit.exe")) {
    Write-Host "Virtual environment not found. Run first:" -ForegroundColor Yellow
    Write-Host "  python -m venv .venv"
    Write-Host "  .\.venv\Scripts\pip install -r requirements.txt"
    exit 1
}

if (-not (Test-Path "$ProjectRoot\.env")) {
    Copy-Item "$ProjectRoot\.env.example" "$ProjectRoot\.env"
    Write-Host "Created .env (Ollama = free). Install Ollama and run: ollama pull llama3.2" -ForegroundColor Yellow
}

Write-Host "Starting app at http://localhost:8501 ..." -ForegroundColor Green
& "$ProjectRoot\.venv\Scripts\streamlit.exe" run "$ProjectRoot\app.py"
