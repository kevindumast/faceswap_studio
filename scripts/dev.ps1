# Lance l'API (127.0.0.1:8000), le worker de rendu et le frontend Vite (127.0.0.1:5173), chacun dans sa fenêtre.
# Usage (depuis la racine du repo) :  .\scripts\dev.ps1
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Environnement Python absent. Lance d'abord :  uv venv -p 3.11 ; uv pip install -e ." -ForegroundColor Yellow
    exit 1
}
if (-not (Test-Path (Join-Path $root "app\web\node_modules"))) {
    Write-Host "Installation des dépendances du frontend..." -ForegroundColor Cyan
    Push-Location (Join-Path $root "app\web"); npm install; Pop-Location
}

Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='API'; & '$py' -m app.api.main"
Start-Process powershell -WorkingDirectory $root -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Worker'; & '$py' -m app.worker.main"
Start-Process powershell -WorkingDirectory (Join-Path $root "app\web") -ArgumentList "-NoExit", "-Command", "`$host.UI.RawUI.WindowTitle='Frontend'; npm run dev"

Start-Sleep -Seconds 4
Start-Process "http://127.0.0.1:5173"
Write-Host "Faceswap Studio : http://127.0.0.1:5173" -ForegroundColor Green
