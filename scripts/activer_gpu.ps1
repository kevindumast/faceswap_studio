# Passe le moteur de calcul local sur la carte graphique (DirectML), ou revient au CPU.
#
#   .\scripts\activer_gpu.ps1        → carte graphique (onnxruntime-directml)
#   .\scripts\activer_gpu.ps1 -Cpu   → retour au CPU (onnxruntime)
#
# Avec « device: auto » dans config.yaml, l'appli utilise automatiquement ce qui est installé : rien d'autre à régler.
# À relancer après chaque « uv pip install -e . » : insightface réinstalle d'office la version CPU d'onnxruntime.
param([switch]$Cpu)

$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"

if (-not (Test-Path $py)) {
    Write-Host "Environnement Python absent (.venv). Voir README.md, section Installation." -ForegroundColor Yellow
    exit 1
}
if (-not (Get-Command uv -ErrorAction SilentlyContinue)) {
    Write-Host "uv est requis : https://docs.astral.sh/uv/" -ForegroundColor Yellow
    exit 1
}

# Windows verrouille les fichiers d'onnxruntime tant que l'API ou le worker tournent : un remplacement à moitié fait
# casserait l'installation. On refuse donc proprement.
$running = Get-CimInstance Win32_Process -Filter "Name='python.exe'" |
    Where-Object { $_.CommandLine -match 'app\.(api|worker)\.main' }
if ($running) {
    Write-Host "L'API ou le worker tourne encore : arrête-les (Ctrl+C dans leurs terminaux), puis relance ce script." -ForegroundColor Yellow
    exit 1
}

# Les deux paquets fournissent le même module « onnxruntime » : il faut retirer l'un avant d'installer l'autre.
if ($Cpu) {
    Write-Host "Retour au CPU..." -ForegroundColor Cyan
    uv pip uninstall --python $py onnxruntime-directml
    uv pip install --python $py --reinstall onnxruntime
} else {
    Write-Host "Passage sur la carte graphique (DirectML)..." -ForegroundColor Cyan
    uv pip uninstall --python $py onnxruntime
    uv pip install --python $py --reinstall onnxruntime-directml
}
if ($LASTEXITCODE -ne 0) { Write-Host "L'installation a échoué." -ForegroundColor Red; exit 1 }

& $py -c "import onnxruntime as o; p = o.get_available_providers(); print('onnxruntime', o.__version__, '| moteurs :', ', '.join(p)); print('=> carte graphique active' if 'DmlExecutionProvider' in p else '=> CPU')"
Write-Host "Relance l'API et le worker (voir DEMARRAGE.md)." -ForegroundColor Green
