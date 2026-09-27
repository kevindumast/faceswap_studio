# Démarrer Faceswap Studio

Il faut **3 terminaux PowerShell, qui restent ouverts** pendant toute l'utilisation.

## 1. L'API (Python)

```powershell
cd C:\Users\kevin.dumast\Documents\behero
.venv\Scripts\python.exe -m app.api.main
```

Doit afficher : `Uvicorn running on http://127.0.0.1:8000`

## 2. Le worker (Python : calcule les rendus)

```powershell
cd C:\Users\kevin.dumast\Documents\behero
.venv\Scripts\python.exe -m app.worker.main
```

Doit afficher : `Worker prêt, en attente de jobs…`

## 3. Le front (site web)

```powershell
cd C:\Users\kevin.dumast\Documents\behero\app\web
npm run dev
```

Doit afficher : `Local: http://127.0.0.1:5173/` → ouvrir cette adresse dans le navigateur.

## Tout lancer d'un coup

```powershell
cd C:\Users\kevin.dumast\Documents\behero
.\scripts\dev.ps1
```

Ouvre les 3 fenêtres et le navigateur. Si PowerShell refuse d'exécuter le script :

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\dev.ps1
```

## Carte graphique (plus rapide)

Avec `device: auto` dans `config.yaml`, l'appli utilise **automatiquement la carte graphique** si elle est installée,
sinon le CPU. Sur ce PC (MX450 + Iris Xe) : environ **5× plus rapide** qu'en CPU, image identique.

Activer (à faire une fois, **API et worker arrêtés**) :

```powershell
cd C:\Users\kevin.dumast\Documents\behero
.\scripts\activer_gpu.ps1
```

Revenir au CPU : `.\scripts\activer_gpu.ps1 -Cpu`

La pastille en haut à droite du site indique le moteur utilisé : « Moteur prêt · Carte graphique · MX450 + Iris Xe ».
À relancer après une réinstallation des dépendances (`uv pip install -e .`), qui remet la version CPU.

## GPU distant ZeroGPU (optionnel)

Un Space privé sur Hugging Face (GPU 48 Go, 5 min/jour en gratuit, 40 en PRO). Indispensable pour le niveau 4.
Une fois branché, la case « Utiliser le GPU » apparaît à l'étape Rendu : **décochée par défaut, à cocher à chaque rendu**.

1. Crée un jeton `write` : https://huggingface.co/settings/tokens
2. Déploie ton Space privé (API et worker peuvent tourner) :

```powershell
cd C:\Users\kevin.dumast\Documents\behero
.venv\Scripts\python.exe scripts\deploy_space.py --space ton-pseudo/faceswap-gpu --save
```

3. Attends la construction (5 à 15 min la première fois), puis dans l'appli : pastille du moteur (en haut) → « Tester la connexion ».

Relancer la même commande met le Space à jour après une modification du code.

## Arrêter

`Ctrl + C` dans chaque terminal (ou fermer les fenêtres).

## Si ça ne marche pas

| Ce que le site affiche | Cause | Quoi faire |
|---|---|---|
| « API hors ligne » | le terminal 1 n'est pas lancé ou a planté | relancer la commande 1 et lire l'erreur affichée |
| « Worker arrêté » (rendus bloqués « En attente ») | le terminal 2 n'est pas lancé | relancer la commande 2 (**un seul** worker à la fois) |
| « Modèles manquants » | poids non téléchargés | `.venv\Scripts\python.exe scripts\download_models.py` |
| « ffmpeg manquant » | ffmpeg non installé | `winget install Gyan.FFmpeg` |
| Le site ne s'ouvre pas | le terminal 3 n'est pas lancé | relancer la commande 3 |

- `No module named app` : la commande n'a pas été lancée depuis le dossier `behero` (faire le `cd` d'abord).
- `Port 5173 is in use, trying another one…` : un autre front tourne déjà. Utiliser l'adresse affichée (ex. `5174`) ou fermer l'ancien terminal.
