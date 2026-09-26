# Faceswap Studio

Appli locale de face swap vidéo (usage perso / parodie, étiquetée IA) : vidéo YouTube ou fichier → choix d'un passage de 5 à 30 s → photos du visage source → rendu.

> Modèles InsightFace (`inswapper_128`, `buffalo_l`) : **usage non commercial uniquement**. L'API n'écoute que sur `127.0.0.1`.

## Installation (une fois)

```powershell
winget install Gyan.FFmpeg                 # ffmpeg + ffprobe
uv venv -p 3.11 .venv
uv pip install --python .venv\Scripts\python.exe -e ".[dev]"
.venv\Scripts\python.exe scripts\download_models.py   # ~800 Mo dans models/
cd app\web; npm install; cd ..\..
```

## Lancer

Tout d'un coup (3 fenêtres + navigateur) :

```powershell
.\scripts\dev.ps1
```

Ou à la main, dans 3 terminaux depuis la racine :

```powershell
.venv\Scripts\python.exe -m app.api.main      # API    → http://127.0.0.1:8000
.venv\Scripts\python.exe -m app.worker.main   # worker de rendu
cd app\web; npm run dev                        # front  → http://127.0.0.1:5173
```

Version « un seul serveur » : `cd app\web; npm run build`, puis seule l'API suffit (elle sert le front sur http://127.0.0.1:8000).

## Plusieurs personnes

À l'étape « Visages », les photos importées sont **rangées automatiquement par personne** (A, B, C…) en comparant
les visages (similarité ArcFace ≥ 0,35 = même personne). Chaque personne est ensuite associée à un visage du clip
(A → le plus grand, B → le suivant…). Tout se corrige à la main : déplacer une photo vers une autre personne,
choisir la personne d'un visage (choisir une personne déjà prise échange les deux), « Inverser », ou « Ne pas remplacer ».
Chaque visage remplacé en plus ajoute ~75 % de temps de calcul.

## En ligne de commande (sans l'appli)

```powershell
.venv\Scripts\python.exe -m src.pipeline --video clip.mp4 --start 12 --end 20 --faces data\source_faces --out out.mp4 [--full] [--no-label]
```

## Architecture

- `src/` : pipeline indépendant (ffmpeg, yt-dlp, détection + ArcFace, inswapper, lissage, assemblage).
- `app/api/` : FastAPI (vidéos, visages, jobs). `app/worker/` : process de rendu. Les deux partagent `data/app.db` (SQLite, file de jobs + progression).
- `app/web/` : React + Vite + Tailwind.
- `config.yaml` : device (`cpu` / `cuda`), limites, seuils. Même code en local et sur GPU cloud.

Sur CPU (i7 4 cœurs) : ~2,5 s par image, soit ~5 min pour 5 s de vidéo à 25 i/s. Pour les rendus longs, lancer le worker sur une machine GPU (`device: cuda` + `onnxruntime-gpu`).
