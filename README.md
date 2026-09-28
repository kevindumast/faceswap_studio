# Faceswap Studio

Appli locale de face swap vidéo (usage perso / parodie, étiquetée IA) : vidéo YouTube ou fichier → choix d'un passage de 5 à 60 s → photos du visage source → rendu.

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

Pas à pas avec dépannage : voir [DEMARRAGE.md](DEMARRAGE.md).

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

## Moteur de calcul

`device` dans `config.yaml` : `auto` (recommandé) prend la carte graphique si elle est installée, sinon le CPU.
Même code partout ; seul le paquet onnxruntime change selon la machine :

| Machine | Moteur | Paquet |
|---|---|---|
| Windows, toute carte graphique | DirectML (`dml`) | `onnxruntime-directml` → `scripts\activer_gpu.ps1` |
| Serveur / Space avec carte NVIDIA | CUDA (`cuda`) | `onnxruntime-gpu[cuda,cudnn]` |
| Sans carte graphique | CPU (`cpu`) | `onnxruntime` |

Mesuré sur ce PC (i7-1185G7, MX450 2 Go + Iris Xe), rendu de 5 s : CPU ≈ 4 min 20, DirectML réparti ≈ 49 s (niveau 1).
`dml_placement: split` garde la MX450 pour le swap seul : ses 2 Go saturent si tous les modèles y sont chargés.
Un moteur demandé explicitement mais absent donne une erreur claire, jamais un repli silencieux sur le CPU.

## Niveaux de transformation

Choisis à l'étape « Visages ». Tout tourne sur ton CPU ; le GPU (ZeroGPU) est une option à cocher à chaque rendu.

| Niveau | Remplace | Moteur | Statut |
|---|---|---|---|
| 1. Visage | traits du visage (inswapper) | CPU | ✓ |
| 2. Visage + teint | + teint du visage, des oreilles et du cou (BiSeNet + transfert LAB) | CPU | ✓ (`--level tone`, 94 Mo, installable depuis l'UI) |
| 3. Tête complète | tête entière : cheveux, forme, teint (LivePortrait + LaMa) | CPU (~8 s/image) | expérimental (`--level head`, ~520 Mo, installable depuis l'UI) |
| 4. Personne entière | tête + corps + habits d'une photo en pied (Wan2.2-Animate) | GPU uniquement | ✓ |

Niveau 1 : le visage généré (128 px) est recollé avec un bord adouci, et sa couleur est ramenée sur celle du visage
d'origine (écart lissé d'une image à l'autre) : plus de visage grisé qui clignote de profil.

Niveau 2 : la zone recolorée combine la segmentation, une zone autour du visage (ellipse + cou) et un filtre couleur
calé sur le cœur du visage, pour ne jamais teinter la chemise ou le décor. Yeux, bouche, lunettes, cheveux et bijoux
ne sont jamais touchés. Les mains et les bras gardent leur couleur d'origine. Le teint de la cible est mesuré une fois
sur ~16 images réparties dans l'extrait, et la correction est un décalage de couleur constant (compensé de la couleur de
l'éclairage des photos et de la vidéo) : elle ne varie plus d'une image à l'autre. Les ombres sont protégées et le
masque de peau garde la mémoire de l'image précédente. Réglages : `levels.face_tone` dans `config.yaml`.

Niveau 3 : LivePortrait anime la tête d'une de tes photos (la plus de face, ou celle choisie à l'étape Visages) avec la
pose et l'expression de la personne du clip. Ses cheveux, sa casquette et son casque sont effacés : le décor caché est
reconstruit une fois par extrait à partir des autres images (plan fixe), LaMa (CPU) comblant ce qui n'est jamais
visible. Le cou garde sa forme et prend ton teint (calcul du niveau 2). LivePortrait tourne sur le processeur
(~8 s par image) : sur une MX450, son générateur fait décrocher le pilote NVIDIA (délai de 2 s de Windows dépassé,
écran noir). `levels.head.device: gpu` le remet sur la carte (~3× plus rapide) une fois ce délai relevé (`TdrDelay`).
Réglages : `levels.head` dans `config.yaml`.

Comparer des réglages sur les mêmes images : `scripts/bench_level.py` (vidéo côte à côte, planche de visages zoomés,
mesures de saut de teint et de gris-bleu).

## Plusieurs personnes

À l'étape « Visages », les photos importées sont **rangées automatiquement par personne** (A, B, C…) en comparant
les visages (similarité ArcFace ≥ 0,35 = même personne). Chaque personne est ensuite associée à un visage du clip
(A → le plus grand, B → le suivant…). Tout se corrige à la main : déplacer une photo vers une autre personne,
choisir la personne d'un visage (choisir une personne déjà prise échange les deux), « Inverser », ou « Ne pas remplacer ».
Chaque visage remplacé en plus ajoute ~75 % de temps de calcul.

## Vérifier avant l'assemblage

Option « Vérifier les images avant l'assemblage » (cochée par défaut, rendus sur ce PC) : pendant le rendu, chaque
image note ses visages et qui les a remplacés (`plan.json` dans le dossier du rendu). À la fin, les visages sont
suivis d'une image à l'autre dans chaque plan (jamais à travers un changement de plan) et trois cas sont signalés :

- **visage perdu par moments** : remplacé, puis d'origine quelques images (profil, flou, main devant) ;
- **change de personne** : remplacé tantôt par une personne, tantôt par une autre ;
- **visage non remplacé** : jamais remplacé mais ressemblant à une cible (les petits visages et figurants sont rangés à part).

S'il y a quelque chose d'important, le rendu s'arrête **avant le son** sur l'écran de vérification : lecteur image par
image avec les cadres, frise par personne, et pour chaque visage : remplacer par une personne sur toute la séquence,
ne rien remplacer, ou laisser tel quel. Seules les images concernées sont recalculées (sur un trou de détection, le
visage est recherché à la position attendue, sinon ses points clés sont interpolés), puis « Assembler » ajoute le son
et l'étiquette. Un rendu terminé se rouvre avec « Revoir les images ». Réglages : `review` dans `config.yaml`.

Pas encore avec l'option GPU (ZeroGPU) ni au niveau 4 : le détail image par image reste sur le Space.

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
