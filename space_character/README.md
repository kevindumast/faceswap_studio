---
title: Faceswap Studio Niveau 4
emoji: 🕺
colorFrom: gray
colorTo: green
sdk: gradio
sdk_version: 6.28.0
python_version: "3.10"
app_file: app.py
pinned: false
startup_duration_timeout: 1h
short_description: Moteur privé « personne entière » de Faceswap Studio
---

Moteur **privé** du niveau 4 de Faceswap Studio (remplacer toute la personne : corps, habits, gestuelle), appelé par
l'appli locale. Déployé par `scripts/deploy_space.py --kind character` ; ne pas rendre ce Space public.

Modèle : [Wan2.2-Animate-14B](https://huggingface.co/Wan-AI/Wan2.2-Animate-14B) (Apache 2.0), code officiel
[Wan-Video/Wan2.2](https://github.com/Wan-Video/Wan2.2) téléchargé au démarrage à un commit fixé.
Premier démarrage : ~57 Go de poids à télécharger et à charger, compter 15 à 30 min.
