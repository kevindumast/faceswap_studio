import type { Level } from "./api";

export type LevelInfo = {
  id: Level;
  step: number;
  title: string;
  tagline: string;
  changes: string[];
  alerts: string[];
  experimental?: boolean;
  /** Seul le GPU peut le faire tourner (case GPU obligatoire au rendu). */
  gpuOnly?: boolean;
  /** Tourne seulement sur ce PC (carte graphique locale) : pas d'option ZeroGPU. */
  localOnly?: boolean;
};

export const LEVELS: LevelInfo[] = [
  {
    id: "face",
    step: 1,
    title: "Visage",
    tagline: "Tes traits sur son visage",
    changes: ["Yeux, nez, bouche, forme du visage"],
    alerts: ["Le teint, les cheveux et la forme de la tête restent ceux de la personne du clip."],
  },
  {
    id: "face_tone",
    step: 2,
    title: "Visage + teint",
    tagline: "Tes traits et ta couleur de peau",
    changes: ["Traits du visage", "Teint du visage, des oreilles et du cou"],
    alerts: [
      "Cheveux et forme de la tête inchangés.",
      "Mains et bras gardent leur couleur d'origine : l'écart se voit si les mains sont à l'image.",
      "De profil, quelques mèches collées au visage peuvent prendre la teinte.",
    ],
  },
  {
    id: "head",
    step: 3,
    title: "Tête complète",
    tagline: "Ta tête entière : cheveux, forme, teint",
    changes: ["Tête entière animée avec ses mouvements et expressions", "Cheveux, forme du crâne, teint"],
    alerts: [
      "Expérimental : raccord visible possible au niveau du cou.",
      "Il faut une photo de toi de face, en gros plan : la tête vient de cette seule photo (choix à l'étape Visages).",
      "Casquette, casque, cheveux longs de la personne du clip : effacés au mieux, restes possibles autour de ta tête.",
      "Décor reconstruit proprement sur un plan fixe ; si la caméra bouge, il est plus approximatif et peut scintiller.",
      "Profils marqués et mouvements rapides plus difficiles.",
      "Lent : sur le processeur de ce PC (≈ 8 s par image, ≈ 40 min pour 10 s), pas sur ZeroGPU.",
    ],
    experimental: true,
    localOnly: true,
  },
  {
    id: "character",
    step: 4,
    title: "Personne entière",
    tagline: "Toi, avec tes habits, à sa place",
    changes: ["Tête, corps et habits de ta photo en pied", "Mouvements et expressions du clip conservés"],
    alerts: [
      "Nécessite ton 2e Space ZeroGPU (Wan2.2-Animate) : impossible sur ce PC.",
      "Personne générée en 360p ou 480p puis recollée : la zone autour d'elle est moins nette que le reste de l'image.",
      "10 s maximum, 2 personnes au plus : un passage GPU par personne (2 personnes ≈ 2× plus de quota).",
      "Mains, objets tenus et décor autour de la personne régénérés : petits défauts possibles.",
      "Il faut une photo en pied, habits visibles (sinon le corps est inventé).",
      "≈ 2 min de GPU pour 5 s en 360p : 2 clips par jour environ avec le quota gratuit (5 min), bien plus en PRO.",
    ],
    gpuOnly: true,
  },
];

export const levelInfo = (id: Level): LevelInfo => LEVELS.find((l) => l.id === id) ?? LEVELS[0];
