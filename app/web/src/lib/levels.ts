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
      "Cheveux longs ou volumineux, chapeaux, micro devant la bouche : résultat moins propre.",
      "Profils marqués et mouvements rapides plus difficiles.",
      "Le décor caché par la tête d'origine est reconstruit et peut scintiller.",
      "Lent sur CPU (≈ 5 s par image).",
    ],
    experimental: true,
  },
  {
    id: "character",
    step: 4,
    title: "Personne entière",
    tagline: "Toi, avec tes habits, à sa place",
    changes: ["Tête, corps et habits de ta photo en pied", "Mouvements et expressions du clip conservés"],
    alerts: [
      "Nécessite l'option GPU (ZeroGPU) : impossible sur ce PC.",
      "Sortie en 360p ou 480p, moins nette que l'original.",
      "10 s maximum, une seule personne à la fois.",
      "Mains, objets tenus et décor autour de la personne légèrement régénérés.",
      "Il faut une photo en pied, habits visibles.",
      "Quota gratuit ≈ 1 clip de 5 s par jour : PRO conseillé pour un usage régulier.",
    ],
    gpuOnly: true,
  },
];

export const levelInfo = (id: Level): LevelInfo => LEVELS.find((l) => l.id === id) ?? LEVELS[0];
