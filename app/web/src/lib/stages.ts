import type { Job } from "./api";

export type StageKey = "cut" | "wake" | "queue" | "gpu" | "swap" | "pose" | "mask" | "generate" | "assemble" | "fix" | "review";

/**
 * Étape d'un rendu. Au niveau 4, chaque personne est un passage sur le Space : « generate@2/2 » = génération de la
 * 2e personne sur 2 (sans suffixe : une seule personne).
 */
export function parseStage(stage: Job["stage"]): { key: StageKey | null; pass: number; passes: number } {
  if (!stage) return { key: null, pass: 1, passes: 1 };
  const [key, rest] = stage.split("@");
  const [pass, passes] = rest ? rest.split("/").map(Number) : [1, 1];
  return { key: key as StageKey, pass: pass || 1, passes: passes || 1 };
}
