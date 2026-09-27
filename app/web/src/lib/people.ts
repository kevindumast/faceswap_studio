import type { Box, Mapping, PassageScan, Person } from "./api";

// Une couleur et une lettre par personne de la vidéo (A, B, C… dans l'ordre où elles ont été ajoutées),
// reprises partout : photos, cadres sur la vidéo, associations, récapitulatif.
const COLORS = ["#c8ff3d", "#4de1ff", "#ff7ad9", "#b69bff", "#ffd166", "#7dffb3"];
const NEUTRAL = "#9a9aa6";

export type PersonStyle = { color: string; letter: string; index: number };

export function styleAt(index: number): PersonStyle {
  if (index < 0) return { color: NEUTRAL, letter: "–", index };
  return { color: COLORS[index % COLORS.length], letter: String.fromCharCode(65 + (index % 26)), index };
}

/** Style d'une personne (par son id) selon sa place dans la vidéo. */
export function styleOf(persons: Person[], pid: string | null | undefined): PersonStyle {
  return styleAt(pid ? persons.findIndex((p) => p.id === pid) : -1);
}

export function sameBox(a: Box, b: Box): boolean {
  return a.every((v, i) => Math.abs(v - b[i]) < 1e-4);
}

/** Index de la personne du passage correspondant à une association (ou -1). */
export function faceIndex(scan: PassageScan, m: Mapping): number {
  return scan.faces.findIndex((f) => Math.abs(f.t - m.t) < 0.01 && sameBox(f.box, m.box));
}

/**
 * Réconcilie les associations avec les personnes trouvées dans le passage et celles choisies pour la vidéo :
 * - garde les choix encore valides (y compris « ne pas remplacer ») ;
 * - donne chaque personne pas encore utilisée au prochain visage libre, du plus présent au moins présent,
 *   en sautant les visages signalés comme doublon probable (profil flou d'un visage déjà listé).
 */
export function reconcile(mappings: Mapping[], scan: PassageScan, persons: Person[]): Mapping[] {
  const known = new Set(persons.map((p) => p.id));
  const kept: Mapping[] = [];
  const takenFaces = new Set<number>();
  for (const m of mappings) {
    const i = faceIndex(scan, m);
    if (i < 0 || takenFaces.has(i) || (m.person !== null && !known.has(m.person))) continue;
    kept.push(m);
    takenFaces.add(i);
  }
  const used = new Set(kept.map((m) => m.person));
  const free = persons.filter((p) => !used.has(p.id));
  scan.faces.forEach((f, i) => {
    if (takenFaces.has(i) || f.maybe_same !== null || !free.length) return;
    kept.push({ t: f.t, box: f.box, person: free.shift()!.id });
  });
  return kept;
}

export function mappingsEqual(a: Mapping[], b: Mapping[]): boolean {
  return a.length === b.length && a.every((m, i) => m.person === b[i].person && Math.abs(m.t - b[i].t) < 1e-6 && sameBox(m.box, b[i].box));
}
