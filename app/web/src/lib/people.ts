import type { Box, FramesFaces, Mapping, Person } from "./api";

// Une couleur stable par personne (A, B, C…), reprise partout : photos, boîtes sur la vidéo, récapitulatif.
const COLORS = ["#c8ff3d", "#4de1ff", "#ff7ad9", "#b69bff", "#ffd166", "#7dffb3"];

export function personColor(id: string | null | undefined): string {
  if (!id) return "#9a9aa6";
  return COLORS[(id.charCodeAt(0) - 65 + COLORS.length) % COLORS.length];
}

export function sameBox(a: Box, b: Box): boolean {
  return a.every((v, i) => Math.abs(v - b[i]) < 1e-4);
}

/** Index du visage détecté correspondant à une association (ou -1). */
export function faceIndex(data: FramesFaces, m: Mapping): number {
  if (Math.abs(m.t - data.t) > 0.01) return -1;
  return data.faces.findIndex((f) => sameBox(f.box, m.box));
}

/**
 * Réconcilie les associations avec les visages de l'image affichée et les personnes connues :
 * - garde les choix encore valides (y compris « ne pas remplacer ») ;
 * - donne automatiquement chaque personne pas encore utilisée au prochain visage libre (du plus grand au plus petit).
 */
export function reconcile(mappings: Mapping[], data: FramesFaces, persons: Person[]): Mapping[] {
  mappings = carryOver(mappings, data);
  const known = new Set(persons.map((p) => p.id));
  const kept: Mapping[] = [];
  const takenFaces = new Set<number>();
  for (const m of mappings) {
    const i = faceIndex(data, m);
    if (i < 0 || takenFaces.has(i) || (m.person !== null && !known.has(m.person))) continue;
    kept.push(m);
    takenFaces.add(i);
  }
  const usedPersons = new Set(kept.map((m) => m.person));
  const free = persons.filter((p) => !usedPersons.has(p.id));
  data.faces.forEach((f, i) => {
    if (takenFaces.has(i) || !free.length) return;
    kept.push({ t: data.t, box: f.box, person: free.shift()!.id });
  });
  return kept;
}

/**
 * Associations faites sur une autre image du passage (Début / Milieu / Fin) : on les reporte sur les visages
 * de l'image affichée par proximité horizontale (dans un clip, les gens changent rarement de côté).
 */
function carryOver(mappings: Mapping[], data: FramesFaces): Mapping[] {
  if (!mappings.length || mappings.some((m) => Math.abs(m.t - data.t) <= 0.01)) return mappings;
  const cx = (b: Box) => (b[0] + b[2]) / 2;
  const free = data.faces.map((_, i) => i);
  const out: Mapping[] = [];
  for (const m of [...mappings].sort((a, b) => cx(a.box) - cx(b.box))) {
    if (!free.length) break;
    const best = free.reduce((k, i) => (Math.abs(cx(data.faces[i].box) - cx(m.box)) < Math.abs(cx(data.faces[k].box) - cx(m.box)) ? i : k));
    free.splice(free.indexOf(best), 1);
    out.push({ t: data.t, box: data.faces[best].box, person: m.person });
  }
  return out;
}

export function mappingsEqual(a: Mapping[], b: Mapping[]): boolean {
  return a.length === b.length && a.every((m, i) => m.person === b[i].person && Math.abs(m.t - b[i].t) < 1e-6 && sameBox(m.box, b[i].box));
}
