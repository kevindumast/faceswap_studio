import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { PersonStanding, UserRound } from "lucide-react";
import { api, type Person, type ReferenceLevel } from "../lib/api";
import { FRAMING_TEXT } from "./PhotoViewer";
import { cx } from "./ui";

/** Cadrage des photos d'une personne + photos retenues pour les niveaux 3 et 4 (même cache partout : bibliothèque, Visages, Rendu). */
export function useFraming(person: Person, enabled = true) {
  return useQuery({
    queryKey: ["framing", person.id, person.photos.map((ph) => ph.id).join(",")],
    queryFn: () => api.framing(person.id),
    enabled: enabled && person.photos.length > 0,
    staleTime: Infinity,
  });
}

export { referenceOf } from "./PhotoViewer";

/** Choisit la photo du niveau 4, ou de la tête au niveau 3 (null : retour au choix automatique). */
export function useChooseReference(pid: string, level: ReferenceLevel = "character") {
  const qc = useQueryClient();
  return useMutation({
    mutationFn: (photoId: string | null) => api.setReference(pid, photoId, level),
    onSuccess: () => qc.invalidateQueries({ predicate: (q) => q.queryKey[0] === "framing" && q.queryKey[1] === pid }),
  });
}

/** Pastille « 3 » ou « 4 » sur la vignette de la photo retenue pour ce niveau. */
export function ReferenceBadge({ manual, level = "character" }: { manual: boolean; level?: ReferenceLevel }) {
  const head = level === "head";
  const title = head
    ? manual ? "Photo choisie pour la tête (niveau 3)" : "Photo retenue automatiquement pour la tête (niveau 3 : la plus de face)"
    : manual ? "Photo choisie pour le niveau 4" : "Photo retenue automatiquement pour le niveau 4 (la plus en pied)";
  return (
    <span
      title={title}
      className="pointer-events-none absolute bottom-1 left-1 flex h-5 items-center gap-0.5 rounded-full bg-accent px-1.5 text-[10px] font-semibold text-accent-ink shadow"
    >
      {head ? <UserRound className="size-3" /> : <PersonStanding className="size-3" />}
      {head ? 3 : 4}
    </span>
  );
}

/** Récapitulatif du rendu (niveau 4) : la photo entière qui partira au Space, avec son cadrage. */
export function ReferenceThumb({ person }: { person: Person }) {
  const framing = useFraming(person);
  const ref = person.photos.find((ph) => ph.id === framing.data?.reference);
  const info = ref ? framing.data?.photos[ref.id] : undefined;
  if (!ref) return null;
  const text = info ? FRAMING_TEXT[info.framing] : undefined;
  return (
    <span className="ml-auto flex items-center gap-2" title="Photo envoyée au Space pour ce niveau (modifiable à l'étape Visages : clic sur une photo)">
      <img src={ref.full_url} alt="" className="h-10 w-8 rounded-md bg-black object-cover ring-1 ring-line" />
      {text && (
        <span className={cx("text-[12px]", text.tone === "ok" ? "text-accent" : text.tone === "warn" ? "text-warn" : "text-muted")}>{text.label}</span>
      )}
    </span>
  );
}
