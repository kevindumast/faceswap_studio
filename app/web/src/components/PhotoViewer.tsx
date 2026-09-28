import { AnimatePresence, motion } from "motion/react";
import { ChevronLeft, ChevronRight, PersonStanding, UserRound, X } from "lucide-react";
import { useEffect } from "react";
import type { Framing, PersonFraming, Photo, ReferenceLevel } from "../lib/api";
import { cx } from "./ui";

type Advice = { label: string; hint: string; tone: "ok" | "warn" | "muted" };

export const FRAMING_TEXT: Record<Framing, Advice> = {
  full: { label: "De la tête aux pieds", hint: "Idéal pour le niveau 4 : corps et habits repris de cette photo.", tone: "ok" },
  half: { label: "À mi-corps", hint: "Au niveau 4, le bas du corps (pantalon, chaussures) sera inventé.", tone: "warn" },
  portrait: { label: "Portrait", hint: "Au niveau 4, le corps et les habits seront inventés.", tone: "muted" },
  none: { label: "Visage non détecté", hint: "Cette photo ne peut pas servir de référence.", tone: "muted" },
};

/** Qualité d'une photo pour la tête du niveau 3 (note : visage grand et de face). */
export function headAdvice(score: number | null | undefined): Advice {
  if (score == null) return { label: "Visage non détecté", hint: "Cette photo ne peut pas servir pour la tête.", tone: "muted" };
  if (score >= 0.5) return { label: "Visage de face, en gros plan", hint: "Idéal pour le niveau 3 : tête, cheveux et teint repris de cette photo.", tone: "ok" };
  if (score >= 0.2) return { label: "Visage petit ou un peu de biais", hint: "Au niveau 3, la tête sera moins nette ou moins fidèle.", tone: "warn" };
  return { label: "Visage trop petit ou de profil", hint: "Au niveau 3, préfère une photo de face en gros plan.", tone: "muted" };
}

/** Photo retenue pour ce niveau, et si elle a été choisie à la main. */
export function referenceOf(framing: PersonFraming | undefined, level: ReferenceLevel): { id: string | null; manual: boolean } {
  if (!framing) return { id: null, manual: false };
  return level === "head"
    ? { id: framing.head_reference, manual: framing.head_manual }
    : { id: framing.reference, manual: framing.manual };
}

/** Photo en grand (version 1024 px, non recadrée), avec son intérêt pour le niveau 4 (cadrage) ou pour la tête du
 *  niveau 3. ← → pour naviguer, Échap pour fermer. */
export function PhotoViewer({
  photos,
  index,
  framing,
  framingError,
  onIndex,
  onClose,
  onChooseReference,
  choosing,
  referenceLevel = "character",
}: {
  photos: Photo[];
  index: number | null;
  framing?: PersonFraming;
  framingError?: string;
  /** Choix de la photo de ce niveau (null : retour au choix automatique). Sans ce rappel : simple visionneuse. */
  onChooseReference?: (photoId: string | null) => void;
  choosing?: boolean;
  onIndex: (i: number) => void;
  onClose: () => void;
  /** Niveau dont on regarde / choisit la photo : tête (3) ou personne entière (4). */
  referenceLevel?: ReferenceLevel;
}) {
  const photo = index != null ? photos[index] : undefined;
  const info = photo ? framing?.photos[photo.id] : undefined;
  const head = referenceLevel === "head";
  const text = info ? (head ? headAdvice(info.head_score) : FRAMING_TEXT[info.framing]) : undefined;
  const ref = referenceOf(framing, referenceLevel);
  const isReference = !!photo && ref.id === photo.id;
  const forLevel = head ? "pour la tête (niveau 3)" : "pour le niveau 4";

  useEffect(() => {
    if (index == null) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === "Escape") {
        e.stopPropagation(); // ne ferme pas aussi le tiroir de la bibliothèque
        onClose();
      }
      if (e.key === "ArrowRight" && index < photos.length - 1) onIndex(index + 1);
      if (e.key === "ArrowLeft" && index > 0) onIndex(index - 1);
    };
    window.addEventListener("keydown", onKey, true);
    return () => window.removeEventListener("keydown", onKey, true);
  }, [index, photos.length, onIndex, onClose]);

  const go = (i: number) => (e: React.MouseEvent) => {
    e.stopPropagation();
    onIndex(i);
  };

  return (
    <AnimatePresence>
      {photo && index != null && (
        <motion.div
          role="dialog"
          aria-label={`Photo ${photo.name}`}
          className="fixed inset-0 z-[60] flex flex-col bg-black/85 backdrop-blur-sm"
          initial={{ opacity: 0 }}
          animate={{ opacity: 1 }}
          exit={{ opacity: 0 }}
          onClick={onClose}
        >
          <header className="flex items-center gap-3 px-4 py-3 text-[13px] text-white/80" onClick={(e) => e.stopPropagation()}>
            <span className="truncate">{photo.name}</span>
            <span className="font-mono text-white/50 tabular">
              {index + 1}/{photos.length}
            </span>
            <button onClick={onClose} aria-label="Fermer" className="ml-auto flex size-9 items-center justify-center rounded-full hover:bg-white/10">
              <X className="size-5" />
            </button>
          </header>
          <div className="relative flex min-h-0 flex-1 items-center justify-center px-14">
            <img
              key={photo.id}
              src={photo.full_url}
              alt={photo.name}
              className="max-h-full max-w-full rounded-lg object-contain shadow-2xl"
              onClick={(e) => e.stopPropagation()}
              onError={(e) => {
                // Version 1024 px indisponible (API pas encore relancée, ancienne photo) : version 480 px.
                const img = e.currentTarget;
                if (!img.dataset.fallback) {
                  img.dataset.fallback = "1";
                  img.src = photo.photo_url;
                }
              }}
            />
            {index > 0 && (
              <button onClick={go(index - 1)} aria-label="Photo précédente" className="absolute left-3 flex size-10 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20">
                <ChevronLeft className="size-5" />
              </button>
            )}
            {index < photos.length - 1 && (
              <button onClick={go(index + 1)} aria-label="Photo suivante" className="absolute right-3 flex size-10 items-center justify-center rounded-full bg-white/10 text-white hover:bg-white/20">
                <ChevronRight className="size-5" />
              </button>
            )}
          </div>
          <footer className="flex flex-wrap items-center justify-center gap-2 px-4 py-4 text-[13px]" onClick={(e) => e.stopPropagation()}>
            {text ? (
              <>
                <span
                  className={cx(
                    "inline-flex items-center gap-1.5 rounded-full px-2.5 py-1 font-medium",
                    text.tone === "ok" && "bg-accent-soft text-accent",
                    text.tone === "warn" && "bg-warn-soft text-warn",
                    text.tone === "muted" && "bg-white/10 text-white/80",
                  )}
                >
                  {head ? <UserRound className="size-3.5" /> : <PersonStanding className="size-3.5" />} {text.label}
                </span>
                <span className="text-white/70">{text.hint}</span>
                {isReference && (
                  <span className="rounded-full bg-white/10 px-2.5 py-1 text-white/90">
                    Photo utilisée {forLevel}{ref.manual ? " (ton choix)" : " (choix automatique)"}
                  </span>
                )}
                {onChooseReference && !isReference && (
                  <button
                    onClick={() => onChooseReference(photo.id)}
                    disabled={choosing}
                    className="rounded-full bg-accent px-3 py-1 font-medium text-accent-ink hover:brightness-110 disabled:opacity-60"
                  >
                    Utiliser {forLevel}
                  </button>
                )}
                {onChooseReference && isReference && ref.manual && (
                  <button onClick={() => onChooseReference(null)} disabled={choosing} className="rounded-full px-3 py-1 text-white/70 underline-offset-2 hover:underline">
                    Revenir au choix automatique
                  </button>
                )}
              </>
            ) : framingError ? (
              <span className="text-white/50">Cadrage indisponible : {framingError}</span>
            ) : (
              <span className="text-white/50">Analyse du cadrage…</span>
            )}
          </footer>
        </motion.div>
      )}
    </AnimatePresence>
  );
}
