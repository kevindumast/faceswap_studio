import { AnimatePresence, motion } from "motion/react";
import { ChevronLeft, ChevronRight, PersonStanding, X } from "lucide-react";
import { useEffect } from "react";
import type { Framing, PersonFraming, Photo } from "../lib/api";
import { cx } from "./ui";

export const FRAMING_TEXT: Record<Framing, { label: string; hint: string; tone: "ok" | "warn" | "muted" }> = {
  full: { label: "De la tête aux pieds", hint: "Idéal pour le niveau 4 : corps et habits repris de cette photo.", tone: "ok" },
  half: { label: "À mi-corps", hint: "Au niveau 4, le bas du corps (pantalon, chaussures) sera inventé.", tone: "warn" },
  portrait: { label: "Portrait", hint: "Au niveau 4, le corps et les habits seront inventés.", tone: "muted" },
  none: { label: "Visage non détecté", hint: "Cette photo ne peut pas servir de référence.", tone: "muted" },
};

/** Photo en grand (version 1024 px, non recadrée), avec son cadrage pour le niveau 4. ← → pour naviguer, Échap pour fermer. */
export function PhotoViewer({
  photos,
  index,
  framing,
  framingError,
  onIndex,
  onClose,
}: {
  photos: Photo[];
  index: number | null;
  framing?: PersonFraming;
  framingError?: string;
  onIndex: (i: number) => void;
  onClose: () => void;
}) {
  const photo = index != null ? photos[index] : undefined;
  const info = photo ? framing?.photos[photo.id] : undefined;
  const text = info ? FRAMING_TEXT[info.framing] : undefined;
  const isReference = !!photo && framing?.reference === photo.id;

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
                  <PersonStanding className="size-3.5" /> {text.label}
                </span>
                <span className="text-white/70">{text.hint}</span>
                {isReference && <span className="rounded-full bg-white/10 px-2.5 py-1 text-white/90">Photo utilisée au niveau 4</span>}
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
