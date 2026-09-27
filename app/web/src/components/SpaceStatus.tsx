import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AlarmClock, RotateCcw } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type SpaceKind, type SpaceState } from "../lib/api";
import { duration } from "../lib/time";
import { Button, cx } from "./ui";

/** État en direct d'un Space (métadonnées Hugging Face, sans quota) : toutes les 5 s pendant un réveil, sinon 30 s. */
export function useSpaceState(kind: SpaceKind, enabled: boolean) {
  return useQuery({
    queryKey: ["space", kind],
    queryFn: () => api.spaceState(kind),
    enabled,
    refetchInterval: (q) => (q.state.data?.phase === "starting" ? 5000 : 30000),
    retry: false,
  });
}

/** Secondes écoulées depuis `since` (horodatage en secondes), rafraîchies chaque seconde. */
export function useElapsed(since: number | null | undefined): number | null {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    if (!since) return;
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, [since]);
  return since ? Math.max(0, now / 1000 - since) : null;
}

/** Chrono « m:ss » (ou « h:mm:ss »). */
export function chrono(seconds: number): string {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const ss = String(s % 60).padStart(2, "0");
  return h ? `${h}:${String(m).padStart(2, "0")}:${ss}` : `${m}:${ss}`;
}

export const PHASE_LABEL: Record<SpaceState["phase"], string> = {
  ready: "Prêt",
  starting: "Réveil en cours",
  asleep: "Endormi",
  error: "En erreur",
  unknown: "État inconnu",
  unconfigured: "Non branché",
};

export function phaseTone(phase: SpaceState["phase"] | undefined): "ok" | "warn" | "danger" | "muted" {
  if (phase === "ready") return "ok";
  if (phase === "starting") return "warn";
  if (phase === "error") return "danger";
  return "muted";
}

const DOT = { ok: "bg-accent shadow-[0_0_6px_var(--color-accent)]", warn: "bg-warn animate-pulse", danger: "bg-danger", muted: "bg-faint" };

/** Ligne d'état d'un Space dans le panneau Moteur : état, chrono du réveil, bouton Réveiller / Redémarrer. */
export function SpaceLive({ kind }: { kind: SpaceKind }) {
  const qc = useQueryClient();
  const state = useSpaceState(kind, true);
  const s = state.data;
  const elapsed = useElapsed(s?.phase === "starting" ? s.waking_since : null);
  const wake = useMutation({
    mutationFn: () => api.wakeSpace(kind),
    onSuccess: (data) => qc.setQueryData(["space", kind], data),
  });
  const tone = phaseTone(s?.phase);

  return (
    <div className="rounded-xl bg-raised p-3 ring-1 ring-line">
      <div className="flex flex-wrap items-center gap-2 text-[13px]">
        <span className={cx("size-2 rounded-full", DOT[tone])} />
        <span className="font-medium">{s ? PHASE_LABEL[s.phase] : state.isLoading ? "Lecture de l'état…" : "État indisponible"}</span>
        {s?.phase === "starting" && elapsed != null && (
          <span className="font-mono text-muted tabular">
            {chrono(elapsed)} <span className="text-faint">/ ≈ {duration(s.expected_s)}</span>
          </span>
        )}
        {s?.phase === "starting" && s.stage === "BUILDING" && <span className="text-muted">(installation)</span>}
        <span className="ml-auto">
          {(s?.phase === "asleep" || s?.phase === "error") && (
            <Button
              variant="secondary"
              size="sm"
              loading={wake.isPending}
              onClick={() => wake.mutate()}
              icon={s.phase === "error" ? <RotateCcw className="size-3.5" /> : <AlarmClock className="size-3.5" />}
            >
              {s.phase === "error" ? "Redémarrer" : "Réveiller"}
            </Button>
          )}
        </span>
      </div>
      {s?.phase === "starting" && elapsed != null && (
        <div className="mt-2 h-1 overflow-hidden rounded-full bg-overlay">
          <div className="h-full rounded-full bg-warn transition-[width] duration-1000" style={{ width: `${Math.min(95, (elapsed / s.expected_s) * 100)}%` }} />
        </div>
      )}
      <p className="mt-1.5 text-[12px] text-muted">
        {s?.phase === "asleep" && "Il se réveille tout seul au prochain rendu ; « Réveiller » le prépare en avance, sans consommer de quota."}
        {s?.phase === "starting" && "Le Space démarre (chargement du modèle). Un rendu lancé maintenant attendra la fin du réveil."}
        {s?.phase === "ready" && "Prêt à calculer : un rendu démarre sans attente de réveil."}
        {s?.phase === "error" && (s.error ? `Hugging Face : ${s.error}` : "Le Space a planté : « Redémarrer », ou relance deploy_space.py.")}
        {state.error && state.error.message}
        {wake.error && ` ${wake.error.message}`}
      </p>
    </div>
  );
}
