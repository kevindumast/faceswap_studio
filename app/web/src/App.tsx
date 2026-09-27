import { useQuery } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { History } from "lucide-react";
import { useCallback, useEffect, useState } from "react";
import { api, type Job, type Level, type Mapping, type Status, type Video } from "./lib/api";
import { Stepper } from "./components/Stepper";
import { HistoryDrawer } from "./components/HistoryDrawer";
import { Button, cx } from "./components/ui";
import type { Selection } from "./components/trimmer/Trimmer";
import { VideoStep } from "./steps/VideoStep";
import { SegmentStep } from "./steps/SegmentStep";
import { FacesStep } from "./steps/FacesStep";
import { RenderStep } from "./steps/RenderStep";

type Session = {
  step: number;
  videoId: string | null;
  selection: Selection | null;
  faceSetId: string | null;
  consent: boolean;
  mappings: Mapping[];
  level: Level;
  jobId: string | null;
};

// L'option GPU n'est volontairement pas dans la session : elle revient décochée à chaque rendu.
const EMPTY: Session = { step: 0, videoId: null, selection: null, faceSetId: null, consent: false, mappings: [], level: "face", jobId: null };
const KEY = "faceswap.session";

function loadSession(): Session {
  try {
    const raw = localStorage.getItem(KEY);
    const saved = raw ? JSON.parse(raw) : {};
    return { ...EMPTY, ...saved, mappings: Array.isArray(saved.mappings) ? saved.mappings : [] };
  } catch {
    return EMPTY;
  }
}

export default function App() {
  const [s, setS] = useState<Session>(loadSession);
  const [historyOpen, setHistoryOpen] = useState(false);
  const patch = useCallback((p: Partial<Session>) => setS((prev) => ({ ...prev, ...p })), []);

  useEffect(() => {
    try {
      localStorage.setItem(KEY, JSON.stringify(s));
    } catch {
      /* stockage indisponible : pas grave */
    }
  }, [s]);

  const status = useQuery({ queryKey: ["status"], queryFn: api.status, refetchInterval: 5000 });
  const video = useQuery({
    queryKey: ["video", s.videoId],
    queryFn: () => api.video(s.videoId!),
    enabled: !!s.videoId,
    retry: false,
  });

  // Vidéo disparue (nettoyage auto) : retour à l'étape 1.
  useEffect(() => {
    if (video.error) setS(EMPTY);
  }, [video.error]);

  const min = status.data?.segment.min_s ?? 5;
  // Niveau mémorisé mais plus proposé (ex. session d'une autre version) : retour au niveau 1.
  const level: Level = status.data && !status.data.levels[s.level]?.available ? "face" : s.level;
  const ready = video.data?.status === "ready" ? video.data : null;

  const onVideoReady = useCallback(
    (v: Video) => {
      const d = v.info?.duration ?? 0;
      patch({ videoId: v.id, selection: { start: 0, end: Math.min(10, Math.max(min, d)) }, mappings: [], jobId: null, step: 1 });
    },
    [patch, min],
  );

  // Référence stable : FacesStep réconcilie les associations dans un effet qui dépend de ce callback.
  const onMappings = useCallback((m: Mapping[]) => patch({ mappings: m }), [patch]);

  const openJob = (j: Job) => {
    patch({
      videoId: j.video_id,
      faceSetId: j.face_set_id,
      selection: { start: j.params.start, end: j.params.end },
      mappings: j.params.mappings ?? [],
      level: j.params.level ?? "face",
      jobId: j.id,
      consent: true,
      step: 3,
    });
    setHistoryOpen(false);
  };

  const reachable = (i: number) => i === 0 || (!!ready && (i < 3 || (!!s.faceSetId && s.consent && s.mappings.some((m) => m.person))));
  const step = reachable(s.step) ? s.step : 0;

  return (
    <div className="bg-studio min-h-dvh">
      <header className="sticky top-0 z-30 border-b border-line/70 bg-bg/80 backdrop-blur-xl">
        <div className="mx-auto flex h-16 max-w-6xl items-center gap-4 px-4 sm:px-6">
          <button onClick={() => patch({ step: 0 })} className="flex shrink-0 items-center gap-2.5" aria-label="Accueil">
            <span className="flex size-8 items-center justify-center rounded-[9px] bg-accent text-accent-ink">
              <svg viewBox="0 0 20 20" className="size-4" aria-hidden>
                <path d="M4 6h12M4 10h7M4 14h12" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" />
              </svg>
            </span>
            <span className="hidden font-semibold tracking-tight md:inline">
              Faceswap <span className="text-muted">Studio</span>
            </span>
          </button>
          <div className="flex flex-1 justify-center">
            <Stepper current={step} reachable={reachable} onGo={(i) => patch({ step: i })} />
          </div>
          <EngineStatus data={status.data} error={!!status.error} />
          <Button variant="ghost" size="sm" onClick={() => setHistoryOpen(true)} icon={<History className="size-4" />} aria-label="Historique">
            <span className="hidden lg:inline">Historique</span>
          </Button>
        </div>
      </header>

      <main className="mx-auto max-w-6xl px-4 pt-8 pb-20 sm:px-6 sm:pt-10">
        <AnimatePresence mode="wait">
          <motion.div
            key={step}
            initial={{ opacity: 0, y: 10 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: -6 }}
            transition={{ duration: 0.18, ease: "easeOut" }}
          >
            {step === 0 && <VideoStep status={status.data} onReady={onVideoReady} />}
            {step === 1 && ready && s.selection && (
              <SegmentStep
                video={ready}
                status={status.data}
                selection={s.selection}
                onChange={(sel) => patch({ selection: sel, jobId: null })}
                onNext={() => patch({ step: 2 })}
              />
            )}
            {step === 2 && ready && s.selection && (
              <FacesStep
                video={ready}
                status={status.data}
                selection={s.selection}
                faceSetId={s.faceSetId}
                onFaceSet={(id) => patch({ faceSetId: id })}
                consent={s.consent}
                onConsent={(v) => patch({ consent: v })}
                mappings={s.mappings}
                onMappings={onMappings}
                level={level}
                onLevel={(l) => patch({ level: l, jobId: null })}
                onNext={() => patch({ step: 3, jobId: null })}
              />
            )}
            {step === 3 && ready && s.selection && s.faceSetId && (
              <RenderStep
                video={ready}
                status={status.data}
                selection={s.selection}
                faceSetId={s.faceSetId}
                mappings={s.mappings}
                level={level}
                consent={s.consent}
                jobId={s.jobId}
                onJob={(id) => patch({ jobId: id })}
                onEditSegment={() => patch({ step: 1, jobId: null })}
                onNewVideo={() => patch({ ...EMPTY, faceSetId: s.faceSetId, consent: s.consent })}
              />
            )}
          </motion.div>
        </AnimatePresence>
      </main>

      <HistoryDrawer open={historyOpen} onClose={() => setHistoryOpen(false)} onOpenJob={openJob} />
    </div>
  );
}

function EngineStatus({ data, error }: { data?: Status; error: boolean }) {
  let tone: "ok" | "warn" | "down" = "ok";
  const engine = data?.engine;
  const gpus = engine?.gpus.length ? ` · ${engine.gpus.join(" + ")}` : "";
  let label = `Moteur prêt · ${engine?.label ?? "CPU"}${gpus}`;
  let title = "API, worker, modèles et ffmpeg opérationnels.";
  if (engine?.error) {
    tone = "warn";
    label = "Moteur mal configuré · CPU";
    title = engine.error;
  }
  if (error) {
    tone = "down";
    label = "API hors ligne";
    title = "Lance l'API : python -m app.api.main";
  } else if (data && (!data.models || !data.ffmpeg)) {
    tone = "down";
    label = !data.models ? "Modèles manquants" : "ffmpeg manquant";
    title = !data.models ? "python scripts/download_models.py" : "winget install Gyan.FFmpeg";
  } else if (data && !data.worker) {
    tone = "warn";
    label = "Worker arrêté";
    title = "Lance le worker : python -m app.worker.main (ou scripts\\dev.ps1)";
  }
  if (!data && !error) return null;
  return (
    <span
      title={title}
      className={cx(
        "hidden shrink-0 items-center gap-2 rounded-full px-3 py-1 text-[12px] font-medium ring-1 sm:flex",
        tone === "ok" && "text-muted ring-line",
        tone === "warn" && "bg-warn-soft text-warn ring-warn/30",
        tone === "down" && "bg-danger-soft text-danger ring-danger/30",
      )}
    >
      <span className={cx("size-1.5 rounded-full", tone === "ok" ? "bg-accent shadow-[0_0_6px_var(--color-accent)]" : tone === "warn" ? "bg-warn" : "bg-danger")} />
      {label}
    </span>
  );
}
