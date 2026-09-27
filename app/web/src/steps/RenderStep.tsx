import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { ArrowLeft, ArrowRight, Check, Clapperboard, Cpu, Download, Film, Loader2, RotateCcw, Scissors, Sparkles, Wand2, X, Zap } from "lucide-react";
import { useState } from "react";
import { ApiError, api, type Job, type JobParams, type Level, type Mapping, type Status, type Video } from "../lib/api";
import { levelInfo } from "../lib/levels";
import { faceIndex, styleOf } from "../lib/people";
import { cappedFps, duration, seconds, timecode } from "../lib/time";
import { Button, Card, Notice, ProgressBar, SectionTitle, SegmentedControl, Switch, cx } from "../components/ui";
import { Compare } from "../components/Compare";
import type { Selection } from "../components/trimmer/Trimmer";
import { PersonBadge } from "./FacesStep";

type Props = {
  video: Video;
  status?: Status;
  selection: Selection;
  faceSetId: string;
  mappings: Mapping[];
  level: Level;
  consent: boolean;
  jobId: string | null;
  onJob: (id: string | null) => void;
  onEditSegment: () => void;
  onNewVideo: () => void;
};

const TERMINAL = ["done", "error", "cancelled"];

export function RenderStep(p: Props) {
  const job = useQuery({
    queryKey: ["job", p.jobId],
    queryFn: () => api.job(p.jobId!),
    enabled: !!p.jobId,
    refetchInterval: (q) => (q.state.data && TERMINAL.includes(q.state.data.status) ? false : 1000),
    // Rendu supprimé (404) : on arrête tout de suite. API qui redémarre : on réessaie sans rien afficher.
    retry: (count, err) => !(err instanceof ApiError && err.status === 404) && count < 10,
    retryDelay: 1000,
  });

  if (!p.jobId) return <Setup {...p} />;
  if (job.error) {
    return (
      <Notice tone="danger">
        Ce rendu n'existe plus. <button className="underline" onClick={() => p.onJob(null)}>Revenir aux options</button>
      </Notice>
    );
  }
  if (!job.data) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="size-6 animate-spin text-muted" />
      </div>
    );
  }
  const j = job.data;
  if (j.status === "done") return <Done job={j} {...p} />;
  if (j.status === "error" || j.status === "cancelled") {
    return (
      <div>
        <SectionTitle eyebrow="Étape 4 · Rendu" title={j.status === "error" ? "Le rendu a échoué" : "Rendu annulé"} />
        {j.error && <Notice tone="danger">{j.error}</Notice>}
        <div className="mt-6 flex gap-3">
          <Button variant="primary" icon={<RotateCcw className="size-4" />} onClick={() => p.onJob(null)}>
            Revenir aux options
          </Button>
          <Button variant="secondary" icon={<Scissors className="size-4" />} onClick={p.onEditSegment}>
            Modifier le passage
          </Button>
        </div>
      </div>
    );
  }
  return <Running job={j} video={p.video} worker={p.status?.worker ?? true} />;
}

function Setup(p: Props) {
  const qc = useQueryClient();
  const [output, setOutput] = useState<JobParams["output"]>("segment");
  const [stabilize, setStabilize] = useState(true);
  const [aiLabel, setAiLabel] = useState(true);
  // Jamais coché d'office ni mémorisé : le GPU ne consomme du quota que si on le demande pour CE rendu.
  const [useGpu, setUseGpu] = useState(false);
  const info = levelInfo(p.level);
  const fps = p.video.info!.fps;
  const cap = p.status?.fps_cap ?? 30;
  const highFps = fps > cap + 0.5;
  // Coché par défaut pour les vidéos > 30 i/s : cadence standard des réseaux sociaux, rendu 2× plus rapide en 60 i/s.
  const [limitFps, setLimitFps] = useState(true);
  const renderFps = highFps && limitFps ? cappedFps(fps, cap) : fps;
  const len = p.selection.end - p.selection.start;
  const frames = Math.round(len * renderFps);
  const active = p.mappings.filter((m): m is Mapping & { person: string } => !!m.person);
  // Le swap domine le temps de calcul : chaque visage remplacé en plus coûte ~75 % d'une passe.
  const swapFactor = 1 + 0.75 * Math.max(0, active.length - 1);
  const spf = p.status?.levels[p.level]?.sec_per_frame ?? p.status?.sec_per_frame ?? 2.2;
  const gpuConfigured = !!p.status?.gpu.configured;
  const blockedByGpu = !!info.gpuOnly && !useGpu;
  // Mode « vidéo complète » : le reste du clip est seulement réencodé (≈ 4× plus vite que le temps réel).
  const fullExtra = p.video.info!.duration * 0.25;
  const estimate = frames * spf * swapFactor + 15 + (output === "full" ? fullExtra : 0);
  const faceSet = useQuery({ queryKey: ["faceset", p.faceSetId], queryFn: () => api.faceSet(p.faceSetId) });
  // Même requête que l'étape Visages : la recherche du passage est déjà en cache.
  const scan = useQuery({
    queryKey: ["scan", p.video.id, p.selection.start, p.selection.end],
    queryFn: () => api.scan(p.video.id, p.selection.start, p.selection.end),
    staleTime: Infinity,
  });
  const cropOf = (m: Mapping) => (scan.data ? scan.data.faces[faceIndex(scan.data, m)]?.crop : undefined);

  const launch = useMutation({
    mutationFn: () =>
      api.createJob({
        video_id: p.video.id,
        face_set_id: p.faceSetId,
        start: p.selection.start,
        end: p.selection.end,
        output,
        stabilize,
        ai_label: aiLabel,
        mappings: active.map(({ t, box, person }) => ({ t, box, person })),
        level: p.level,
        use_gpu: useGpu,
        limit_fps: highFps && limitFps,
        consent: p.consent,
      }),
    onSuccess: (j) => {
      qc.setQueryData(["job", j.id], j);
      qc.invalidateQueries({ queryKey: ["jobs"] });
      p.onJob(j.id);
    },
  });

  const persons = faceSet.data?.persons ?? [];

  return (
    <div>
      <SectionTitle eyebrow="Étape 4 · Rendu" title="Dernières options, puis on lance" subtitle="Le calcul tourne en arrière-plan : tu peux fermer l'onglet et revenir, il sera dans l'historique." />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1fr)]">
        <Card className="p-5 sm:p-6">
          <h2 className="mb-4 font-medium">Récapitulatif</h2>
          <dl className="divide-y divide-line text-sm">
            <Row label="Vidéo">
              <span className="line-clamp-1">{p.video.title}</span>
            </Row>
            <Row label="Niveau">
              <span className="mr-2 rounded-md bg-accent-soft px-1.5 py-0.5 font-mono text-[11px] text-accent">Niv. {info.step}</span>
              {info.title}
            </Row>
            <Row label="Passage">
              <span className="font-mono tabular">
                {timecode(p.selection.start, fps)} → {timecode(p.selection.end, fps)}
              </span>
              <span className="ml-2 text-muted">
                {seconds(len)} · {frames} img
              </span>
            </Row>
            <Row label="Visages">
              <ul className="flex flex-col gap-2 py-0.5">
                {active.map((m, i) => {
                  const person = persons.find((x) => x.id === m.person);
                  const crop = cropOf(m);
                  return (
                    <li key={i} className="flex items-center gap-2.5">
                      {crop ? <img src={crop} alt="" className="size-8 rounded-full object-cover ring-1 ring-line-strong" /> : <span className="size-8 rounded-full bg-raised" />}
                      <ArrowRight className="size-3.5 text-faint" />
                      {person?.cover_url ? (
                        <img src={person.cover_url} alt="" className="size-8 rounded-full object-cover ring-2" style={{ ["--tw-ring-color" as string]: styleOf(persons, m.person).color }} />
                      ) : (
                        <PersonBadge style={styleOf(persons, m.person)} size="md" />
                      )}
                      <span>{person?.name ?? `Personne ${m.person}`}</span>
                      <span className="text-[12px] text-muted">
                        {person ? `${person.count} photo${person.count > 1 ? "s" : ""}` : ""}
                      </span>
                    </li>
                  );
                })}
              </ul>
            </Row>
          </dl>
          {!active.length && (
            <Notice tone="warn" className="mt-3">
              Aucun visage associé : reviens à l'étape Visages.
            </Notice>
          )}
          <Button variant="ghost" size="sm" className="mt-4 -ml-2" icon={<Scissors className="size-4" />} onClick={p.onEditSegment}>
            Modifier le passage
          </Button>
        </Card>

        <Card className="flex flex-col p-5 sm:p-6">
          <h2 className="mb-4 font-medium">Options</h2>
          <div className="mb-1 text-[13px] font-medium">Fichier de sortie</div>
          <SegmentedControl
            value={output}
            onChange={setOutput}
            className="w-full"
            options={[
              { value: "segment", label: "Extrait seul", icon: <Scissors className="size-4" /> },
              { value: "full", label: "Vidéo complète", icon: <Film className="size-4" /> },
            ]}
          />
          <p className="mt-2 mb-4 text-[13px] text-muted">
            {output === "segment"
              ? `Un clip de ${seconds(len)} avec le son d'origine, prêt pour TikTok / Reels.`
              : `Toute la vidéo (${duration(p.video.info!.duration)}) : seul le passage de ${seconds(len)} est transformé, le reste est simplement recopié autour (≈ +${duration(fullExtra)}).`}
          </p>
          <div className="space-y-3 border-t border-line pt-4">
            <Switch checked={stabilize} onChange={setStabilize} label="Stabilisation" description="Lisse les tremblements du visage d'une image à l'autre." />
            <Switch checked={aiLabel} onChange={setAiLabel} label="Étiquette « Contenu modifié par IA »" description="Petite mention en bas à droite, recommandée pour publier." />
            {highFps && (
              <Switch
                checked={limitFps}
                onChange={setLimitFps}
                label={`Limiter à ${Math.round(cappedFps(fps, cap))} i/s`}
                description={`Vidéo en ${Math.round(fps)} i/s : une image sur ${Math.round(fps / cappedFps(fps, cap))} est gardée, rendu ${Math.round(fps / cappedFps(fps, cap))}× plus rapide. Standard TikTok / Reels ; décoché : cadence d'origine.`}
              />
            )}
            {!aiLabel && info.step >= 2 && (
              <Notice tone="warn">Plus le rendu est réaliste, plus l'étiquette compte : sans elle, la vidéo peut passer pour vraie une fois partagée.</Notice>
            )}
          </div>
          <GpuOption checked={useGpu} onChange={setUseGpu} configured={gpuConfigured} required={!!info.gpuOnly} />
          <div className="mt-auto pt-6">
            {p.status && !p.status.worker && (
              <Notice tone="warn" className="mb-4">
                Le worker n'est pas lancé : le rendu restera en file d'attente. Lance <code className="font-mono">scripts\dev.ps1</code>.
              </Notice>
            )}
            {launch.error && (
              <Notice tone="danger" className="mb-4">
                {launch.error.message}
              </Notice>
            )}
            <div className="flex items-center justify-between gap-4">
              <div>
                <div className="flex items-center gap-1.5 text-[13px] text-muted">
                  <Cpu className="size-3.5" /> Temps estimé
                </div>
                <div className="font-mono text-lg tabular">≈ {duration(estimate)}</div>
                <div className="mt-0.5 font-mono text-[11px] text-faint tabular" title="Images du passage × secondes par image (mesurée sur ta machine) × visages remplacés">
                  {frames} img × {spf.toFixed(1).replace(".", ",")} s{active.length > 1 ? ` × ${swapFactor.toFixed(2).replace(".", ",")} (${active.length} visages)` : ""}
                  {output === "full" ? ` + ${duration(fullExtra)} recopie` : ""}
                </div>
              </div>
              <Button
                variant="primary"
                size="lg"
                loading={launch.isPending}
                disabled={!active.length || blockedByGpu}
                title={blockedByGpu ? "Ce niveau nécessite l'option GPU" : undefined}
                onClick={() => launch.mutate()}
                icon={<Wand2 className="size-4" />}
              >
                Lancer le rendu
              </Button>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

/** Option GPU : décochée par défaut, grisée tant qu'aucun GPU n'est branché, jamais cochée à la place de l'utilisateur. */
function GpuOption({ checked, onChange, configured, required }: { checked: boolean; onChange: (v: boolean) => void; configured: boolean; required: boolean }) {
  return (
    <div className="mt-4 border-t border-line pt-4">
      <label className={cx("flex items-start gap-3", configured ? "cursor-pointer" : "cursor-not-allowed opacity-60")}>
        <input type="checkbox" checked={checked} disabled={!configured} onChange={(e) => onChange(e.target.checked)} className="peer sr-only" />
        <span
          className={cx(
            "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-md ring-1 transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-accent",
            checked ? "bg-accent text-accent-ink ring-accent" : "bg-bg ring-line-strong",
          )}
        >
          {checked && <Check className="size-3.5" strokeWidth={3} />}
        </span>
        <span>
          <span className="flex items-center gap-1.5 text-sm font-medium">
            <Zap className="size-3.5 text-warn" /> Utiliser le GPU (ZeroGPU)
          </span>
          <span className="mt-0.5 block text-[13px] text-muted">
            {configured
              ? "Consomme ton quota ZeroGPU. Décoché : calcul sur ce PC."
              : "Aucun GPU ZeroGPU branché : le calcul se fait sur ce PC."}
          </span>
        </span>
      </label>
      {required && !checked && (
        <Notice tone="warn" className="mt-3">
          Ce niveau ne peut tourner que sur GPU : coche l'option{configured ? "" : " (après avoir branché ZeroGPU)"} pour lancer le rendu.
        </Notice>
      )}
    </div>
  );
}

function Row({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div className="flex items-center gap-4 py-3">
      <dt className="w-20 shrink-0 text-muted">{label}</dt>
      <dd className="flex min-w-0 flex-1 items-center">{children}</dd>
    </div>
  );
}

const STAGES: { key: Job["stage"]; label: string }[] = [
  { key: "cut", label: "Découpe" },
  { key: "swap", label: "Remplacement du visage" },
  { key: "assemble", label: "Assemblage + son" },
];

function overall(j: Job): number {
  if (j.status === "queued") return 0;
  if (j.stage === "cut") return 0.03;
  if (j.stage === "swap") return 0.05 + 0.9 * (j.total ? j.done / j.total : 0);
  if (j.stage === "assemble") return 0.97;
  return 0;
}

function Running({ job, video, worker }: { job: Job; video: Video; worker: boolean }) {
  const qc = useQueryClient();
  const cancel = useMutation({ mutationFn: () => api.cancelJob(job.id), onSuccess: (j) => qc.setQueryData(["job", j.id], j) });
  const frac = overall(job);
  const stageIdx = STAGES.findIndex((s) => s.key === job.stage);
  const queued = job.status === "queued";
  const hasPreview = job.stage === "swap" || job.stage === "assemble";
  const info = video.info!;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 4 · Rendu"
        title={queued ? "En file d'attente…" : job.status === "cancelling" ? "Annulation…" : "Rendu en cours"}
        subtitle={
          queued && !worker
            ? "Le worker n'est pas lancé : démarre scripts\\dev.ps1 pour que le rendu commence."
            : queued && job.queue_ahead > 0
              ? `${job.queue_ahead} rendu${job.queue_ahead > 1 ? "s" : ""} avant le tien : le worker les traite un par un (historique pour voir ou annuler).`
              : "Tu peux changer d'onglet, le calcul continue."
        }
        right={
          <Button variant="danger" size="md" icon={<X className="size-4" />} onClick={() => cancel.mutate()} loading={cancel.isPending} disabled={job.status === "cancelling"}>
            Annuler
          </Button>
        }
      />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="relative overflow-hidden rounded-[var(--radius-card)] bg-black ring-1 ring-line" style={{ aspectRatio: `${info.width} / ${info.height}` }}>
          {hasPreview && <img src={`${job.preview_url}?v=${job.done}`} alt="Dernière image calculée" className="absolute inset-0 size-full object-contain" />}
          {!hasPreview && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 text-muted">
              <Clapperboard className="size-8 animate-pulse" />
              <span className="text-sm">{queued ? "En attente du worker" : "Préparation de l'extrait"}</span>
            </div>
          )}
          {hasPreview && (
            <span className="absolute top-3 left-3 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-[11px] backdrop-blur-md">
              <span className="size-1.5 animate-pulse rounded-full bg-accent" /> Aperçu en direct
            </span>
          )}
        </div>

        <Card className="flex flex-col p-5 sm:p-6">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-5xl font-medium tracking-tight tabular">{Math.round(frac * 100)}<span className="text-2xl text-muted">%</span></span>
            {job.eta != null && <span className="font-mono text-sm text-muted tabular">reste ≈ {duration(job.eta)}</span>}
          </div>
          <ProgressBar value={frac} className="mt-4 h-2" />

          <ol className="mt-6 space-y-1">
            {STAGES.map((s, i) => {
              const state = queued || i > stageIdx ? "todo" : i < stageIdx ? "done" : "active";
              return (
                <li key={s.key} className={cx("flex items-center gap-3 rounded-lg px-2 py-2 text-sm", state === "active" && "bg-raised")}>
                  <span
                    className={cx(
                      "flex size-6 items-center justify-center rounded-full text-[11px]",
                      state === "done" && "bg-accent text-accent-ink",
                      state === "active" && "bg-accent-soft text-accent ring-1 ring-accent/40",
                      state === "todo" && "bg-overlay text-faint",
                    )}
                  >
                    {state === "done" ? <Check className="size-3.5" strokeWidth={3} /> : state === "active" ? <Loader2 className="size-3.5 animate-spin" /> : i + 1}
                  </span>
                  <span className={cx(state === "todo" ? "text-faint" : "text-fg")}>{s.label}</span>
                  {s.key === "swap" && state !== "todo" && (
                    <span className="ml-auto font-mono text-[12px] text-muted tabular">
                      {job.done}/{job.total}
                    </span>
                  )}
                </li>
              );
            })}
          </ol>

          <div className="mt-auto grid grid-cols-2 gap-3 border-t border-line pt-4 text-[13px]">
            <div>
              <div className="text-faint">Écoulé</div>
              <div className="font-mono tabular">{job.elapsed ? duration(job.elapsed) : "—"}</div>
            </div>
            <div>
              <div className="text-faint">Vitesse</div>
              <div className="font-mono tabular">{job.elapsed && job.done ? `${(job.elapsed / job.done).toFixed(1).replace(".", ",")} s/img` : "—"}</div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Done({ job, video, onJob, onEditSegment, onNewVideo }: { job: Job } & Props) {
  const info = video.info!;
  const [showConfetti] = useState(() => Date.now() / 1000 - job.created_at < 3600);
  return (
    <div>
      <SectionTitle
        eyebrow="Étape 4 · Rendu"
        title={
          <span className="flex items-center gap-3">
            C'est prêt
            {showConfetti && (
              <motion.span initial={{ rotate: -20, scale: 0 }} animate={{ rotate: 0, scale: 1 }} transition={{ type: "spring", stiffness: 300, damping: 14 }}>
                <Sparkles className="size-6 text-accent" />
              </motion.span>
            )}
          </span>
        }
        subtitle="Fais glisser le rideau pour comparer avant / après."
        right={
          <a href={`${job.result_url}?download=1`} download>
            <Button variant="primary" size="lg" icon={<Download className="size-4" />}>
              Télécharger le MP4
            </Button>
          </a>
        }
      />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.6fr)_minmax(0,1fr)]">
        <Compare before={job.before_url!} after={job.result_url!} aspect={`${info.width} / ${info.height}`} fps={info.fps} />
        <div className="flex flex-col gap-4">
          <Card className="p-5">
            <dl className="grid grid-cols-2 gap-4 text-[13px]">
              <Stat label="Images traitées" value={`${job.total}`} />
              <Stat label="Temps total" value={job.elapsed ? duration(job.elapsed) : "—"} />
              <Stat label="Vitesse" value={job.sec_per_frame ? `${job.sec_per_frame.toFixed(2).replace(".", ",")} s/img` : "—"} />
              <Stat label="Sortie" value={job.params.output === "full" ? "Vidéo complète" : "Extrait seul"} />
            </dl>
          </Card>
          <AnimatePresence>
            {job.warnings.map((w) => (
              <Notice key={w} tone="warn">
                {w}
              </Notice>
            ))}
          </AnimatePresence>
          <div className="grid gap-2">
            <Button variant="secondary" icon={<Scissors className="size-4" />} onClick={onEditSegment}>
              Autre passage, même vidéo
            </Button>
            <Button variant="secondary" icon={<RotateCcw className="size-4" />} onClick={() => onJob(null)}>
              Relancer avec d'autres options
            </Button>
            <Button variant="ghost" icon={<ArrowLeft className="size-4" />} onClick={onNewVideo}>
              Nouvelle vidéo
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

function Stat({ label, value }: { label: string; value: string }) {
  return (
    <div>
      <dt className="text-faint">{label}</dt>
      <dd className="mt-0.5 font-mono text-[15px] tabular">{value}</dd>
    </div>
  );
}
