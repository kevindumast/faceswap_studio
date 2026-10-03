import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { ArrowLeft, ArrowRight, Check, Clapperboard, Cpu, Download, Film, Loader2, Pause, Play, RotateCcw, ScanSearch, Scissors, Sparkle, Sparkles, Wand2, X, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { ApiError, api, type Job, type JobParams, type Level, type Mapping, type Resolution, type Status, type Video } from "../lib/api";
import { levelInfo } from "../lib/levels";
import { faceIndex, styleOf } from "../lib/people";
import { cappedFps, clockAt, duration, renderSpeed, seconds, timecode } from "../lib/time";
import { Button, Card, Notice, ProgressBar, SectionTitle, SegmentedControl, Switch, cx } from "../components/ui";
import { Compare } from "../components/Compare";
import { openEngineSettings } from "../components/EngineSettings";
import { chrono } from "../components/SpaceStatus";
import { ReferenceThumb } from "../components/PersonReference";
import { parseStage, type StageKey } from "../lib/stages";
import type { Selection } from "../components/trimmer/Trimmer";
import { PersonBadge } from "./FacesStep";
import { ReviewStep } from "./ReviewStep";
import { fitToScreen } from "../lib/fit";

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
    refetchInterval: (q) => (q.state.data && TERMINAL.includes(q.state.data.status) ? false : ["paused", "review"].includes(q.state.data?.status ?? "") ? 3000 : 1000),
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
  if (j.status === "paused") return <PausedView job={j} video={p.video} />;
  if (j.status === "review") return <ReviewStep job={j} />;
  return <Running job={j} video={p.video} worker={p.status?.worker ?? true} />;
}

function Setup(p: Props) {
  const qc = useQueryClient();
  const [output, setOutput] = useState<JobParams["output"]>("segment");
  const [stabilize, setStabilize] = useState(true);
  const [aiLabel, setAiLabel] = useState(true);
  // Jamais coché d'office ni mémorisé : le GPU ne consomme du quota que si on le demande pour CE rendu.
  const [useGpu, setUseGpu] = useState(false);
  // Idem : coûte du temps de calcul en plus (CodeFormer), jamais activé d'office.
  const [restore, setRestore] = useState(false);
  // Arrêt avant l'assemblage si des visages sont mal suivis (rendu sur ce PC seulement : il faut le journal image par image).
  const [review, setReview] = useState(true);
  const info = levelInfo(p.level);
  const isFaceLevel = p.level === "face" || p.level === "face_tone";
  const restoreStatus = p.status?.restore;
  const install = useMutation({
    mutationFn: () => api.installModels("restore"),
    onSettled: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });
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
  // Niveau 4 : son propre Space (Wan2.2-Animate), 1 ou 2 personnes (un passage chacune), durée limitée, résolution au choix.
  const isCharacter = p.level === "character";
  const character = p.status?.gpu.character;
  // 480p par défaut (compte PRO : 40 min/jour) ; 360p si le compte est gratuit (5 min/jour).
  const [resolution, setResolution] = useState<Resolution>(() => (p.status?.gpu.pro === false ? "360p" : "480p"));
  const [steps, setSteps] = useState(() => character?.steps ?? 4);
  // Lumière du décor sur la personne : plus intégrée, mais elle change ses couleurs (tout orange sur fond orange).
  const [relight, setRelight] = useState(false);
  // Visage refait net sur ce PC après la génération (Wan le génère en quelques dizaines de pixels).
  const [facePass, setFacePass] = useState(true);
  const maxCharacter = character?.max_s ?? 10;
  const tooLong = isCharacter && len > maxCharacter + 0.01;
  const maxPeople = character?.max_people ?? 2;
  const people = Math.max(1, active.length);
  const badCount = isCharacter && active.length > maxPeople;
  const gpuConfigured = isCharacter ? !!character?.configured : !!p.status?.gpu.configured;
  const canReview = !isCharacter && !useGpu;
  const blockedByGpu = !!info.gpuOnly && !useGpu;
  // Mode « vidéo complète » : le reste du clip est seulement réencodé (≈ 4× plus vite que le temps réel).
  const fullExtra = p.video.info!.duration * 0.25;
  // ZeroGPU : temps GPU (= quota) par image, + envoi, file d'attente et réveil éventuel du Space.
  const gpuSpf = p.status?.gpu.sec_per_frame?.[p.level] ?? 0.1;
  const perBlockStep = character?.gpu_s_per_block_step[resolution] ?? (resolution === "480p" ? 36 : 16);
  const characterBlocks = blocksOf(len);
  // Niveau 4 : un passage GPU par personne remplacée, l'un après l'autre.
  const characterPass = characterGpuSeconds(len, perBlockStep, steps);
  const gpuSeconds = isCharacter ? characterPass * people : frames * gpuSpf * swapFactor;
  // Le Space du niveau 4 réserve 15 % de plus que l'estimation (marge), et ZeroGPU multiplie encore par 1,5 (cartes
  // Blackwell : 227 s demandés → 340 s refusés) : il refuse d'emblée une réservation plus grande que le quota restant.
  // Les passages s'enchaînent : seule la réservation du dernier compte en plus.
  const gpuReserved = isCharacter ? gpuSeconds + characterPass * (1.15 * 1.5 - 1) : gpuSeconds;
  // Visage net sur ce PC : même vitesse que le niveau 1 mesurée ici, pour chaque personne.
  const facePassSeconds = isCharacter && facePass ? frames * (p.status?.levels.face?.sec_per_frame ?? 1.5) * swapFactor : 0;
  // Reste exact donné par ZeroGPU à son dernier refus (jusqu'à l'heure du prochain essai), sinon estimation locale.
  const knownQuota = p.status?.gpu.quota ?? null;
  const quotaLeft = knownQuota?.left_s != null
    ? knownQuota.left_s
    : (p.status?.gpu.free_quota_s ?? 300) - (p.status?.gpu.used_today_s ?? 0);
  const estimate = useGpu
    ? gpuSeconds + (isCharacter ? 90 + facePassSeconds : 60) + (output === "full" ? fullExtra : 0)
    : frames * spf * swapFactor + 15 + (output === "full" ? fullExtra : 0);
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
        ...(isCharacter ? { resolution, steps, relight, face_pass: facePass } : {}),
        ...(isFaceLevel ? { restore } : {}),
        review: canReview && review,
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
                      {isCharacter && person ? (
                        <ReferenceThumb person={person} />
                      ) : (
                        <span className="text-[12px] text-muted">{person ? `${person.count} photo${person.count > 1 ? "s" : ""}` : ""}</span>
                      )}
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
            {!isCharacter && (
              <Switch checked={stabilize} onChange={setStabilize} label="Stabilisation" description="Lisse les tremblements du visage d'une image à l'autre." />
            )}
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
            {!isCharacter && (
              <Switch
                checked={canReview && review}
                onChange={setReview}
                disabled={!canReview}
                label="Vérifier les images avant l'assemblage"
                description={
                  canReview
                    ? "Si un visage est perdu en route ou pas reconnu, le rendu s'arrête avant d'ajouter le son : tu choisis quoi corriger, seules ces images sont recalculées."
                    : "Pas encore avec l'option GPU (ZeroGPU) : le détail image par image reste sur le Space."
                }
              />
            )}
          </div>
          {isFaceLevel && (
            <div className="mt-4 border-t border-line pt-4">
              {restoreStatus?.ready ? (
                <Switch
                  checked={restore}
                  onChange={setRestore}
                  disabled={useGpu}
                  label="Netteté du visage"
                  description={
                    useGpu
                      ? "Pas encore disponible avec l'option GPU (ZeroGPU)."
                      : "Le visage généré (128 px) est ré-agrandi et un peu flou : cette option le retouche avant de le recoller. Plus lent (~2× le temps de ce niveau)."
                  }
                />
              ) : (
                <div>
                  <div className="mb-1 flex items-center gap-1.5 text-[13px] font-medium">
                    <Sparkle className="size-3.5 text-faint" /> Netteté du visage
                  </div>
                  {restoreStatus?.installing ? (
                    <div>
                      <div className="mb-1.5 flex justify-between text-[13px] text-muted">
                        <span>Installation du modèle…</span>
                        <span className="font-mono tabular">{Math.round(restoreStatus.installing.progress * 100)} %</span>
                      </div>
                      <ProgressBar value={restoreStatus.installing.progress} />
                    </div>
                  ) : (
                    <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-raised p-3 ring-1 ring-line">
                      <span className="text-[13px] text-muted">
                        Retouche le visage généré, plus net qu'avec l'agrandissement brut ({restoreStatus?.install_mb ?? 360} Mo, une seule fois).
                      </span>
                      <Button variant="secondary" size="sm" icon={<Download className="size-3.5" />} loading={install.isPending} onClick={() => install.mutate()}>
                        Installer
                      </Button>
                    </div>
                  )}
                  {restoreStatus?.install_error && (
                    <Notice tone="danger" className="mt-2">
                      {restoreStatus.install_error}
                    </Notice>
                  )}
                </div>
              )}
            </div>
          )}
          {isCharacter && (
            <div className="mt-4 border-t border-line pt-4">
              <div className="mb-1 text-[13px] font-medium">Qualité de la personne générée</div>
              <SegmentedControl
                value={resolution}
                onChange={setResolution}
                className="w-full"
                options={[
                  { value: "360p", label: "360p · économique" },
                  { value: "480p", label: "480p · plus net" },
                ]}
              />
              <p className="mt-2 text-[13px] text-muted">
                La personne est générée en {resolution} puis recollée sur la vidéo : le reste de l'image garde sa netteté.
                {resolution === "480p" ? " Environ 2,2× plus de quota qu'en 360p." : ""}
              </p>
              <div className="mt-3 mb-1 text-[13px] font-medium">Étapes de génération</div>
              <SegmentedControl
                value={String(steps)}
                onChange={(v) => setSteps(Number(v))}
                className="w-full"
                options={[
                  { value: "4", label: "4 · rapide" },
                  { value: "8", label: "8 · plus soigné" },
                ]}
              />
              <p className="mt-2 text-[13px] text-muted">
                Modèle distillé : 4 étapes donnent déjà une image proche des 20 étapes officielles. 8 affine les détails, pour 2× plus de quota.
              </p>
              <div className="mt-3 space-y-3">
                <Switch
                  checked={facePass}
                  onChange={setFacePass}
                  label="Visage net (sur ce PC)"
                  description={`Après la génération, ton vrai visage est remis en pleine résolution, comme au niveau 1${
                    restoreStatus?.ready ? ", avec la netteté" : ""
                  } : ≈ ${duration(facePassSeconds || frames * (p.status?.levels.face?.sec_per_frame ?? 1.5) * swapFactor)} de plus, sans quota.`}
                />
                <Switch
                  checked={relight}
                  onChange={setRelight}
                  label="Lumière de la scène"
                  description="La personne prend l'éclairage du décor : plus intégrée, mais ses couleurs changent (tout vire à l'orange sur un fond orange). Décoché : les couleurs de ta photo."
                />
              </div>
              {tooLong && (
                <Notice tone="warn" className="mt-3">
                  Le niveau 4 est limité à {seconds(maxCharacter)} (quota GPU) : raccourcis le passage ({seconds(len)} actuellement).
                </Notice>
              )}
              {badCount && (
                <Notice tone="warn" className="mt-3">
                  Le niveau 4 remplace au plus {maxPeople} personnes par rendu : passe les autres sur « Ne pas remplacer » à l'étape Visages.
                </Notice>
              )}
              {!badCount && active.length > 1 && (
                <Notice tone="info" className="mt-3">
                  {active.length} personnes = {active.length} passages sur le GPU, l'un après l'autre : environ {active.length}× plus de quota et de temps
                  qu'une seule.
                </Notice>
              )}
            </div>
          )}
          {info.localOnly ? (
            <Notice tone="info" className="mt-4">
              Ce niveau tourne sur le processeur de ce PC (pas sur ZeroGPU) : compte environ 8 s par image. Le décor caché par
              l'ancienne tête est reconstruit avant le rendu (~30 s de préparation).
            </Notice>
          ) : (
            <GpuOption
              checked={useGpu}
              onChange={setUseGpu}
              configured={gpuConfigured}
              required={!!info.gpuOnly}
              quotaNeeded={gpuSeconds}
              quotaReserved={gpuReserved}
              quotaLeft={quotaLeft}
              quotaBackAt={knownQuota?.retry_at}
              pro={!!p.status?.gpu.pro}
              spaceLabel={isCharacter ? "le Space du niveau 4" : undefined}
            />
          )}
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
                  {isCharacter ? (
                    <>
                      {characterBlocks} bloc{characterBlocks > 1 ? "s" : ""} × {steps} étapes × {Math.round(perBlockStep)} s ({resolution}) + préparation
                      {people > 1 ? ` × ${people} personnes` : ""} + ~1 min 30 d'envoi et de recollage
                      {facePass ? ` + ${duration(facePassSeconds)} de visage sur ce PC` : ""}
                    </>
                  ) : (
                    <>
                      {frames} img × {(useGpu ? gpuSpf : spf).toFixed(useGpu ? 2 : 1).replace(".", ",")} s{useGpu ? " (ZeroGPU)" : ""}
                      {active.length > 1 ? ` × ${swapFactor.toFixed(2).replace(".", ",")} (${active.length} visages)` : ""}
                      {useGpu ? " + ~1 min d'envoi" : ""}
                    </>
                  )}
                  {output === "full" ? ` + ${duration(fullExtra)} recopie` : ""}
                </div>
              </div>
              <Button
                variant="primary"
                size="lg"
                loading={launch.isPending}
                disabled={!active.length || blockedByGpu || tooLong || badCount}
                title={blockedByGpu ? "Ce niveau nécessite l'option GPU" : tooLong ? "Passage trop long pour le niveau 4" : badCount ? `${maxPeople} personnes au plus au niveau 4` : undefined}
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

/** Blocs de génération du niveau 4 : Wan-Animate travaille par blocs de 77 images à 30 i/s (dont 1 reprise). */
function blocksOf(len: number): number {
  return Math.max(1, Math.ceil((Math.max(1, Math.round(len * 30)) - 1) / 76));
}

/** Temps GPU d'un passage du niveau 4 : même formule que le Space (préparation + chaque étape de chaque bloc). */
function characterGpuSeconds(len: number, perBlockStep: number, steps: number): number {
  return 10 + 6 * len + blocksOf(len) * steps * perBlockStep;
}

/** Option GPU : décochée par défaut, grisée tant qu'aucun GPU n'est branché, jamais cochée à la place de l'utilisateur. */
function GpuOption({
  checked,
  onChange,
  configured,
  required,
  quotaNeeded,
  quotaReserved,
  quotaLeft,
  quotaBackAt,
  pro,
  spaceLabel,
}: {
  checked: boolean;
  onChange: (v: boolean) => void;
  configured: boolean;
  required: boolean;
  quotaNeeded: number;
  /** Temps réservé auprès de ZeroGPU (estimation + marge du Space) : c'est lui qui doit tenir dans le quota restant. */
  quotaReserved: number;
  quotaLeft: number;
  /** Heure du prochain essai annoncée par ZeroGPU à son dernier refus (le reste est alors exact). */
  quotaBackAt?: number;
  /** Compte PRO : 40 min/jour au lieu de 5. */
  pro: boolean;
  /** Space concerné, s'il n'est pas celui des niveaux 1-2 (ex. « le Space du niveau 4 »). */
  spaceLabel?: string;
}) {
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
            {configured ? (
              <>
                ≈ {duration(quotaNeeded)} de GPU sur ton quota (il reste{" "}
                {quotaBackAt ? `${duration(Math.max(0, quotaLeft))} d'après ZeroGPU, prochain essai possible ${clockAt(quotaBackAt)}` : `≈ ${duration(Math.max(0, quotaLeft))} aujourd'hui${pro ? " en PRO" : " en gratuit"}`}).
                Décoché : calcul sur ce PC.
              </>
            ) : spaceLabel ? (
              `Pas encore branché : ${spaceLabel} se crée en une commande (Moteur → Niveau 4).`
            ) : (
              "Aucun GPU ZeroGPU branché : le calcul se fait sur ce PC."
            )}
          </span>
          {!configured && (
            <button type="button" onClick={openEngineSettings} className="mt-1 text-[13px] text-accent underline-offset-2 hover:underline">
              {spaceLabel ? "Brancher le Space du niveau 4" : "Brancher ZeroGPU"}
            </button>
          )}
        </span>
      </label>
      {configured && checked && quotaReserved > quotaLeft && (
        <Notice tone="warn" className="mt-3">
          Ce rendu réserve ≈ {duration(quotaReserved)} de GPU, plus que le quota restant (≈ {duration(Math.max(0, quotaLeft))}) : ZeroGPU le
          refusera au lancement. {quotaBackAt && <>Prochain essai possible {clockAt(quotaBackAt)}. </>}
          {required ? "Raccourcis le passage ou passe en 360p." : "Raccourcis le passage ou décoche l'option."}
        </Notice>
      )}
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

type Stage = { key: Job["stage"]; label: string };

const STAGES: Stage[] = [
  { key: "cut", label: "Découpe" },
  { key: "swap", label: "Remplacement du visage" },
  { key: "assemble", label: "Assemblage + son" },
];

/** Rendu sur ZeroGPU : réveil éventuel du Space, file du Space, attente d'un GPU, puis le calcul. */
const REMOTE_WAIT: Stage[] = [
  { key: "wake", label: "Réveil du Space" },
  { key: "queue", label: "File d'attente du Space" },
  { key: "gpu", label: "En attente d'un GPU" },
];

const REMOTE_STAGES: Stage[] = [STAGES[0], ...REMOTE_WAIT, { key: "swap", label: "Remplacement du visage (ZeroGPU)" }, STAGES[2]];

const CHARACTER_STAGES: Stage[] = [
  STAGES[0],
  ...REMOTE_WAIT,
  { key: "pose", label: "Squelette et visage" },
  { key: "mask", label: "Silhouette" },
  { key: "generate", label: "Génération de la personne" },
  { key: "face", label: "Visage net (sur ce PC)" },
  { key: "assemble", label: "Recollage + son" },
];

const FIX_STAGES: Stage[] = [{ key: "fix", label: "Recalcul des images corrigées" }];
const REVIEWED_STAGES: Stage[] = [{ key: "assemble", label: "Assemblage + son" }];

function stagesOf(j: Job): Stage[] {
  if (j.params.review_step) return j.params.review_step === "fix" ? FIX_STAGES : REVIEWED_STAGES;
  if (j.params.level === "character") return CHARACTER_STAGES.filter((s) => s.key !== "face" || j.params.face_pass !== false);
  return j.params.use_gpu ? REMOTE_STAGES : STAGES;
}

function overall(j: Job): number {
  if (j.status === "queued") return 0;
  const frac = j.total ? j.done / j.total : 0;
  const { key, pass, passes } = parseStage(j.stage);
  const before = (pass - 1) / passes; // part déjà faite par les passages précédents (niveau 4, une personne chacun)
  switch (key) {
    case "cut":
      return 0.03;
    case "wake":
      return pass > 1 ? 0.07 + gpuShare(j) * before : 0.04;
    case "queue":
      return pass > 1 ? 0.07 + gpuShare(j) * before : 0.05;
    case "gpu":
      return pass > 1 ? 0.07 + gpuShare(j) * before : 0.06;
    case "swap":
      return j.params.use_gpu ? 0.07 + 0.88 * frac : 0.05 + 0.9 * frac;
    case "fix":
      return 0.02 + 0.97 * frac;
    case "pose":
    case "mask":
    case "generate":
      return 0.07 + gpuShare(j) * (before + frac / passes); // progression du Space niveau 4, en ‰
    case "face":
      return 0.07 + gpuShare(j) + (0.95 - 0.07 - gpuShare(j)) * frac;
    case "assemble":
      return 0.97;
    default:
      return 0;
  }
}

/** Part de la barre pour les passages sur le Space (niveau 4) : un peu moins s'il reste le visage à refaire sur ce PC. */
function gpuShare(j: Job): number {
  return j.params.level === "character" && j.params.face_pass !== false ? 0.7 : 0.88;
}

/** Secondes passées dans l'étape en cours (mesurées dans le navigateur, depuis le dernier changement d'étape). */
function useStageClock(stage: Job["stage"]): number {
  const [since, setSince] = useState(() => Date.now());
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => setSince(Date.now()), [stage]);
  useEffect(() => {
    const id = setInterval(() => setNow(Date.now()), 1000);
    return () => clearInterval(id);
  }, []);
  return Math.max(0, (now - since) / 1000);
}

/** Détail affiché à droite de l'étape active. */
function stageDetail(j: Job, inStage: number): string | null {
  switch (parseStage(j.stage).key) {
    case "wake":
      return `${chrono(j.done + inStage)} / ≈ ${duration(j.total)}`;
    case "queue":
      return j.total ? `${j.done + 1}ᵉ sur ${j.total}` : chrono(inStage);
    case "gpu":
      return chrono(inStage);
    case "swap":
    case "fix":
    case "face":
      return `${j.done}/${j.total}`;
    case "pose":
    case "mask":
    case "generate":
      return j.total ? `${Math.round((j.done / j.total) * 100)} %` : null;
    default:
      return null;
  }
}

const REMOTE_HINT: Partial<Record<StageKey, string>> = {
  wake: "Le Space dormait : il recharge son modèle (≈ 3 min, ou 20 à 40 min pour le niveau 4). Aucun quota consommé pendant l'attente.",
  queue: "Ton rendu attend son tour sur ton Space.",
  gpu: "Le calcul est lancé : ZeroGPU attribue un GPU (file partagée avec les autres utilisateurs de Hugging Face).",
  pose: "Calcul sur ZeroGPU : repérage du squelette et du visage de la personne choisie.",
  mask: "Calcul sur ZeroGPU : découpe de sa silhouette.",
  generate: "Calcul sur ZeroGPU : la personne de ta photo est générée à sa place.",
  swap: "Calcul sur ZeroGPU : remplacement des visages.",
  face: "Sur ce PC, sans quota : ton vrai visage est remis net, en pleine résolution, sur la personne générée.",
};

/** Rendu en pause : ce qui est déjà calculé est gardé, la reprise repart à la même image. */
function PausedView({ job, video }: { job: Job; video: Video }) {
  const qc = useQueryClient();
  const resume = useMutation({ mutationFn: () => api.resumeJob(job.id), onSuccess: (j) => qc.setQueryData(["job", j.id], j) });
  const cancel = useMutation({ mutationFn: () => api.cancelJob(job.id), onSuccess: (j) => qc.setQueryData(["job", j.id], j) });
  const [watching, setWatching] = useState(false);
  const info = video.info!;
  const frac = job.total ? job.done / job.total : 0;
  const canWatch = !!(job.partial_url && job.before_url);
  return (
    <div>
      <SectionTitle
        eyebrow="Étape 4 · Rendu"
        title="En pause"
        subtitle="Tu peux fermer l'appli ou éteindre le PC : le rendu reprendra exactement à cette image."
        right={
          <div className="flex gap-2">
            <Button variant="primary" size="md" icon={<Play className="size-4 fill-current" />} onClick={() => resume.mutate()} loading={resume.isPending}>
              Reprendre
            </Button>
            <Button variant="danger" size="md" icon={<X className="size-4" />} onClick={() => cancel.mutate()} loading={cancel.isPending}>
              Annuler
            </Button>
          </div>
        }
      />
      {job.error && (
        <Notice tone="info" className="mb-5">
          {job.error}
        </Notice>
      )}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        {watching && canWatch ? (
          <Compare key={job.partial_url} before={job.before_url!} after={job.partial_url!} width={info.width} height={info.height} fps={info.fps} autoPlay />
        ) : (
          <div className="relative overflow-hidden rounded-[var(--radius-card)] bg-black ring-1 ring-line" style={fitToScreen(info.width, info.height)}>
            {job.done > 0 && <img src={`${job.preview_url}?v=${job.done}`} alt="Dernière image calculée" className="absolute inset-0 size-full object-contain opacity-70" />}
            <span className="absolute top-3 left-3 flex items-center gap-1.5 rounded-full bg-black/60 px-2.5 py-1 text-[11px] backdrop-blur-md">
              <Pause className="size-3 fill-current" /> Dernière image calculée
            </span>
            {canWatch && (
              <button type="button" onClick={() => setWatching(true)} className="group absolute inset-0 flex flex-col items-center justify-center gap-3 outline-none">
                <span className="flex size-16 items-center justify-center rounded-full bg-accent text-accent-ink shadow-lg transition-transform group-hover:scale-105 group-focus-visible:ring-4 group-focus-visible:ring-accent/40">
                  <Play className="ml-1 size-7 fill-current" />
                </span>
                <span className="rounded-full bg-black/60 px-3 py-1 text-[13px] backdrop-blur-md">Revoir ce qui est déjà rendu (avant / après)</span>
              </button>
            )}
          </div>
        )}
        <Card className="flex flex-col p-5 sm:p-6">
          <div className="flex items-baseline justify-between">
            <span className="font-mono text-5xl font-medium tracking-tight tabular">
              {Math.round(frac * 100)}
              <span className="text-2xl text-muted">%</span>
            </span>
            <span className="font-mono text-sm text-muted tabular">
              {job.done}/{job.total} images
            </span>
          </div>
          <ProgressBar value={frac} tone="warn" className="mt-4 h-2" />
          <p className="mt-6 text-[13px] text-muted">
            Déjà calculé : {job.done} image{job.done > 1 ? "s" : ""}. Pendant la pause, le worker est libre pour d'autres rendus ;
            à la reprise, ce rendu repasse en tête de file.
          </p>
          {(resume.error || cancel.error) && (
            <Notice tone="danger" className="mt-4">
              {(resume.error ?? cancel.error)!.message}
            </Notice>
          )}
        </Card>
      </div>
    </div>
  );
}

function Running({ job, video, worker }: { job: Job; video: Video; worker: boolean }) {
  const qc = useQueryClient();
  const cancel = useMutation({ mutationFn: () => api.cancelJob(job.id), onSuccess: (j) => qc.setQueryData(["job", j.id], j) });
  const pause = useMutation({ mutationFn: () => api.pauseJob(job.id), onSuccess: (j) => qc.setQueryData(["job", j.id], j) });
  const frac = overall(job);
  const stages = stagesOf(job);
  const { key: stageKey, pass, passes } = parseStage(job.stage);
  const stageIdx = stages.findIndex((s) => s.key === stageKey);
  const passLabel = passes > 1 ? `personne ${pass}/${passes}` : null;
  const queued = job.status === "queued";
  const remote = job.params.use_gpu;
  // Aperçu en direct seulement pour un rendu sur ce PC (le Space ne renvoie que le résultat final).
  const reviewStep = job.params.review_step;
  const hasPreview = !remote && !reviewStep && (stageKey === "swap" || stageKey === "assemble");
  const inStage = useStageClock(job.stage);
  const info = video.info!;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 4 · Rendu"
        title={
          queued
            ? "En file d'attente…"
            : job.status === "cancelling"
              ? "Annulation…"
              : job.status === "pausing"
                ? "Mise en pause…"
                : reviewStep === "fix"
                  ? "Corrections en cours"
                  : reviewStep === "assemble"
                    ? "Assemblage"
                    : `Rendu en cours${passLabel ? ` · ${passLabel}` : ""}`
        }
        subtitle={
          queued && !worker
            ? "Le worker n'est pas lancé : démarre scripts\\dev.ps1 pour que le rendu commence."
            : queued && job.queue_ahead > 0
              ? `${job.queue_ahead} rendu${job.queue_ahead > 1 ? "s" : ""} avant le tien : le worker les traite un par un (historique pour voir ou annuler).`
              : "Tu peux changer d'onglet, le calcul continue."
        }
        right={
          <div className="flex gap-2">
            {job.pausable && (
              <Button
                variant="secondary"
                size="md"
                icon={<Pause className="size-4" />}
                onClick={() => pause.mutate()}
                loading={pause.isPending || job.status === "pausing"}
                disabled={job.status === "cancelling"}
                title="Arrêter ici et reprendre plus tard, à la même image"
              >
                Pause
              </Button>
            )}
            <Button
              variant="danger"
              size="md"
              icon={<X className="size-4" />}
              onClick={() => cancel.mutate()}
              loading={cancel.isPending}
              disabled={job.status === "cancelling"}
              title={reviewStep ? "Revenir à la vérification (rien n'est perdu)" : undefined}
            >
              {reviewStep ? "Revenir à la vérification" : "Annuler"}
            </Button>
          </div>
        }
      />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.5fr)_minmax(0,1fr)]">
        <div className="relative overflow-hidden rounded-[var(--radius-card)] bg-black ring-1 ring-line" style={fitToScreen(info.width, info.height)}>
          {hasPreview && <img src={`${job.preview_url}?v=${job.done}`} alt="Dernière image calculée" className="absolute inset-0 size-full object-contain" />}
          {!hasPreview && (
            <div className="absolute inset-0 flex flex-col items-center justify-center gap-3 px-8 text-center text-muted">
              {remote && !queued && stageKey !== "cut" ? <Zap className="size-8 animate-pulse text-warn" /> : <Clapperboard className="size-8 animate-pulse" />}
              <span className="text-sm text-fg">
                {queued ? "En attente du worker" : stageIdx >= 0 && stageKey !== "cut" ? stages[stageIdx].label : "Préparation de l'extrait"}
                {!queued && passLabel && stageKey !== "cut" && stageKey !== "assemble" ? ` · ${passLabel}` : ""}
                {!queued && stageDetail(job, inStage) ? <span className="ml-2 font-mono text-muted tabular">{stageDetail(job, inStage)}</span> : null}
              </span>
              {remote && !queued && stageKey && REMOTE_HINT[stageKey] && <span className="max-w-md text-[13px]">{REMOTE_HINT[stageKey]}</span>}
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
            {stages.map((s, i) => {
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
                  {state === "active" && stageDetail(job, inStage) && (
                    <span className="ml-auto font-mono text-[12px] text-muted tabular">{stageDetail(job, inStage)}</span>
                  )}
                  {state === "done" && s.key === "swap" && !remote && (
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
              {/* Corrections : toutes les images sont relues, seules les corrigées sont calculées (vitesse sans intérêt). */}
              <div className="text-faint">{remote ? "Moteur" : reviewStep ? "Images relues" : "Vitesse"}</div>
              <div className="font-mono tabular">
                {remote
                  ? "ZeroGPU"
                  : reviewStep
                    ? `${job.done}/${job.total}`
                    : job.elapsed && job.done
                      ? renderSpeed(job.elapsed / job.done)
                      : "—"}
              </div>
            </div>
          </div>
        </Card>
      </div>
    </div>
  );
}

function Done({ job, video, onJob, onEditSegment, onNewVideo }: { job: Job } & Props) {
  const info = video.info!;
  const qc = useQueryClient();
  const [showConfetti] = useState(() => Date.now() / 1000 - job.created_at < 3600);
  const reopen = useMutation({
    mutationFn: () => api.reopenJob(job.id),
    onSuccess: (j) => {
      qc.setQueryData(["job", j.id], j);
      qc.invalidateQueries({ queryKey: ["jobs"] });
    },
  });
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
        <Compare before={job.before_url!} after={job.result_url!} width={info.width} height={info.height} fps={info.fps} />
        <div className="flex flex-col gap-4">
          <Card className="p-5">
            <dl className="grid grid-cols-2 gap-4 text-[13px]">
              <Stat label="Images traitées" value={`${job.total}`} />
              <Stat label="Temps total" value={job.elapsed ? duration(job.elapsed) : "—"} />
              <Stat label="Vitesse" value={job.sec_per_frame ? renderSpeed(job.sec_per_frame, 2) : "—"} />
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
          {reopen.error && <Notice tone="danger">{reopen.error.message}</Notice>}
          <div className="grid gap-2">
            {job.reviewable && (
              <Button variant="secondary" icon={<ScanSearch className="size-4" />} loading={reopen.isPending} onClick={() => reopen.mutate()} title="Voir image par image les visages mal suivis et les corriger, puis réassembler">
                Revoir les images
              </Button>
            )}
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
