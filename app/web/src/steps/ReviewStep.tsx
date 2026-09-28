import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ChevronLeft, ChevronRight, Eye, Loader2, Pause, Play, RotateCcw, SkipBack, SkipForward, Wand2 } from "lucide-react";
import { useCallback, useEffect, useImperativeHandle, useMemo, useRef, useState } from "react";
import { api, type Job, type Person, type Review, type ReviewDecision, type ReviewIssue } from "../lib/api";
import { styleOf } from "../lib/people";
import { clamp, timecode } from "../lib/time";
import { Button, Card, Notice, SectionTitle, cx } from "../components/ui";
import { PersonBadge } from "./FacesStep";

/** Choix pour une piste, en attente d'être appliqué. */
type Choice = Omit<ReviewDecision, "issue">;

const KIND: Record<ReviewIssue["kind"], string> = {
  lost: "Visage perdu par moments",
  mixed: "Change de personne",
  missed: "Visage non remplacé",
  track: "Visage choisi sur l'image",
};

/**
 * Vérification avant l'assemblage : les visages mal suivis (pistes) sont listés, on choisit pour chacun, et seules
 * les images concernées sont recalculées. Le son et l'étiquette ne sont ajoutés qu'au clic sur « Assembler ».
 */
export function ReviewStep({ job }: { job: Job }) {
  const qc = useQueryClient();
  // Refaite à chaque retour sur cet écran : après un recalcul, les pistes et la vidéo ont changé.
  const review = useQuery({ queryKey: ["review", job.id], queryFn: () => api.review(job.id), staleTime: Infinity, refetchOnMount: "always" });
  const faceSet = useQuery({ queryKey: ["faceset", job.face_set_id], queryFn: () => api.faceSet(job.face_set_id) });
  const persons = faceSet.data?.persons ?? [];
  const [choices, setChoices] = useState<Record<number, Choice>>({});
  const [selected, setSelected] = useState<number | null>(null);
  // Pistes cliquées sur l'image sans problème signalé (ex. mauvais visage remplacé) : mêmes choix que les autres.
  const [picked, setPicked] = useState<number[]>([]);
  const [showMinor, setShowMinor] = useState(false);
  const player = useRef<PlayerHandle>(null);

  const r = review.data;
  // Nouvelle analyse : les choix proposés d'office (piste surtout remplacée par une personne) sont repris.
  useEffect(() => {
    if (!r) return;
    const init: Record<number, Choice> = {};
    for (const x of r.issues) if (x.preselect) init[x.id] = { action: "assign", person: x.preselect };
    setChoices(init);
    setSelected(null);
    setPicked([]);
  }, [r]);

  const onJob = (j: Job) => {
    qc.setQueryData(["job", j.id], j);
    qc.invalidateQueries({ queryKey: ["jobs"] });
    if (j.status === "review") qc.invalidateQueries({ queryKey: ["review", j.id] });
  };
  const apply = useMutation({
    mutationFn: (decisions: ReviewDecision[]) => api.submitReview(job.id, r!.version, decisions),
    onSuccess: onJob,
  });
  const assemble = useMutation({ mutationFn: () => api.assembleJob(job.id), onSuccess: onJob });

  if (review.error) {
    return (
      <div>
        <SectionTitle eyebrow="Étape 4 · Vérification" title="Vérification indisponible" />
        <Notice tone="danger">{review.error.message}</Notice>
      </div>
    );
  }
  if (!r) {
    return (
      <div className="flex justify-center py-24">
        <Loader2 className="size-6 animate-spin text-muted" />
      </div>
    );
  }

  const people = [...new Set(r.mappings.map((m) => m.person).filter((p): p is string => !!p))];
  const major = r.issues.filter((x) => !x.minor);
  const minor = r.issues.filter((x) => x.minor);
  const extras = picked.map((t) => trackAsIssue(r, t)).filter((x): x is ReviewIssue => !!x);
  const known = [...r.issues, ...extras];
  const decided = Object.entries(choices).map(([id, c]) => ({ issue: Number(id), ...c }));
  const toCompute = decided.reduce((n, d) => n + framesFor(known.find((x) => x.id === d.issue), d), 0);
  const select = (x: ReviewIssue) => {
    setSelected(x.id);
    player.current?.seek(x.thumb.frame);
  };
  const choose = (id: number, c: Choice | null) =>
    setChoices((prev) => {
      const next = { ...prev };
      if (c) next[id] = c;
      else delete next[id];
      return next;
    });
  const onAssemble = () => {
    if (decided.length && !window.confirm("Tes choix pas encore appliqués seront ignorés. Assembler quand même ?")) return;
    assemble.mutate();
  };
  const error = apply.error ?? assemble.error;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 4 · Vérification"
        title="Vérifie avant l'assemblage"
        subtitle={
          major.length
            ? `${major.length} visage${major.length > 1 ? "s" : ""} mal suivi${major.length > 1 ? "s" : ""}. Choisis quoi faire pour chacun et applique : seules ces images sont recalculées. Le son est ajouté à l'assemblage.`
            : "Plus rien à corriger d'important. Tu peux encore revoir la vidéo, puis assembler."
        }
        right={
          <Button variant={decided.length ? "secondary" : "primary"} size="lg" icon={<Wand2 className="size-4" />} loading={assemble.isPending} onClick={onAssemble}>
            Assembler la vidéo
          </Button>
        }
      />
      {job.error && (
        <Notice tone="info" className="mb-4">
          {job.error}
        </Notice>
      )}
      {r.pending > 0 && (
        <Notice tone="warn" className="mb-4">
          <span className="flex flex-wrap items-center justify-between gap-3">
            {r.pending} image{r.pending > 1 ? "s" : ""} corrigée{r.pending > 1 ? "s" : ""} pas encore recalculée{r.pending > 1 ? "s" : ""} (recalcul interrompu).
            <Button size="sm" variant="secondary" icon={<RotateCcw className="size-3.5" />} loading={apply.isPending} onClick={() => apply.mutate([])}>
              Relancer le recalcul
            </Button>
          </span>
        </Notice>
      )}
      {error && (
        <Notice tone="danger" className="mb-4">
          {error.message}
        </Notice>
      )}
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1.55fr)_minmax(0,1fr)]">
        <div className="min-w-0">
          <Player
            ref={player}
            review={r}
            persons={persons}
            selected={selected}
            onPickTrack={(track) => {
              const x = r.issues.find((i) => i.id === track);
              if (x?.minor) setShowMinor(true);
              if (!x) setPicked((prev) => (prev.includes(track) ? prev : [track, ...prev]));
              setSelected(track);
            }}
          />
          <p className="mt-3 text-[12px] text-faint">
            Cadres : couleur de la personne = remplacé · rouge = visage à vérifier · gris = laissé d'origine. Clic sur n'importe quel
            cadre pour le corriger (même un visage remplacé par erreur). Flèches ← → : image par image (Maj : 1 s), espace : lecture.
          </p>
        </div>
        <Card className="flex max-h-[min(80vh,820px)] min-h-80 flex-col">
          <div className="border-b border-line px-4 py-3">
            <h2 className="font-medium">À vérifier</h2>
            <p className="mt-0.5 text-[12px] text-muted">
              Un visage suivi d'une image à l'autre dans un même plan. Proposé d'office : le remplacer partout par la personne qui le
              remplace déjà la plupart du temps.
            </p>
          </div>
          <div className="flex-1 space-y-1 overflow-y-auto p-2">
            {extras.length > 0 && (
              <div className="mb-2 space-y-1 rounded-xl bg-bg/60 pb-1 ring-1 ring-line">
                {extras.map((x) => (
                  <IssueRow key={x.id} issue={x} review={r} jobId={job.id} persons={persons} people={people} choice={choices[x.id]} selected={selected === x.id} onSelect={() => select(x)} onChoose={(c) => choose(x.id, c)} />
                ))}
              </div>
            )}
            {!major.length && !minor.length && <p className="px-2 py-6 text-center text-sm text-muted">Aucun visage mal suivi. Clic sur un cadre de la vidéo pour corriger un visage quand même.</p>}
            {major.map((x) => (
              <IssueRow key={x.id} issue={x} review={r} jobId={job.id} persons={persons} people={people} choice={choices[x.id]} selected={selected === x.id} onSelect={() => select(x)} onChoose={(c) => choose(x.id, c)} />
            ))}
            {minor.length > 0 && (
              <div className="pt-2">
                <button onClick={() => setShowMinor((v) => !v)} className="w-full rounded-lg px-2 py-1.5 text-left text-[12px] text-muted hover:bg-raised">
                  {showMinor ? "▾" : "▸"} Autres visages non remplacés ({minor.length}) : petits, ou ne ressemblant à personne
                </button>
                {showMinor &&
                  minor.map((x) => (
                    <IssueRow key={x.id} issue={x} review={r} jobId={job.id} persons={persons} people={people} choice={choices[x.id]} selected={selected === x.id} onSelect={() => select(x)} onChoose={(c) => choose(x.id, c)} />
                  ))}
              </div>
            )}
          </div>
          <div className="flex items-center justify-between gap-3 border-t border-line px-4 py-3">
            <span className="text-[12px] text-muted">
              {decided.length ? `${decided.length} choix · ${toCompute ? `≈ ${toCompute} image${toCompute > 1 ? "s" : ""} à recalculer` : "rien à recalculer"}` : "Aucun choix"}
            </span>
            <Button variant="primary" size="sm" disabled={!decided.length} loading={apply.isPending} onClick={() => apply.mutate(decided)}>
              {toCompute ? "Appliquer" : "Enregistrer"}
            </Button>
          </div>
        </Card>
      </div>
    </div>
  );
}

/** Piste choisie d'un clic sur l'image (aucun problème signalé) décrite comme un problème, pour offrir les mêmes choix. */
function trackAsIssue(r: Review, track: number): ReviewIssue | null {
  const at = new Map<number, [number, number, number, number, number, number]>();
  r.boxes.forEach((faces, i) => {
    const f = faces.find((b) => b[4] === track);
    if (f) at.set(i, f);
  });
  if (!at.size) return null;
  const frames = [...at.keys()];
  const start = Math.min(...frames);
  const end = Math.max(...frames);
  const replaced: Record<string, number> = {};
  let unreplaced = 0;
  let thumb = start;
  const states: string[] = [];
  for (let i = start; i <= end; i++) {
    const f = at.get(i);
    const person = f && f[5] >= 0 ? r.mappings[f[5]]?.person ?? null : null;
    states.push(!f ? "gap" : person ?? "none");
    if (!f) continue;
    if (person) replaced[person] = (replaced[person] ?? 0) + 1;
    else unreplaced += 1;
    const g = at.get(thumb)!;
    if (f[3] - f[1] > g[3] - g[1]) thumb = i;
  }
  const bar: [number, number, string][] = [];
  states.forEach((s, k) => {
    const last = bar[bar.length - 1];
    if (last && last[2] === s) last[1] = k;
    else bar.push([k, k, s]);
  });
  const t = at.get(thumb)!;
  return {
    id: track,
    kind: "track",
    minor: false,
    start,
    end,
    frames: end - start + 1,
    replaced,
    unreplaced,
    undetected: end - start + 1 - at.size,
    height: Math.round((t[3] - t[1]) * r.height),
    suggest: null,
    preselect: null,
    thumb: { frame: thumb, box: [t[0] * r.width, t[1] * r.height, t[2] * r.width, t[3] * r.height] },
    bar,
  };
}

/** Images recalculées par un choix (estimation affichée avant d'appliquer). */
function framesFor(x: ReviewIssue | undefined, c: Choice): number {
  if (!x) return 0;
  const replaced = Object.values(x.replaced).reduce((a, b) => a + b, 0);
  if (c.action === "remove") return replaced;
  if (c.action === "assign") return x.unreplaced + x.undetected + replaced - (x.replaced[c.person!] ?? 0);
  return 0;
}

function personName(persons: Person[], pid: string): string {
  return persons.find((p) => p.id === pid)?.name ?? `Personne ${styleOf(persons, pid).letter}`;
}

function IssueRow({
  issue: x,
  review: r,
  jobId,
  persons,
  people,
  choice,
  selected,
  onSelect,
  onChoose,
}: {
  issue: ReviewIssue;
  review: Review;
  jobId: string;
  persons: Person[];
  people: string[];
  choice: Choice | undefined;
  selected: boolean;
  onSelect: () => void;
  onChoose: (c: Choice | null) => void;
}) {
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (selected) ref.current?.scrollIntoView({ block: "nearest", behavior: "smooth" });
  }, [selected]);
  const [x1, y1, x2, y2] = x.thumb.box.map(Math.round);
  const crop = `/api/jobs/${jobId}/review/crop.jpg?frame=${x.thumb.frame}&x1=${x1}&y1=${y1}&x2=${x2}&y2=${y2}`;
  const replaced = Object.values(x.replaced).reduce((a, b) => a + b, 0);
  const others = x.unreplaced + x.undetected;
  const detail =
    x.kind === "track"
      ? replaced
        ? `Remplacé sur ${replaced} image${replaced > 1 ? "s" : ""}${others ? `, d'origine sur ${others}` : ""}. Mauvais visage ? « Ne rien remplacer ».`
        : "Laissé d'origine sur toute la séquence."
      : x.kind === "lost"
      ? `Remplacé sur ${replaced} image${replaced > 1 ? "s" : ""}, d'origine sur ${others} (profil, flou, main devant…).`
      : x.kind === "mixed"
        ? `Remplacé tantôt par ${Object.keys(x.replaced).map((p) => personName(persons, p)).join(", tantôt par ")}.`
        : x.suggest
          ? `Ressemble à ${personName(persons, x.suggest)}. ${x.height} px de haut.`
          : `Ne ressemble à aucune personne choisie (figurant ?). ${x.height} px de haut.`;
  // La personne proposée d'abord.
  const order = x.suggest ? [x.suggest, ...people.filter((p) => p !== x.suggest)] : people;
  const is = (c: Choice) => choice?.action === c.action && (c.action !== "assign" || choice?.person === c.person);
  const toggle = (c: Choice) => onChoose(is(c) ? null : c);
  const fps = r.fps;

  return (
    <div
      ref={ref}
      onClick={onSelect}
      className={cx("cursor-pointer rounded-xl p-2.5 ring-1 transition-colors", selected ? "bg-raised ring-accent/50" : "ring-transparent hover:bg-raised/60")}
    >
      <div className="flex gap-3">
        <img src={crop} alt="" loading="lazy" className="size-12 shrink-0 rounded-lg bg-overlay object-cover ring-1 ring-line" />
        <div className="min-w-0 flex-1">
          <div className="flex items-baseline justify-between gap-2">
            <span className={cx("text-[13px] font-medium", x.minor ? "text-muted" : x.kind === "missed" ? "text-fg" : "text-warn")}>{KIND[x.kind]}</span>
            <span className="shrink-0 font-mono text-[11px] text-muted tabular">
              {timecode(x.start / fps, fps)} · {x.frames} img
            </span>
          </div>
          <IssueBar issue={x} persons={persons} />
          <p className="mt-1 text-[12px] leading-snug text-muted">{detail}</p>
        </div>
      </div>
      <div className="mt-2 flex flex-wrap gap-1.5" onClick={(e) => e.stopPropagation()}>
        {order.map((pid) => {
          const c: Choice = { action: "assign", person: pid };
          const style = styleOf(persons, pid);
          return (
            <Chip key={pid} active={is(c)} onClick={() => toggle(c)} title={`Remplacer ce visage par ${personName(persons, pid)} sur toute la piste`}>
              <PersonBadge style={style} />
              <span className="max-w-24 truncate">{personName(persons, pid)}</span>
              {pid === x.suggest && !is(c) && <span className="text-faint">· proposé</span>}
            </Chip>
          );
        })}
        {replaced > 0 && (
          <Chip active={is({ action: "remove" })} onClick={() => toggle({ action: "remove" })} title="Remettre le visage d'origine sur toute la piste (mauvais visage remplacé)">
            Ne rien remplacer
          </Chip>
        )}
        <Chip active={is({ action: "keep" })} onClick={() => toggle({ action: "keep" })} title="Ne rien changer et ne plus le signaler">
          Laisser tel quel
        </Chip>
      </div>
    </div>
  );
}

function Chip({ active, onClick, title, children }: { active: boolean; onClick: () => void; title?: string; children: React.ReactNode }) {
  return (
    <button
      type="button"
      title={title}
      onClick={onClick}
      className={cx(
        "inline-flex h-7 items-center gap-1.5 rounded-lg px-2 text-[12px] ring-1 transition-colors",
        active ? "bg-accent-soft text-fg ring-accent/60" : "bg-bg text-muted ring-line hover:text-fg hover:ring-line-strong",
      )}
    >
      {children}
    </button>
  );
}

/** Barre d'une piste : remplacé (couleur de la personne), d'origine (rouge), non détecté (hachuré). */
function IssueBar({ issue: x, persons }: { issue: ReviewIssue; persons: Person[] }) {
  return (
    <div className="mt-1.5 flex h-1.5 overflow-hidden rounded-full bg-overlay">
      {x.bar.map(([a, b, s]) => (
        <span
          key={a}
          style={{
            width: `${((b - a + 1) / x.frames) * 100}%`,
            background:
              s === "none"
                ? "var(--color-danger)"
                : s === "gap"
                  ? "repeating-linear-gradient(135deg, var(--color-danger) 0 2px, transparent 2px 4px)"
                  : styleOf(persons, s).color,
          }}
        />
      ))}
    </div>
  );
}

/* ───────────────────────── Lecteur image par image ───────────────────────── */

type PlayerHandle = { seek: (frame: number) => void };

type PlayerProps = {
  review: Review;
  persons: Person[];
  selected: number | null;
  onPickTrack: (track: number) => void;
  ref: React.Ref<PlayerHandle>;
};

function Player({ review: r, persons, selected, onPickTrack, ref }: PlayerProps) {
  const after = useRef<HTMLVideoElement>(null);
  const before = useRef<HTMLVideoElement>(null);
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [original, setOriginal] = useState(false);
  const [ready, setReady] = useState(false);
  const fps = r.fps;
  const last = r.frames - 1;

  const seek = useCallback(
    (i: number) => {
      const f = clamp(Math.round(i), 0, last);
      // Milieu de l'image voulue : le navigateur affiche bien celle-là, pas sa voisine.
      const t = (f + 0.5) / fps;
      for (const v of [after.current, before.current]) if (v) v.currentTime = t;
      setFrame(f);
    },
    [fps, last],
  );

  useImperativeHandle(ref, () => ({ seek }), [seek]);

  // Image affichée : rappel à chaque image présentée (précis à l'image), sinon position de lecture.
  useEffect(() => {
    const v = after.current;
    if (!v) return;
    if (typeof v.requestVideoFrameCallback === "function") {
      let handle = 0;
      const onFrame = (_: number, meta: VideoFrameCallbackMetadata) => {
        setFrame(clamp(Math.round(meta.mediaTime * fps), 0, last));
        handle = v.requestVideoFrameCallback(onFrame);
      };
      handle = v.requestVideoFrameCallback(onFrame);
      return () => v.cancelVideoFrameCallback(handle);
    }
    const onTime = () => setFrame(clamp(Math.floor(v.currentTime * fps + 1e-3), 0, last));
    v.addEventListener("timeupdate", onTime);
    v.addEventListener("seeked", onTime);
    return () => {
      v.removeEventListener("timeupdate", onTime);
      v.removeEventListener("seeked", onTime);
    };
  }, [fps, last, r.after_url]);

  // Lecture : la vidéo d'origine suit celle du rendu.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      const a = after.current;
      const b = before.current;
      if (a && b && Math.abs(b.currentTime - a.currentTime) > 0.08) b.currentTime = a.currentTime;
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  const toggle = useCallback(() => {
    const a = after.current;
    const b = before.current;
    if (!a || !b) return;
    if (a.paused) {
      b.currentTime = a.currentTime;
      a.play().catch(() => {});
      b.play().catch(() => {});
    } else {
      a.pause();
      b.pause();
      seek(frame); // arrêt net sur une image entière (cadres alignés)
    }
  }, [frame, seek]);

  const issueStarts = useMemo(() => [...new Set(r.issues.filter((x) => !x.minor).map((x) => x.thumb.frame))].sort((a, b) => a - b), [r.issues]);
  const jump = (dir: 1 | -1) => {
    const next = dir > 0 ? issueStarts.find((f) => f > frame) : [...issueStarts].reverse().find((f) => f < frame);
    if (next !== undefined) seek(next);
  };

  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest("input, textarea, select, [contenteditable]")) return;
      if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
        e.preventDefault();
        after.current?.pause();
        before.current?.pause();
        seek(frame + (e.key === "ArrowRight" ? 1 : -1) * (e.shiftKey ? Math.round(fps) : 1));
      } else if (e.key === " " && !el.closest("button")) {
        e.preventDefault();
        toggle();
      }
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [frame, fps, seek, toggle]);

  const issueOf = useMemo(() => new Map(r.issues.map((x) => [x.id, x])), [r.issues]);
  const boxes = r.boxes[frame] ?? [];

  return (
    <div className="flex flex-col gap-3">
      <div className="relative overflow-hidden rounded-[var(--radius-card)] bg-black ring-1 ring-line select-none" style={{ aspectRatio: `${r.width} / ${r.height}` }}>
        <video ref={before} src={r.before_url} muted playsInline preload="auto" className="absolute inset-0 size-full object-contain" style={{ opacity: original ? 1 : 0 }} />
        <video
          ref={after}
          key={r.after_url}
          src={r.after_url}
          muted
          playsInline
          preload="auto"
          onLoadedData={() => {
            setReady(true);
            seek(frame);
          }}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onEnded={() => before.current?.pause()}
          onClick={toggle}
          className="absolute inset-0 size-full object-contain"
          style={{ opacity: original ? 0 : 1 }}
        />
        <svg className="pointer-events-none absolute inset-0 size-full" viewBox="0 0 1 1" preserveAspectRatio="none">
          {boxes.map(([x1, y1, x2, y2, track, m], k) => {
            const issue = issueOf.get(track);
            const person = m >= 0 ? r.mappings[m]?.person : null;
            const color = person ? styleOf(persons, person).color : issue && !issue.minor ? "var(--color-danger)" : issue ? "var(--color-warn)" : "#9a9aa6";
            const isSelected = track === selected;
            return (
              <rect
                key={k}
                x={x1}
                y={y1}
                width={x2 - x1}
                height={y2 - y1}
                rx={0.004}
                fill="transparent"
                stroke={color}
                strokeWidth={isSelected ? 3 : person ? 1.5 : 1.5}
                strokeDasharray={person ? undefined : "5 4"}
                vectorEffect="non-scaling-stroke"
                pointerEvents="all"
                className="cursor-pointer"
                onClick={() => onPickTrack(track)}
                opacity={isSelected || selected === null ? 1 : 0.7}
              />
            );
          })}
        </svg>
        {!ready && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/40">
            <Loader2 className="size-6 animate-spin text-muted" />
          </div>
        )}
        <span className={cx("pointer-events-none absolute top-3 left-3 rounded-full px-2.5 py-1 text-[11px] font-medium backdrop-blur-md", original ? "bg-black/60" : "bg-accent text-accent-ink")}>
          {original ? "Original" : "Rendu (sans son)"}
        </span>
      </div>

      <div className="flex flex-wrap items-center gap-1.5">
        <Button variant="ghost" size="sm" className="w-8 px-0" onClick={() => jump(-1)} aria-label="Visage à vérifier précédent" title="Visage à vérifier précédent">
          <SkipBack className="size-4" />
        </Button>
        <Button variant="ghost" size="sm" className="w-8 px-0" onClick={() => seek(frame - 1)} aria-label="Image précédente" title="Image précédente (←)">
          <ChevronLeft className="size-4" />
        </Button>
        <Button variant="secondary" size="sm" className="w-10 px-0" onClick={toggle} aria-label={playing ? "Pause" : "Lecture"}>
          {playing ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
        </Button>
        <Button variant="ghost" size="sm" className="w-8 px-0" onClick={() => seek(frame + 1)} aria-label="Image suivante" title="Image suivante (→)">
          <ChevronRight className="size-4" />
        </Button>
        <Button variant="ghost" size="sm" className="w-8 px-0" onClick={() => jump(1)} aria-label="Visage à vérifier suivant" title="Visage à vérifier suivant">
          <SkipForward className="size-4" />
        </Button>
        <span className="ml-1 font-mono text-[13px] text-muted tabular">
          {timecode(frame / fps, fps)} <span className="text-faint">· image {frame + 1}/{r.frames}</span>
        </span>
        <Button
          variant={original ? "secondary" : "ghost"}
          size="sm"
          className="ml-auto"
          icon={<Eye className="size-4" />}
          onClick={() => setOriginal((v) => !v)}
          title="Voir la vidéo d'origine à la même image"
        >
          Original
        </Button>
      </div>
      <Timeline review={r} persons={persons} frame={frame} selected={selected} onSeek={seek} />
    </div>
  );
}

/** Frise : où chaque personne est remplacée, et les visages à vérifier (clic pour s'y rendre). */
function Timeline({ review: r, persons, frame, selected, onSeek }: { review: Review; persons: Person[]; frame: number; selected: number | null; onSeek: (f: number) => void }) {
  const box = useRef<HTMLDivElement>(null);
  const dragging = useRef(false);
  const rows = useMemo(() => {
    const people = [...new Set(r.mappings.map((m) => m.person).filter((p): p is string => !!p))];
    return people.map((pid) => {
      const runs: [number, number][] = [];
      r.boxes.forEach((faces, i) => {
        if (!faces.some((f) => f[5] >= 0 && r.mappings[f[5]]?.person === pid)) return;
        const lastRun = runs[runs.length - 1];
        if (lastRun && lastRun[1] === i - 1) lastRun[1] = i;
        else runs.push([i, i]);
      });
      return { pid, runs };
    });
  }, [r]);
  const pct = (i: number) => `${(i / r.frames) * 100}%`;
  const at = (clientX: number) => {
    const rect = box.current!.getBoundingClientRect();
    onSeek(((clientX - rect.left) / rect.width) * r.frames);
  };

  return (
    <div className="flex gap-2">
      <div className="flex w-5 shrink-0 flex-col gap-1 pt-0.5">
        {rows.map(({ pid }) => (
          <span key={pid} className="flex h-2.5 items-center" title={persons.find((p) => p.id === pid)?.name}>
            <span className="size-2.5 rounded-full" style={{ background: styleOf(persons, pid).color }} />
          </span>
        ))}
        <span className="h-3" />
      </div>
      <div
        ref={box}
        className="relative flex-1 cursor-pointer touch-none"
        onPointerDown={(e) => {
          dragging.current = true;
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          at(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && at(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
      >
        <div className="flex flex-col gap-1 pt-0.5">
          {rows.map(({ pid, runs }) => (
            <div key={pid} className="relative h-2.5 rounded-sm bg-overlay">
              {runs.map(([a, b]) => (
                <span key={a} className="absolute inset-y-0 rounded-[1px]" style={{ left: pct(a), width: pct(b - a + 1), background: styleOf(persons, pid).color, opacity: 0.85 }} />
              ))}
            </div>
          ))}
          <div className="relative h-3">
            {r.issues.map((x) => (
              <span
                key={x.id}
                title={`${KIND[x.kind]} · ${timecode(x.start / r.fps, r.fps)}`}
                className={cx("absolute top-0.5 h-2 min-w-[3px] rounded-sm", x.id === selected && "ring-2 ring-accent")}
                style={{ left: pct(x.start), width: pct(x.end - x.start + 1), background: x.minor ? "var(--color-faint)" : "var(--color-danger)" }}
              />
            ))}
          </div>
        </div>
        <span className="pointer-events-none absolute -top-1 -bottom-1 w-px bg-fg" style={{ left: pct(frame + 0.5) }} />
      </div>
    </div>
  );
}
