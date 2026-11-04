import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import {
  ArrowDown,
  ArrowUp,
  Clapperboard,
  Download,
  Eye,
  Gift,
  Loader2,
  Music,
  Plus,
  Smartphone,
  Sparkles,
  Square,
  Trash2,
  Tv,
  X,
} from "lucide-react";
import { useEffect, useMemo, useRef, useState } from "react";
import { api, type Job, type Montage, type MontageIn } from "../lib/api";
import { duration } from "../lib/time";
import { Button, Notice, ProgressBar, SegmentedControl, Switch, cx } from "./ui";

/** Réglages tout prêts : un clic pour l'ambiance, tout reste modifiable ensuite. */
const PRESETS: { id: string; label: string; icon: React.ReactNode; apply: Partial<MontageIn> }[] = [
  {
    id: "noel",
    label: "Noël en famille",
    icon: <Gift className="size-4" />,
    apply: { title: "Joyeux Noël la famille !", subtitle: "Les parodies de l'année", outro: "Joyeux Noël !", grade: "warm", snow: true, vignette: true, zoom: true, mask: "santa", transition: "fade" },
  },
  {
    id: "renne",
    label: "Rennes et nez rouges",
    icon: <Sparkles className="size-4" />,
    apply: { title: "La famille au complet", subtitle: "", outro: "Bonnes fêtes !", grade: "warm", snow: true, vignette: false, zoom: true, mask: "reindeer", transition: "flash" },
  },
  {
    id: "cine",
    label: "Bande-annonce",
    icon: <Clapperboard className="size-4" />,
    apply: { title: "Bientôt dans votre salon", subtitle: "", outro: "Fin", grade: "cinema", snow: false, vignette: true, zoom: true, mask: "none", transition: "fade" },
  },
];

const DEFAULT: Omit<MontageIn, "clips"> = {
  format: "9:16",
  fit: "blur",
  title: "",
  subtitle: "",
  outro: "",
  grade: "none",
  snow: false,
  vignette: false,
  zoom: false,
  mask: "none",
  transition: "fade",
  music_id: null,
  music_volume: 0.35,
  clip_audio: true,
  ai_label: true,
};

const MASKS: { value: MontageIn["mask"]; label: string }[] = [
  { value: "none", label: "Aucun" },
  { value: "santa", label: "Bonnet" },
  { value: "reindeer", label: "Renne" },
  { value: "nose", label: "Nez rouge" },
];

const GRADES: { value: MontageIn["grade"]; label: string }[] = [
  { value: "none", label: "Naturel" },
  { value: "warm", label: "Chaleureux" },
  { value: "cinema", label: "Ciné" },
  { value: "vintage", label: "Vintage" },
  { value: "bw", label: "N&B" },
];

const STATUS: Record<Montage["status"], string> = {
  queued: "En attente",
  running: "En cours",
  cancelling: "Annulation…",
  cancelled: "Annulé",
  done: "Prêt",
  error: "Erreur",
};

interface Item {
  job: Job;
  caption: string;
}

export function MontageStudio({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const jobs = useQuery({ queryKey: ["jobs"], queryFn: api.jobs, enabled: open, refetchInterval: open ? 5000 : false });
  const montages = useQuery({
    queryKey: ["montages"],
    queryFn: api.montages,
    enabled: open,
    refetchInterval: (q) => (open && q.state.data?.some((m) => ["queued", "running", "cancelling"].includes(m.status)) ? 1500 : open ? 8000 : false),
  });
  const [items, setItems] = useState<Item[]>([]);
  const [opts, setOpts] = useState(DEFAULT);
  const [musicName, setMusicName] = useState<string | null>(null);
  const [musicProgress, setMusicProgress] = useState<number | null>(null);
  const [preview, setPreview] = useState<string | null>(null);
  const [watching, setWatching] = useState<string | null>(null);
  const musicInput = useRef<HTMLInputElement>(null);

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const finished = useMemo(() => (jobs.data ?? []).filter((j) => j.result_url), [jobs.data]);
  const chosen = new Set(items.map((i) => i.job.id));
  const body = (): MontageIn => ({ ...opts, clips: items.map((i) => ({ job_id: i.job.id, caption: i.caption })) });
  const set = <K extends keyof typeof opts>(k: K, v: (typeof opts)[K]) => {
    setOpts((o) => ({ ...o, [k]: v }));
    setPreview(null);
  };

  const create = useMutation({
    mutationFn: () => api.createMontage(body()),
    onSuccess: (m) => {
      qc.invalidateQueries({ queryKey: ["montages"] });
      setWatching(m.id);
    },
  });
  const look = useMutation({ mutationFn: () => api.previewMontage(body()), onSuccess: (r) => setPreview(r.image) });
  const cancel = useMutation({ mutationFn: api.cancelMontage, onSuccess: () => qc.invalidateQueries({ queryKey: ["montages"] }) });
  const remove = useMutation({ mutationFn: api.deleteMontage, onSuccess: () => qc.invalidateQueries({ queryKey: ["montages"] }) });
  const music = useMutation({
    mutationFn: (f: File) => api.uploadMusic(f, setMusicProgress),
    onSuccess: (r) => {
      set("music_id", r.id);
      setMusicName(r.name);
      setMusicProgress(null);
    },
    onError: () => setMusicProgress(null),
  });

  const move = (i: number, d: number) =>
    setItems((xs) => {
      const j = i + d;
      if (j < 0 || j >= xs.length) return xs;
      const out = [...xs];
      [out[i], out[j]] = [out[j], out[i]];
      return out;
    });
  const total = items.reduce((s, i) => s + (i.job.params.end - i.job.params.start), 0) + (opts.title.trim() ? 3 : 0) + (opts.outro.trim() ? 2.5 : 0);
  const watched = montages.data?.find((m) => m.id === watching) ?? null;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-40 bg-black/60 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.section
            role="dialog"
            aria-label="Montage"
            aria-modal="true"
            className="fixed inset-x-0 top-0 bottom-0 z-50 mx-auto flex max-w-6xl flex-col bg-surface shadow-2xl ring-1 ring-line sm:inset-y-6 sm:rounded-2xl"
            initial={{ opacity: 0, y: 24 }}
            animate={{ opacity: 1, y: 0 }}
            exit={{ opacity: 0, y: 24 }}
            transition={{ type: "spring", stiffness: 380, damping: 36 }}
          >
            <header className="flex items-center justify-between gap-3 border-b border-line px-5 py-4">
              <div>
                <h2 className="flex items-center gap-2 font-medium">
                  <Clapperboard className="size-4.5 text-accent" /> Montage
                </h2>
                <p className="mt-0.5 text-[13px] text-muted">Enchaîne tes rendus en une seule vidéo à partager : vertical pour TikTok, Reels et Shorts.</p>
              </div>
              <Button variant="ghost" size="sm" onClick={onClose} aria-label="Fermer" className="w-8 px-0">
                <X className="size-4" />
              </Button>
            </header>

            <div className="grid flex-1 grid-cols-1 gap-5 overflow-x-hidden overflow-y-auto p-4 sm:p-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
              {/* Séquences */}
              <div className="min-w-0 space-y-4">
                <div>
                  <h3 className="mb-2 text-[12px] font-medium tracking-wide text-faint uppercase">Séquences du montage</h3>
                  {!items.length && (
                    <p className="rounded-xl bg-raised px-4 py-6 text-center text-sm text-muted ring-1 ring-line">
                      Ajoute des rendus terminés ci-dessous, dans l'ordre voulu.
                    </p>
                  )}
                  <ol className="space-y-2">
                    {items.map((it, i) => (
                      <li key={it.job.id} className="flex items-center gap-3 rounded-xl bg-raised p-2.5 ring-1 ring-line">
                        <span className="w-5 text-center font-mono text-[12px] text-faint">{i + 1}</span>
                        <video src={`${it.job.result_url}#t=0.5`} preload="metadata" muted className="aspect-video w-20 shrink-0 rounded-md bg-black object-cover" />
                        <div className="min-w-0 flex-1">
                          <div className="truncate text-[13px] text-muted">{it.job.video_title ?? "Rendu"}</div>
                          <input
                            value={it.caption}
                            maxLength={120}
                            onChange={(e) => {
                              const v = e.target.value;
                              setItems((xs) => xs.map((x, k) => (k === i ? { ...x, caption: v } : x)));
                              setPreview(null);
                            }}
                            placeholder="Légende (ex. Papa en Rocky)"
                            aria-label={`Légende de la séquence ${i + 1}`}
                            className="mt-1 w-full rounded-lg bg-bg px-2.5 py-1.5 text-sm ring-1 ring-line outline-none placeholder:text-faint focus:ring-accent"
                          />
                        </div>
                        <div className="flex flex-col">
                          <button onClick={() => move(i, -1)} disabled={i === 0} aria-label="Monter" className="rounded p-1 text-muted hover:text-fg disabled:opacity-30">
                            <ArrowUp className="size-3.5" />
                          </button>
                          <button onClick={() => move(i, 1)} disabled={i === items.length - 1} aria-label="Descendre" className="rounded p-1 text-muted hover:text-fg disabled:opacity-30">
                            <ArrowDown className="size-3.5" />
                          </button>
                        </div>
                        <button onClick={() => setItems((xs) => xs.filter((_, k) => k !== i))} aria-label="Retirer" className="rounded p-1.5 text-muted hover:text-danger">
                          <X className="size-4" />
                        </button>
                      </li>
                    ))}
                  </ol>
                </div>

                <div>
                  <h3 className="mb-2 text-[12px] font-medium tracking-wide text-faint uppercase">Rendus terminés</h3>
                  {jobs.isLoading && <Loader2 className="mx-auto my-6 size-5 animate-spin text-muted" />}
                  {jobs.data && !finished.length && <p className="text-sm text-muted">Aucun rendu terminé pour l'instant : fais d'abord quelques parodies.</p>}
                  <div className="grid grid-cols-2 gap-2 sm:grid-cols-3">
                    {finished.map((j) => (
                      <button
                        key={j.id}
                        onClick={() => {
                          setItems((xs) => [...xs, { job: j, caption: "" }]);
                          setPreview(null);
                        }}
                        disabled={chosen.has(j.id)}
                        className="group relative overflow-hidden rounded-xl bg-raised text-left ring-1 ring-line transition hover:ring-accent disabled:opacity-40 disabled:hover:ring-line"
                      >
                        <video src={`${j.result_url}#t=0.5`} preload="metadata" muted className="aspect-video w-full bg-black object-cover" />
                        <div className="flex items-center justify-between gap-2 px-2 py-1.5">
                          <span className="truncate text-[12px]">{j.video_title ?? "Rendu"}</span>
                          <span className="shrink-0 font-mono text-[11px] text-faint">{duration(j.params.end - j.params.start)}</span>
                        </div>
                        {!chosen.has(j.id) && (
                          <span className="absolute top-1.5 right-1.5 flex size-6 items-center justify-center rounded-full bg-accent text-accent-ink opacity-0 transition group-hover:opacity-100">
                            <Plus className="size-4" />
                          </span>
                        )}
                      </button>
                    ))}
                  </div>
                </div>
              </div>

              {/* Réglages */}
              <div className="min-w-0 space-y-4">
                <div>
                  <h3 className="mb-2 text-[12px] font-medium tracking-wide text-faint uppercase">Ambiance en un clic</h3>
                  <div className="flex flex-wrap gap-2">
                    {PRESETS.map((p) => (
                      <Button
                        key={p.id}
                        variant="secondary"
                        size="sm"
                        icon={p.icon}
                        onClick={() => {
                          setOpts((o) => ({ ...o, ...p.apply }));
                          setPreview(null);
                        }}
                      >
                        {p.label}
                      </Button>
                    ))}
                  </div>
                </div>

                <div className="grid gap-3 sm:grid-cols-2 lg:grid-cols-1 xl:grid-cols-2">
                  <Field label="Format">
                    <SegmentedControl
                      value={opts.format}
                      onChange={(v) => set("format", v)}
                      className="w-full"
                      options={[
                        { value: "9:16", label: "9:16", icon: <Smartphone className="size-3.5" /> },
                        { value: "1:1", label: "1:1", icon: <Square className="size-3.5" /> },
                        { value: "16:9", label: "16:9", icon: <Tv className="size-3.5" /> },
                      ]}
                    />
                  </Field>
                  <Field label="Cadrage">
                    <SegmentedControl
                      value={opts.fit}
                      onChange={(v) => set("fit", v)}
                      className="w-full"
                      options={[
                        { value: "blur", label: "Fond flouté" },
                        { value: "crop", label: "Plein cadre" },
                      ]}
                    />
                  </Field>
                </div>
                <p className="-mt-2 text-[12px] text-faint">Plein cadre : l'image remplit l'écran et suit les visages.</p>

                <Field label="Masque sur les visages">
                  <Chips value={opts.mask} options={MASKS} onChange={(v) => set("mask", v)} />
                </Field>
                <Field label="Filtre">
                  <Chips value={opts.grade} options={GRADES} onChange={(v) => set("grade", v)} />
                </Field>
                <div className="grid gap-x-6 sm:grid-cols-2">
                  <Switch checked={opts.snow} onChange={(v) => set("snow", v)} label="Neige" />
                  <Switch checked={opts.zoom} onChange={(v) => set("zoom", v)} label="Zoom lent" />
                  <Switch checked={opts.vignette} onChange={(v) => set("vignette", v)} label="Vignettage" />
                  <Switch checked={opts.ai_label} onChange={(v) => set("ai_label", v)} label="Étiquette « IA »" />
                </div>
                <Field label="Transition">
                  <Chips
                    value={opts.transition}
                    onChange={(v) => set("transition", v)}
                    options={[
                      { value: "fade", label: "Fondu" },
                      { value: "flash", label: "Flash" },
                      { value: "cut", label: "Coupe sèche" },
                    ]}
                  />
                </Field>

                <div className="grid gap-3 sm:grid-cols-2">
                  <Field label="Titre d'ouverture">
                    <TextInput value={opts.title} onChange={(v) => set("title", v)} placeholder="Joyeux Noël la famille !" max={80} />
                  </Field>
                  <Field label="Sous-titre">
                    <TextInput value={opts.subtitle} onChange={(v) => set("subtitle", v)} placeholder="Les parodies 2026" max={120} />
                  </Field>
                </div>
                <Field label="Message de fin">
                  <TextInput value={opts.outro} onChange={(v) => set("outro", v)} placeholder="Joyeux Noël !" max={80} />
                </Field>

                <Field label="Musique de fond">
                  <div className="flex flex-wrap items-center gap-2">
                    <input
                      ref={musicInput}
                      type="file"
                      accept=".mp3,.m4a,.aac,.wav,.ogg,.flac,audio/*"
                      className="hidden"
                      onChange={(e) => {
                        const f = e.target.files?.[0];
                        if (f) music.mutate(f);
                        e.target.value = "";
                      }}
                    />
                    <Button variant="secondary" size="sm" icon={<Music className="size-4" />} loading={music.isPending} onClick={() => musicInput.current?.click()}>
                      {musicName ?? "Choisir un fichier"}
                    </Button>
                    {opts.music_id && (
                      <Button variant="ghost" size="sm" onClick={() => { set("music_id", null); setMusicName(null); }}>
                        Retirer
                      </Button>
                    )}
                    {musicProgress !== null && <ProgressBar value={musicProgress} className="w-24" />}
                  </div>
                  {opts.music_id && (
                    <div className="mt-2 space-y-1">
                      <label className="flex items-center gap-3 text-[13px] text-muted">
                        Volume
                        <input
                          type="range"
                          min={0}
                          max={1}
                          step={0.05}
                          value={opts.music_volume}
                          onChange={(e) => set("music_volume", Number(e.target.value))}
                          className="flex-1 accent-[var(--color-accent)]"
                        />
                      </label>
                      <Switch
                        checked={opts.clip_audio}
                        onChange={(v) => set("clip_audio", v)}
                        label="Garder les voix des séquences"
                        description="La musique baisse toute seule quand quelqu'un parle."
                      />
                    </div>
                  )}
                  {music.error && <Notice tone="danger" className="mt-2">{music.error.message}</Notice>}
                </Field>

                <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line pt-4">
                  <div className="text-[13px] text-muted">
                    {items.length} séquence{items.length > 1 ? "s" : ""} · ≈ {duration(total)}
                  </div>
                  <div className="flex gap-2">
                    <Button variant="secondary" icon={<Eye className="size-4" />} disabled={!items.length} loading={look.isPending} onClick={() => look.mutate()}>
                      Aperçu
                    </Button>
                    <Button variant="primary" icon={<Clapperboard className="size-4" />} disabled={!items.length} loading={create.isPending} onClick={() => create.mutate()}>
                      Créer le montage
                    </Button>
                  </div>
                </div>
                {(create.error || look.error) && <Notice tone="danger">{(create.error ?? look.error)!.message}</Notice>}
                {preview && (
                  <div className="flex justify-center rounded-xl bg-bg p-3 ring-1 ring-line">
                    <img src={preview} alt="Aperçu du montage" className="max-h-[420px] rounded-lg" />
                  </div>
                )}

                {watched?.status === "done" && watched.result_url && (
                  <div className="space-y-2 rounded-xl bg-bg p-3 ring-1 ring-accent/40">
                    <video src={watched.result_url} controls playsInline className="mx-auto max-h-[480px] rounded-lg bg-black" />
                  </div>
                )}

                <MontageList montages={montages.data ?? []} onWatch={setWatching} onCancel={(id) => cancel.mutate(id)} onDelete={(id) => remove.mutate(id)} />
              </div>
            </div>
          </motion.section>
        </>
      )}
    </AnimatePresence>
  );
}

function MontageList({ montages, onWatch, onCancel, onDelete }: { montages: Montage[]; onWatch: (id: string) => void; onCancel: (id: string) => void; onDelete: (id: string) => void }) {
  if (!montages.length) return null;
  return (
    <div>
      <h3 className="mb-2 text-[12px] font-medium tracking-wide text-faint uppercase">Mes montages</h3>
      <ul className="space-y-2">
        {montages.map((m) => {
          const busy = ["queued", "running", "cancelling"].includes(m.status);
          return (
            <li key={m.id} className="rounded-xl bg-raised p-3 ring-1 ring-line">
              <div className="flex items-center gap-3">
                <div className="min-w-0 flex-1">
                  <div className="truncate text-sm font-medium">{m.title}</div>
                  <div className={cx("text-[12px]", m.status === "error" ? "text-danger" : "text-muted")}>
                    {STATUS[m.status]}
                    {m.status === "running" && ` · ${Math.round(m.progress * 100)} %`}
                    {m.info && ` · ${duration(m.info.duration)} · ${m.info.width}×${m.info.height}`}
                  </div>
                </div>
                {m.status === "done" && (
                  <>
                    <Button variant="ghost" size="sm" icon={<Eye className="size-4" />} onClick={() => onWatch(m.id)}>
                      Voir
                    </Button>
                    <a href={m.download_url!} className="inline-flex h-8 items-center gap-1.5 rounded-lg bg-accent px-3 text-[13px] font-medium text-accent-ink">
                      <Download className="size-4" /> MP4
                    </a>
                  </>
                )}
                {busy ? (
                  <Button variant="ghost" size="sm" onClick={() => onCancel(m.id)} disabled={m.status === "cancelling"}>
                    Annuler
                  </Button>
                ) : (
                  <button onClick={() => onDelete(m.id)} aria-label="Supprimer le montage" className="rounded p-1.5 text-muted hover:text-danger">
                    <Trash2 className="size-4" />
                  </button>
                )}
              </div>
              {m.status === "running" && <ProgressBar value={m.progress} className="mt-2" />}
              {m.error && <p className="mt-1 text-[12px] text-danger">{m.error}</p>}
            </li>
          );
        })}
      </ul>
    </div>
  );
}

function Field({ label, children }: { label: string; children: React.ReactNode }) {
  return (
    <div>
      <div className="mb-1.5 text-[13px] font-medium">{label}</div>
      {children}
    </div>
  );
}

function TextInput({ value, onChange, placeholder, max }: { value: string; onChange: (v: string) => void; placeholder: string; max: number }) {
  return (
    <input
      value={value}
      maxLength={max}
      onChange={(e) => onChange(e.target.value)}
      placeholder={placeholder}
      className="w-full rounded-lg bg-bg px-3 py-2 text-sm ring-1 ring-line outline-none placeholder:text-faint focus:ring-accent"
    />
  );
}

function Chips<T extends string>({ value, options, onChange }: { value: T; options: { value: T; label: string }[]; onChange: (v: T) => void }) {
  return (
    <div role="radiogroup" className="flex flex-wrap gap-1.5">
      {options.map((o) => (
        <button
          key={o.value}
          role="radio"
          aria-checked={value === o.value}
          onClick={() => onChange(o.value)}
          className={cx(
            "rounded-full px-3 py-1.5 text-[13px] font-medium ring-1 transition-colors",
            value === o.value ? "bg-accent-soft text-accent ring-accent/50" : "text-muted ring-line hover:text-fg hover:ring-line-strong",
          )}
        >
          {o.label}
        </button>
      ))}
    </div>
  );
}
