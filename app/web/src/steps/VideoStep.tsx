import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { AlertCircle, ArrowRight, Clock, Film, Link2, Sparkles, Trash2, UploadCloud, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type Status, type UrlInfo, type Video } from "../lib/api";
import { duration } from "../lib/time";
import { Button, Card, Notice, ProgressBar, SectionTitle, SegmentedControl, cx } from "../components/ui";
import { QualityBadge } from "../components/QualityBadge";

type Props = { status?: Status; onReady: (v: Video) => void };

export function VideoStep({ status, onReady }: Props) {
  const [tab, setTab] = useState<"url" | "file">("url");
  const [pendingId, setPendingId] = useState<string | null>(null);
  const [uploadFrac, setUploadFrac] = useState<number | null>(null);
  const qc = useQueryClient();

  const pending = useQuery({
    queryKey: ["video", pendingId],
    queryFn: () => api.video(pendingId!),
    enabled: !!pendingId,
    refetchInterval: (q) => (q.state.data && ["ready", "error"].includes(q.state.data.status) ? false : 800),
  });

  useEffect(() => {
    if (pending.data?.status === "ready") {
      qc.invalidateQueries({ queryKey: ["videos"] });
      onReady(pending.data);
      setPendingId(null);
    }
  }, [pending.data, onReady, qc]);

  const upload = useMutation({
    mutationFn: (file: File) => api.uploadVideo(file, setUploadFrac),
    onSuccess: (v) => {
      setUploadFrac(null);
      setPendingId(v.id);
    },
    onError: () => setUploadFrac(null),
  });

  const busy = !!pendingId || upload.isPending;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 1 · Vidéo"
        title="D'où vient la vidéo ?"
        subtitle={`Colle un lien YouTube ou importe un fichier. Tu choisiras le passage de ${status?.segment.min_s ?? 5} à ${status?.segment.max_s ?? 60} s juste après.`}
      />

      <AnimatePresence mode="wait">
        {busy ? (
          <motion.div key="busy" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <PendingPanel
              video={pending.data}
              uploadFrac={uploadFrac}
              onCancel={() => {
                setPendingId(null);
                upload.reset();
              }}
            />
          </motion.div>
        ) : (
          <motion.div key="pick" initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
            <SegmentedControl
              value={tab}
              onChange={setTab}
              className="mb-5 w-full max-w-md"
              options={[
                { value: "url", label: "Lien YouTube", icon: <Link2 className="size-4" /> },
                { value: "file", label: "Fichier", icon: <UploadCloud className="size-4" /> },
              ]}
            />
            {pending.data?.status === "error" && (
              <Notice tone="danger" className="mb-5">
                {pending.data.error}
              </Notice>
            )}
            {upload.error && (
              <Notice tone="danger" className="mb-5">
                {upload.error.message}
              </Notice>
            )}
            {tab === "url" ? (
              <UrlPanel exampleUrl={status?.example_url} onStarted={(v) => setPendingId(v.id)} />
            ) : (
              <DropZone status={status} onFile={(f) => upload.mutate(f)} />
            )}
            <RecentVideos onPick={onReady} />
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
}

function UrlPanel({ exampleUrl, onStarted }: { exampleUrl?: string; onStarted: (v: Video) => void }) {
  const [url, setUrl] = useState("");
  const inputRef = useRef<HTMLInputElement>(null);
  const info = useMutation({ mutationFn: (u: string) => api.urlInfo(u) });
  const start = useMutation({ mutationFn: (u: string) => api.fromUrl(u), onSuccess: onStarted });

  const lookup = (u: string) => {
    const clean = u.trim();
    if (/^https?:\/\/\S+$/i.test(clean)) info.mutate(clean);
  };

  useEffect(() => inputRef.current?.focus(), []);

  return (
    <Card className="p-5 sm:p-6">
      <form
        className="flex flex-col gap-3 sm:flex-row"
        onSubmit={(e) => {
          e.preventDefault();
          lookup(url);
        }}
      >
        <div className="relative flex-1">
          <Link2 className="pointer-events-none absolute top-1/2 left-4 size-5 -translate-y-1/2 text-faint" />
          <input
            ref={inputRef}
            value={url}
            onChange={(e) => {
              setUrl(e.target.value);
              info.reset();
            }}
            onPaste={(e) => {
              const pasted = e.clipboardData.getData("text");
              setTimeout(() => lookup(pasted), 0);
            }}
            placeholder="https://www.youtube.com/watch?v=…"
            aria-label="Lien de la vidéo"
            spellCheck={false}
            className="h-12 w-full rounded-xl bg-bg pr-4 pl-12 text-[15px] ring-1 ring-line transition-shadow outline-none placeholder:text-faint focus:ring-2 focus:ring-accent"
          />
        </div>
        <Button type="submit" size="lg" variant="secondary" loading={info.isPending} disabled={!url.trim()}>
          Aperçu
        </Button>
      </form>

      {exampleUrl && !info.data && (
        <div className="mt-4 flex flex-wrap items-center gap-2 text-[13px] text-muted">
          <Sparkles className="size-4 text-accent" />
          Pour tester :
          <button
            onClick={() => {
              setUrl(exampleUrl);
              info.mutate(exampleUrl);
            }}
            className="rounded-full bg-accent-soft px-3 py-1 font-mono text-[12px] text-accent ring-1 ring-accent/30 transition-colors hover:bg-accent/20"
          >
            {exampleUrl.replace(/^https?:\/\/(www\.)?/, "")}
          </button>
        </div>
      )}

      {info.error && (
        <Notice tone="danger" className="mt-4">
          {info.error.message}
        </Notice>
      )}

      <AnimatePresence>
        {info.data && (
          <motion.div initial={{ opacity: 0, height: 0 }} animate={{ opacity: 1, height: "auto" }} exit={{ opacity: 0, height: 0 }}>
            <UrlPreview data={info.data} loading={start.isPending} onUse={() => start.mutate(info.data!.webpage_url)} />
            {start.error && (
              <Notice tone="danger" className="mt-3">
                {start.error.message}
              </Notice>
            )}
          </motion.div>
        )}
      </AnimatePresence>
    </Card>
  );
}

function UrlPreview({ data, loading, onUse }: { data: UrlInfo; loading: boolean; onUse: () => void }) {
  return (
    <div className="mt-5 flex flex-col gap-4 rounded-xl bg-raised p-3 ring-1 ring-line sm:flex-row sm:items-center">
      <div className="relative aspect-video w-full shrink-0 overflow-hidden rounded-lg bg-black sm:w-56">
        {data.thumbnail && <img src={data.thumbnail} alt="" className="size-full object-cover" />}
        <span className="absolute right-2 bottom-2 rounded-md bg-black/75 px-1.5 py-0.5 font-mono text-[11px] tabular">
          {duration(data.duration)}
        </span>
      </div>
      <div className="min-w-0 flex-1">
        <div className="line-clamp-2 font-medium text-pretty">{data.title}</div>
        {data.uploader && <div className="mt-1 text-[13px] text-muted">{data.uploader}</div>}
        {data.too_long && (
          <div className="mt-2 flex items-center gap-1.5 text-[13px] text-warn">
            <AlertCircle className="size-4" /> Trop longue : {duration(data.max_duration_s)} maximum.
          </div>
        )}
      </div>
      <Button variant="primary" size="lg" loading={loading} disabled={data.too_long} onClick={onUse} icon={<ArrowRight className="size-4" />} className="sm:self-center">
        Utiliser
      </Button>
    </div>
  );
}

function DropZone({ status, onFile }: { status?: Status; onFile: (f: File) => void }) {
  const [over, setOver] = useState(false);
  const input = useRef<HTMLInputElement>(null);
  const exts = status?.video_ext ?? [".mp4", ".mov", ".webm", ".mkv"];
  return (
    <button
      onClick={() => input.current?.click()}
      onDragOver={(e) => {
        e.preventDefault();
        setOver(true);
      }}
      onDragLeave={() => setOver(false)}
      onDrop={(e) => {
        e.preventDefault();
        setOver(false);
        const f = e.dataTransfer.files[0];
        if (f) onFile(f);
      }}
      className={cx(
        "checker group flex w-full flex-col items-center justify-center gap-3 rounded-[var(--radius-card)] border-2 border-dashed px-6 py-16 text-center transition-colors",
        over ? "border-accent bg-accent-soft" : "border-line-strong bg-surface hover:border-faint",
      )}
    >
      <div className={cx("flex size-14 items-center justify-center rounded-2xl ring-1 transition-colors", over ? "bg-accent text-accent-ink ring-accent" : "bg-raised text-muted ring-line group-hover:text-fg")}>
        <UploadCloud className="size-6" />
      </div>
      <div className="text-[15px] font-medium">
        Glisse ta vidéo ici <span className="text-muted">ou</span> <span className="text-accent underline-offset-4 group-hover:underline">parcours tes fichiers</span>
      </div>
      <div className="font-mono text-[12px] text-faint uppercase">
        {exts.map((e) => e.slice(1)).join(" · ")} — {status?.upload_max_mb ?? 500} Mo max
      </div>
      <input
        ref={input}
        type="file"
        accept={["video/*", ...exts].join(",")}
        className="hidden"
        onChange={(e) => {
          const f = e.target.files?.[0];
          if (f) onFile(f);
          e.target.value = "";
        }}
      />
    </button>
  );
}

function PendingPanel({ video, uploadFrac, onCancel }: { video?: Video; uploadFrac: number | null; onCancel: () => void }) {
  let label = "Envoi du fichier";
  let frac = uploadFrac ?? 0;
  let hint = "Ne ferme pas la page pendant l'envoi.";
  if (video) {
    frac = video.progress;
    if (video.status === "downloading") {
      label = "Téléchargement depuis YouTube";
      hint = "yt-dlp récupère la vidéo en 720p maximum.";
    } else {
      label = "Préparation du visionneur";
      hint = "Conversion en H.264 et génération des vignettes de la timeline.";
    }
  }
  return (
    <Card className="flex flex-col gap-5 p-6 sm:p-8">
      <div className="flex items-start gap-4">
        <div className="flex size-12 shrink-0 items-center justify-center rounded-xl bg-accent-soft text-accent ring-1 ring-accent/30">
          <Film className="size-5 animate-pulse" />
        </div>
        <div className="min-w-0 flex-1">
          <div className="truncate font-medium">{video?.title ?? "Nouvelle vidéo"}</div>
          <div className="mt-0.5 text-[13px] text-muted">{hint}</div>
        </div>
        <Button variant="ghost" size="sm" onClick={onCancel} aria-label="Fermer" className="w-8 px-0">
          <X className="size-4" />
        </Button>
      </div>
      <div>
        <div className="mb-2 flex items-baseline justify-between text-sm">
          <span>{label}</span>
          <span className="font-mono text-muted tabular">{Math.round(frac * 100)} %</span>
        </div>
        <ProgressBar value={frac} />
      </div>
    </Card>
  );
}

function RecentVideos({ onPick }: { onPick: (v: Video) => void }) {
  const qc = useQueryClient();
  const { data } = useQuery({ queryKey: ["videos"], queryFn: api.videos });
  const del = useMutation({
    mutationFn: api.deleteVideo,
    onSuccess: () => qc.invalidateQueries({ queryKey: ["videos"] }),
  });
  const ready = (data ?? []).filter((v) => v.status === "ready").slice(0, 6);
  if (!ready.length) return null;
  return (
    <div className="mt-10">
      <div className="mb-3 flex items-center gap-2 text-[13px] font-medium text-muted">
        <Clock className="size-4" /> Vidéos récentes
      </div>
      <div className="grid grid-cols-2 gap-3 sm:grid-cols-3">
        {ready.map((v) => (
          <div key={v.id} className="group relative overflow-hidden rounded-xl bg-surface ring-1 ring-line transition-shadow hover:ring-line-strong">
            <button onClick={() => onPick(v)} className="block w-full text-left">
              <div className="relative aspect-video bg-black">
                {v.poster_url && <img src={v.poster_url} alt="" className="size-full object-cover opacity-90 transition-opacity group-hover:opacity-100" />}
                <span className="absolute right-2 bottom-2 rounded-md bg-black/75 px-1.5 py-0.5 font-mono text-[11px] tabular">
                  {duration(v.info?.duration ?? 0)}
                </span>
                {v.info && <QualityBadge info={v.info} compact className="absolute bottom-2 left-2" />}
              </div>
              <div className="truncate px-3 py-2.5 text-[13px]">{v.title}</div>
            </button>
            <button
              onClick={() => del.mutate(v.id)}
              aria-label="Supprimer la vidéo"
              className="absolute top-2 right-2 flex size-7 items-center justify-center rounded-lg bg-black/60 text-fg/80 opacity-0 backdrop-blur transition-opacity group-hover:opacity-100 hover:text-danger focus-visible:opacity-100"
            >
              <Trash2 className="size-3.5" />
            </button>
          </div>
        ))}
      </div>
    </div>
  );
}
