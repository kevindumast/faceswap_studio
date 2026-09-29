import { AnimatePresence, motion } from "motion/react";
import {
  ChevronLeft,
  ChevronRight,
  Keyboard,
  Maximize2,
  Pause,
  Play,
  Repeat,
  Volume2,
  VolumeX,
  ZoomIn,
  ZoomOut,
} from "lucide-react";
import { useCallback, useEffect, useRef, useState, type ReactNode } from "react";
import type { Video } from "../../lib/api";
import { clamp, timecode } from "../../lib/time";
import { Button, Kbd, cx } from "../ui";
import { TimeField } from "./TimeField";
import { Timeline, type DragMode, type LimitHit } from "./Timeline";

export type Selection = { start: number; end: number };

type Props = {
  video: Video;
  selection: Selection;
  onChange: (s: Selection) => void;
  min: number;
  max: number;
  /** Affiché en haut à droite du lecteur (ex. qualité de la vidéo). */
  badge?: ReactNode;
};

const PRESETS = [5, 10, 15, 30, 60];

export function Trimmer({ video, selection, onChange, min, max, badge }: Props) {
  const info = video.info!;
  const fps = info.fps;
  const dur = info.duration;
  const { start, end } = selection;

  const videoRef = useRef<HTMLVideoElement>(null);
  const [current, setCurrent] = useState(start);
  const [playing, setPlaying] = useState(false);
  const [muted, setMuted] = useState(false);
  const [rate, setRate] = useState(1);
  const [loop, setLoop] = useState(true);
  const [zoom, setZoom] = useState(1);
  const [viewStart, setViewStart] = useState(0);
  const [limitHit, setLimitHit] = useState<LimitHit>(null);
  const [help, setHelp] = useState(false);
  const [toast, setToast] = useState<string | null>(null);
  const dragMode = useRef<DragMode | null>(null);
  const sel = useRef(selection);
  sel.current = selection;
  const loopRef = useRef(loop);
  loopRef.current = loop;

  const snap = useCallback((t: number) => Math.round(t * fps) / fps, [fps]);

  const seek = useCallback(
    (t: number) => {
      const v = videoRef.current;
      const target = clamp(t, 0, Math.max(0, dur - 1 / fps));
      if (v) v.currentTime = target;
      setCurrent(target);
    },
    [dur, fps],
  );

  const toastTimer = useRef<number | undefined>(undefined);
  const flash = useCallback((msg: string) => {
    setToast(msg);
    window.clearTimeout(toastTimer.current);
    toastTimer.current = window.setTimeout(() => setToast(null), 1600);
  }, []);

  /** Recale un passage dans [0, durée] et [min, max] en gardant fixe le côté demandé. */
  const fit = useCallback(
    (s: number, e: number, keep: "start" | "end"): Selection => {
      const len = clamp(e - s, min, Math.min(max, dur));
      if (keep === "start") {
        const ns = clamp(s, 0, dur - len);
        return { start: snap(ns), end: snap(ns + len) };
      }
      const ne = clamp(e, len, dur);
      return { start: snap(ne - len), end: snap(ne) };
    },
    [min, max, dur, snap],
  );

  // Boucle de rafraîchissement pendant la lecture : tête de lecture fluide + bouclage du passage.
  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    let last = videoRef.current?.currentTime ?? 0;
    const tick = () => {
      const v = videoRef.current;
      if (v) {
        const t = v.currentTime;
        const { start: s, end: e } = sel.current;
        if (loopRef.current && last >= s - 0.05 && last < e && t >= e) {
          v.currentTime = s;
          last = s;
          setCurrent(s);
        } else {
          last = t;
          setCurrent(t);
        }
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing]);

  // La vue suit la tête de lecture quand on est zoomé.
  useEffect(() => {
    const visible = dur / zoom;
    if (playing && zoom > 1 && (current > viewStart + visible * 0.95 || current < viewStart)) {
      setViewStart(clamp(current - visible * 0.1, 0, dur - visible));
    }
  }, [current, playing, zoom, viewStart, dur]);

  // Au montage : on se place au début du passage.
  useEffect(() => {
    const v = videoRef.current;
    if (!v) return;
    const go = () => seek(sel.current.start);
    if (v.readyState >= 1) go();
    else v.addEventListener("loadedmetadata", go, { once: true });
  }, [seek]);

  const togglePlay = useCallback(() => {
    const v = videoRef.current;
    if (!v) return;
    if (v.paused) void v.play();
    else v.pause();
  }, []);

  const playSelection = useCallback(() => {
    setLoop(true);
    seek(sel.current.start);
    void videoRef.current?.play();
  }, [seek]);

  const step = useCallback(
    (delta: number) => {
      videoRef.current?.pause();
      seek(snap((videoRef.current?.currentTime ?? current) + delta));
    },
    [seek, snap, current],
  );

  const setIn = useCallback(
    (t: number) => {
      const next = fit(t, Math.max(sel.current.end, t + min), "start");
      const wanted = sel.current.end;
      onChange(next);
      if (Math.abs(next.end - wanted) > 1e-3) flash(`Sortie recalée à ${timecode(next.end, fps)}`);
    },
    [fit, min, onChange, flash, fps],
  );

  const setOut = useCallback(
    (t: number) => {
      const next = fit(Math.min(sel.current.start, t - min), t, "end");
      const wanted = sel.current.start;
      onChange(next);
      if (Math.abs(next.start - wanted) > 1e-3) flash(`Entrée recalée à ${timecode(next.start, fps)}`);
    },
    [fit, min, onChange, flash, fps],
  );

  const applyPreset = (d: number) => {
    const t = videoRef.current?.currentTime ?? current;
    onChange(fit(t, t + d, "start"));
  };

  const fitView = useCallback(() => {
    const { start: s, end: e } = sel.current;
    const visible = Math.min(dur, Math.max((e - s) * 2.4, 8));
    setZoom(dur / visible);
    setViewStart(clamp((s + e) / 2 - visible / 2, 0, dur - visible));
  }, [dur]);

  const zoomBy = (factor: number) => {
    const maxZoom = Math.max(1, dur / 4);
    const nz = clamp(zoom * factor, 1, maxZoom);
    const center = viewStart + dur / zoom / 2;
    const vis = dur / nz;
    setZoom(nz);
    setViewStart(clamp(center - vis / 2, 0, dur - vis));
  };

  // Raccourcis clavier (ignorés quand on tape dans un champ).
  useEffect(() => {
    const onKey = (e: KeyboardEvent) => {
      const el = e.target as HTMLElement;
      if (el.closest("input, textarea, [contenteditable=true]") || e.ctrlKey || e.metaKey || e.altKey) return;
      const k = e.key.toLowerCase();
      if (k === " " || k === "k") {
        e.preventDefault();
        togglePlay();
      } else if (k === "i") setIn(snap(videoRef.current?.currentTime ?? 0));
      else if (k === "o") setOut(snap(videoRef.current?.currentTime ?? 0));
      else if (k === "l") setLoop((v) => !v);
      else if (k === "p") playSelection();
      else if (e.key === "ArrowLeft") {
        e.preventDefault();
        step(e.shiftKey ? -1 : -1 / fps);
      } else if (e.key === "ArrowRight") {
        e.preventDefault();
        step(e.shiftKey ? 1 : 1 / fps);
      } else if (e.key === "Home") seek(sel.current.start);
      else if (e.key === "End") seek(sel.current.end - 1 / fps);
      else if (k === "+" || k === "=") zoomBy(1.5);
      else if (k === "-") zoomBy(1 / 1.5);
      else if (k === "f") fitView();
      else if (e.key === "?") setHelp((h) => !h);
      else if (e.key === "Escape") setHelp(false);
    };
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  });

  function onRange(s: number, e: number, mode: DragMode) {
    onChange({ start: s, end: e });
    // Aperçu en direct de l'image sous la poignée.
    if (mode === "start" || mode === "move") seek(s);
    else if (mode === "end") seek(e - 1 / fps);
  }

  function onDrag(mode: DragMode | null) {
    if (mode && mode !== "scrub") videoRef.current?.pause();
    dragMode.current = mode;
  }

  const inside = current >= start - 1e-3 && current < end;
  const len = end - start;

  return (
    <div className="flex flex-col gap-5">
      {/* Lecteur */}
      <div className="group relative overflow-hidden rounded-[var(--radius-card)] bg-black ring-1 ring-line">
        <video
          ref={videoRef}
          src={video.proxy_url!}
          poster={video.poster_url ?? undefined}
          preload="auto"
          playsInline
          muted={muted}
          onPlay={() => setPlaying(true)}
          onPause={() => setPlaying(false)}
          onSeeked={() => !playing && setCurrent(videoRef.current?.currentTime ?? 0)}
          onRateChange={() => setRate(videoRef.current?.playbackRate ?? 1)}
          onClick={togglePlay}
          className="mx-auto block max-h-[56vh] w-full cursor-pointer object-contain"
          style={{ aspectRatio: `${info.width} / ${info.height}` }}
        />
        <div className="pointer-events-none absolute top-3 left-3 flex items-center gap-2">
          <span
            className={cx(
              "flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[11px] backdrop-blur-md transition-colors",
              inside ? "bg-accent/90 text-accent-ink" : "bg-black/55 text-fg/80",
            )}
          >
            <span className={cx("size-1.5 rounded-full", inside ? "bg-accent-ink" : "bg-fg/50")} />
            {inside ? "Dans le passage" : "Hors passage"}
          </span>
        </div>
        {badge && <div className="absolute top-3 right-3">{badge}</div>}
        <AnimatePresence>
          {!playing && (
            <motion.button
              initial={{ opacity: 0, scale: 0.9 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 1.1 }}
              transition={{ duration: 0.15 }}
              onClick={togglePlay}
              aria-label="Lecture"
              className="absolute top-1/2 left-1/2 flex size-16 -translate-x-1/2 -translate-y-1/2 items-center justify-center rounded-full bg-black/50 text-fg ring-1 ring-white/15 backdrop-blur-md transition-transform hover:scale-105"
            >
              <Play className="size-7 translate-x-0.5 fill-current" />
            </motion.button>
          )}
        </AnimatePresence>
        <AnimatePresence>
          {toast && (
            <motion.div
              initial={{ opacity: 0, y: 6 }}
              animate={{ opacity: 1, y: 0 }}
              exit={{ opacity: 0 }}
              className="absolute bottom-3 left-1/2 -translate-x-1/2 rounded-full bg-black/70 px-3 py-1.5 text-[13px] backdrop-blur-md"
            >
              {toast}
            </motion.div>
          )}
        </AnimatePresence>
      </div>

      {/* Contrôles de lecture */}
      <div className="flex flex-wrap items-center gap-2">
        <Button variant="secondary" size="sm" onClick={togglePlay} aria-label={playing ? "Pause" : "Lecture"} className="w-10 px-0">
          {playing ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => step(-1 / fps)} aria-label="Image précédente" className="w-8 px-0">
          <ChevronLeft className="size-4" />
        </Button>
        <Button variant="ghost" size="sm" onClick={() => step(1 / fps)} aria-label="Image suivante" className="w-8 px-0">
          <ChevronRight className="size-4" />
        </Button>
        <div className="ml-1 font-mono text-sm tabular">
          <span className="text-fg">{timecode(current, fps)}</span>
          <span className="text-faint"> / {timecode(dur, fps)}</span>
        </div>
        <div className="ml-auto flex items-center gap-1.5">
          <Button variant="primary" size="sm" icon={<Play className="size-3.5 fill-current" />} onClick={playSelection}>
            Lire le passage
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setLoop((l) => !l)}
            aria-pressed={loop}
            title="Boucler sur le passage (L)"
            className={cx(loop && "text-accent")}
            icon={<Repeat className="size-4" />}
          >
            Boucle
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => {
              const v = videoRef.current;
              if (v) v.playbackRate = rate === 1 ? 0.5 : 1;
            }}
            title="Vitesse de lecture"
            className="w-12 px-0 font-mono"
          >
            {rate === 1 ? "1×" : "½×"}
          </Button>
          <Button variant="ghost" size="sm" onClick={() => setMuted((m) => !m)} aria-label={muted ? "Activer le son" : "Couper le son"} className="w-8 px-0">
            {muted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
          </Button>
        </div>
      </div>

      {/* Timeline */}
      <div className="rounded-[var(--radius-card)] bg-surface px-5 pt-4 pb-5 ring-1 ring-line">
        <div className="mb-1 flex items-center justify-between">
          <span className="text-[13px] text-muted">
            Glisse les poignées ou la fenêtre verte. <span className="text-faint">Ctrl + molette pour zoomer.</span>
          </span>
          <div className="flex items-center gap-1">
            <Button variant="ghost" size="sm" onClick={() => zoomBy(1 / 1.5)} disabled={zoom <= 1.001} aria-label="Dézoomer" className="w-8 px-0">
              <ZoomOut className="size-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={() => zoomBy(1.5)} aria-label="Zoomer" className="w-8 px-0">
              <ZoomIn className="size-4" />
            </Button>
            <Button variant="ghost" size="sm" onClick={fitView} title="Cadrer sur le passage (F)" icon={<Maximize2 className="size-3.5" />}>
              Cadrer
            </Button>
            <div className="relative">
              <Button variant="ghost" size="sm" onClick={() => setHelp((h) => !h)} aria-expanded={help} aria-label="Raccourcis clavier" className="w-8 px-0">
                <Keyboard className="size-4" />
              </Button>
              <AnimatePresence>{help && <ShortcutHelp onClose={() => setHelp(false)} />}</AnimatePresence>
            </div>
          </div>
        </div>

        <Timeline
          duration={dur}
          fps={fps}
          filmstrip={video.filmstrip!}
          filmstripUrl={video.filmstrip_url!}
          start={start}
          end={end}
          current={current}
          min={min}
          max={max}
          zoom={zoom}
          viewStart={viewStart}
          limitHit={limitHit}
          onView={(z, v) => {
            setZoom(z);
            setViewStart(v);
          }}
          onRange={onRange}
          onSeek={seek}
          onDrag={onDrag}
          onLimit={setLimitHit}
        />

        {/* Réglages précis */}
        <div className="mt-5 flex flex-wrap items-end gap-x-6 gap-y-4">
          <div className="flex items-end gap-2">
            <TimeField label="Entrée" value={start} fps={fps} onCommit={setIn} accent />
            <Button variant="secondary" size="sm" className="mb-1" onClick={() => setIn(snap(current))} title="Entrée à la tête de lecture (I)">
              <Kbd>I</Kbd> ici
            </Button>
          </div>
          <div className="flex items-end gap-2">
            <TimeField label="Sortie" value={end} fps={fps} onCommit={setOut} accent />
            <Button variant="secondary" size="sm" className="mb-1" onClick={() => setOut(snap(current))} title="Sortie à la tête de lecture (O)">
              <Kbd>O</Kbd> ici
            </Button>
          </div>
          <div className="flex flex-col gap-1">
            <span className="text-[11px] font-medium tracking-wide text-faint uppercase">Durée</span>
            <div className="flex h-10 items-center font-mono text-[15px] tabular">
              {len.toFixed(2).replace(".", ",")} s
              <span className="ml-2 text-faint">({min}–{max} s)</span>
            </div>
          </div>
          <div className="ml-auto flex flex-col gap-1">
            <span className="text-[11px] font-medium tracking-wide text-faint uppercase">Depuis la tête de lecture</span>
            <div className="flex gap-1.5">
              {PRESETS.filter((d) => d >= min && d <= max).map((d) => (
                <Button
                  key={d}
                  size="sm"
                  variant="secondary"
                  onClick={() => applyPreset(d)}
                  className={cx("h-10 w-12 px-0 font-mono", Math.abs(len - d) < 0.02 && "text-accent ring-accent/50")}
                >
                  {d}s
                </Button>
              ))}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

const SHORTCUTS: [string, string][] = [
  ["Espace", "Lecture / pause"],
  ["P", "Lire le passage en boucle"],
  ["I / O", "Entrée / sortie à la tête de lecture"],
  ["← →", "Image par image"],
  ["⇧ ← →", "Seconde par seconde"],
  ["Début / Fin", "Aller à l'entrée / la sortie"],
  ["L", "Boucle on / off"],
  ["+ / −", "Zoom de la timeline"],
  ["F", "Cadrer sur le passage"],
];

function ShortcutHelp({ onClose }: { onClose: () => void }) {
  return (
    <motion.div
      initial={{ opacity: 0, y: 4, scale: 0.98 }}
      animate={{ opacity: 1, y: 0, scale: 1 }}
      exit={{ opacity: 0, y: 4, scale: 0.98 }}
      transition={{ duration: 0.12 }}
      className="absolute right-0 bottom-10 z-50 w-72 rounded-xl bg-overlay p-3 shadow-[var(--shadow-float)] ring-1 ring-line-strong"
      onMouseLeave={onClose}
    >
      <div className="mb-2 px-1 text-[13px] font-medium">Raccourcis clavier</div>
      <ul className="space-y-1">
        {SHORTCUTS.map(([k, v]) => (
          <li key={k} className="flex items-center justify-between gap-3 rounded-md px-1 py-0.5 text-[13px]">
            <span className="text-muted">{v}</span>
            <Kbd>{k}</Kbd>
          </li>
        ))}
      </ul>
    </motion.div>
  );
}
