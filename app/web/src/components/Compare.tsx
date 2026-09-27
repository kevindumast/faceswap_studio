import { Loader2, Maximize2, Minimize2, Pause, Play, Volume2, VolumeX } from "lucide-react";
import { useCallback, useEffect, useRef, useState } from "react";
import { clamp, timecode } from "../lib/time";
import { Button } from "./ui";

/** Lecteur avant / après : deux vidéos synchronisées, un rideau déplaçable. */
export function Compare({ before, after, aspect, fps, autoPlay = false }: { before: string; after: string; aspect: string; fps: number; autoPlay?: boolean }) {
  const box = useRef<HTMLDivElement>(null);
  const wrapper = useRef<HTMLDivElement>(null);
  const [fullscreen, setFullscreen] = useState(false);

  useEffect(() => {
    const onChange = () => setFullscreen(document.fullscreenElement === wrapper.current);
    document.addEventListener("fullscreenchange", onChange);
    return () => document.removeEventListener("fullscreenchange", onChange);
  }, []);
  const toggleFullscreen = () => {
    if (document.fullscreenElement) void document.exitFullscreen();
    else void wrapper.current?.requestFullscreen();
  };
  const a = useRef<HTMLVideoElement>(null);
  const b = useRef<HTMLVideoElement>(null);
  const [pos, setPos] = useState(50);
  const [playing, setPlaying] = useState(false);
  const [t, setT] = useState(0);
  const [dur, setDur] = useState(0);
  const [muted, setMuted] = useState(false);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const dragging = useRef(false);

  const sync = useCallback(() => {
    if (a.current && b.current && Math.abs(b.current.currentTime - a.current.currentTime) > 0.08) {
      b.current.currentTime = a.current.currentTime;
    }
  }, []);

  useEffect(() => {
    if (!playing) return;
    let raf = 0;
    const tick = () => {
      if (a.current) setT(a.current.currentTime);
      sync();
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [playing, sync]);

  const toggle = () => {
    if (!a.current || !b.current) return;
    if (a.current.paused) {
      sync();
      // Lecture refusée par le navigateur (autoplay) : on reste simplement en pause.
      a.current.play().catch(() => b.current?.pause());
      b.current.play().catch(() => {});
    } else {
      a.current.pause();
      b.current.pause();
    }
  };

  const seek = (v: number) => {
    if (a.current) a.current.currentTime = v;
    if (b.current) b.current.currentTime = v;
    setT(v);
  };

  const moveTo = (clientX: number) => {
    const r = box.current!.getBoundingClientRect();
    setPos(clamp(((clientX - r.left) / r.width) * 100, 0, 100));
  };

  return (
    <div ref={wrapper} className={fullscreen ? "flex h-screen w-screen flex-col gap-3 bg-black p-4" : "flex flex-col gap-3"}>
      <div
        ref={box}
        className="relative touch-none overflow-hidden rounded-[var(--radius-card)] bg-black ring-1 ring-line select-none"
        style={fullscreen ? { flex: 1, minHeight: 0 } : { aspectRatio: aspect }}
        onDoubleClick={toggleFullscreen}
        onPointerDown={(e) => {
          dragging.current = true;
          (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
          moveTo(e.clientX);
        }}
        onPointerMove={(e) => dragging.current && moveTo(e.clientX)}
        onPointerUp={() => (dragging.current = false)}
      >
        <video ref={b} src={before} muted playsInline preload="auto" className="absolute inset-0 size-full object-contain" />
        <video
          ref={a}
          src={after}
          muted={muted}
          playsInline
          preload="auto"
          onLoadedMetadata={() => setDur(a.current?.duration ?? 0)}
          onLoadedData={() => {
            setState("ready");
            if (autoPlay) toggle();
          }}
          onError={() => setState("error")}
          onPlay={() => setPlaying(true)}
          onPause={() => {
            setPlaying(false);
            b.current?.pause();
          }}
          onEnded={() => b.current?.pause()}
          className="absolute inset-0 size-full object-contain"
          style={{ clipPath: `inset(0 0 0 ${pos}%)` }}
        />
        <div className="pointer-events-none absolute inset-y-0 w-0.5 bg-accent shadow-[0_0_12px_rgb(200_255_61/0.6)]" style={{ left: `${pos}%` }}>
          <div className="absolute top-1/2 left-1/2 flex h-9 w-6 -translate-x-1/2 -translate-y-1/2 items-center justify-center gap-[3px] rounded-full bg-accent shadow-lg">
            <span className="h-3.5 w-[2px] rounded-full bg-accent-ink/70" />
            <span className="h-3.5 w-[2px] rounded-full bg-accent-ink/70" />
          </div>
        </div>
        {state !== "ready" && (
          <div className="pointer-events-none absolute inset-0 flex items-center justify-center bg-black/40 text-sm text-muted">
            {state === "loading" ? <Loader2 className="size-6 animate-spin" /> : "Vidéo illisible : recharge la page."}
          </div>
        )}
        <span className="pointer-events-none absolute top-3 left-3 rounded-full bg-black/60 px-2.5 py-1 text-[11px] font-medium backdrop-blur-md">Avant</span>
        <span className="pointer-events-none absolute top-3 right-3 rounded-full bg-accent px-2.5 py-1 text-[11px] font-semibold text-accent-ink">Après</span>
      </div>
      <div className="flex items-center gap-3">
        <Button variant="secondary" size="sm" onClick={toggle} aria-label={playing ? "Pause" : "Lecture"} className="w-10 px-0">
          {playing ? <Pause className="size-4 fill-current" /> : <Play className="size-4 fill-current" />}
        </Button>
        <input
          type="range"
          min={0}
          max={dur || 1}
          step={1 / fps}
          value={t}
          onChange={(e) => seek(parseFloat(e.target.value))}
          aria-label="Position"
          className="h-1 flex-1 cursor-pointer accent-[var(--color-accent)]"
        />
        <span className="font-mono text-[13px] text-muted tabular">
          {timecode(t, fps)} / {timecode(dur, fps)}
        </span>
        <Button
          variant="ghost"
          size="sm"
          onClick={toggleFullscreen}
          aria-label={fullscreen ? "Quitter le plein écran" : "Plein écran"}
          title={fullscreen ? "Quitter le plein écran (Échap)" : "Plein écran (ou double-clic sur la vidéo)"}
          className="w-8 px-0"
        >
          {fullscreen ? <Minimize2 className="size-4" /> : <Maximize2 className="size-4" />}
        </Button>
        <Button variant="ghost" size="sm" onClick={() => setMuted((m) => !m)} aria-label={muted ? "Activer le son" : "Couper le son"} className="w-8 px-0">
          {muted ? <VolumeX className="size-4" /> : <Volume2 className="size-4" />}
        </Button>
      </div>
    </div>
  );
}
