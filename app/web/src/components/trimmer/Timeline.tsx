import { useCallback, useEffect, useLayoutEffect, useMemo, useRef, useState, type PointerEvent as RPointerEvent } from "react";
import type { Filmstrip } from "../../lib/api";
import { clamp, seconds, timecode } from "../../lib/time";
import { cx } from "../ui";

export type DragMode = "start" | "end" | "move" | "scrub";
export type LimitHit = "min" | "max" | null;

type Props = {
  duration: number;
  fps: number;
  filmstrip: Filmstrip;
  filmstripUrl: string;
  start: number;
  end: number;
  current: number;
  min: number;
  max: number;
  zoom: number;
  viewStart: number;
  limitHit: LimitHit;
  onView: (zoom: number, viewStart: number) => void;
  onRange: (start: number, end: number, mode: DragMode) => void;
  onSeek: (t: number) => void;
  onDrag: (mode: DragMode | null) => void;
  onLimit: (hit: LimitHit) => void;
};

const STRIP_H = 64;
const RULER_H = 26;
const TICK_STEPS = [0.2, 0.5, 1, 2, 5, 10, 15, 30, 60, 120, 300, 600];

export function Timeline(p: Props) {
  const trackRef = useRef<HTMLDivElement>(null);
  const [width, setWidth] = useState(800);
  const [hover, setHover] = useState<{ x: number; t: number } | null>(null);
  const [dragging, setDragging] = useState<DragMode | null>(null);
  const drag = useRef<{ mode: DragMode; grabT: number; start: number; end: number; x0: number; moved: boolean } | null>(null);

  useLayoutEffect(() => {
    const el = trackRef.current;
    if (!el) return;
    const ro = new ResizeObserver(([entry]) => setWidth(entry.contentRect.width));
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  const visible = p.duration / p.zoom;
  const toX = useCallback((t: number) => ((t - p.viewStart) / visible) * width, [p.viewStart, visible, width]);
  const toT = useCallback(
    (clientX: number) => {
      const rect = trackRef.current!.getBoundingClientRect();
      return clamp(p.viewStart + ((clientX - rect.left) / rect.width) * visible, 0, p.duration);
    },
    [p.viewStart, visible, p.duration],
  );
  const snap = useCallback((t: number) => Math.round(t * p.fps) / p.fps, [p.fps]);

  // Molette : Ctrl/⌘ = zoom autour du curseur, sinon défilement horizontal quand on est zoomé.
  useEffect(() => {
    const el = trackRef.current?.parentElement;
    if (!el) return;
    const onWheel = (e: WheelEvent) => {
      const rect = trackRef.current!.getBoundingClientRect();
      const maxZoom = Math.max(1, p.duration / 4);
      if (e.ctrlKey || e.metaKey) {
        e.preventDefault();
        const anchorT = p.viewStart + ((e.clientX - rect.left) / rect.width) * visible;
        const zoom = clamp(p.zoom * Math.exp(-e.deltaY * 0.0025), 1, maxZoom);
        const vis = p.duration / zoom;
        const frac = (e.clientX - rect.left) / rect.width;
        p.onView(zoom, clamp(anchorT - frac * vis, 0, p.duration - vis));
      } else if (p.zoom > 1.001) {
        e.preventDefault();
        const delta = (Math.abs(e.deltaX) > Math.abs(e.deltaY) ? e.deltaX : e.deltaY) / rect.width;
        p.onView(p.zoom, clamp(p.viewStart + delta * visible, 0, p.duration - visible));
      }
    };
    el.addEventListener("wheel", onWheel, { passive: false });
    return () => el.removeEventListener("wheel", onWheel);
  }, [p, visible]);

  function beginDrag(e: RPointerEvent, mode: DragMode) {
    e.preventDefault();
    e.stopPropagation();
    (e.currentTarget as HTMLElement).setPointerCapture(e.pointerId);
    const t = toT(e.clientX);
    drag.current = { mode, grabT: t, start: p.start, end: p.end, x0: e.clientX, moved: false };
    setDragging(mode);
    p.onDrag(mode);
    if (mode === "scrub") p.onSeek(snap(t));
  }

  function onPointerMove(e: RPointerEvent) {
    const rect = trackRef.current!.getBoundingClientRect();
    const x = e.clientX - rect.left;
    const d = drag.current;
    if (!d) {
      setHover(x >= 0 && x <= rect.width ? { x, t: toT(e.clientX) } : null);
      return;
    }
    if (Math.abs(e.clientX - d.x0) > 3) d.moved = true;
    const t = toT(e.clientX);
    if (d.mode === "scrub") {
      p.onSeek(snap(t));
    } else if (d.mode === "start") {
      const lo = Math.max(0, d.end - p.max);
      const hi = d.end - p.min;
      p.onLimit(t < lo - 1e-3 ? "max" : t > hi + 1e-3 ? "min" : null);
      p.onRange(snap(clamp(t, lo, hi)), d.end, "start");
    } else if (d.mode === "end") {
      const lo = d.start + p.min;
      const hi = Math.min(p.duration, d.start + p.max);
      p.onLimit(t > hi + 1e-3 && hi < p.duration ? "max" : t < lo - 1e-3 ? "min" : null);
      p.onRange(d.start, snap(clamp(t, lo, hi)), "end");
    } else if (d.mode === "move" && d.moved) {
      const len = d.end - d.start;
      const s = snap(clamp(d.start + (t - d.grabT), 0, p.duration - len));
      p.onRange(s, s + len, "move");
    }
  }

  function endDrag(e: RPointerEvent) {
    const d = drag.current;
    if (!d) return;
    // Clic simple dans la sélection = placer la tête de lecture.
    if (d.mode === "move" && !d.moved) p.onSeek(snap(toT(e.clientX)));
    drag.current = null;
    setDragging(null);
    p.onDrag(null);
    p.onLimit(null);
  }

  function onHandleKey(e: React.KeyboardEvent, which: "start" | "end") {
    const step = e.shiftKey ? 1 : 1 / p.fps;
    const dir = e.key === "ArrowLeft" || e.key === "ArrowDown" ? -1 : e.key === "ArrowRight" || e.key === "ArrowUp" ? 1 : 0;
    if (!dir) return;
    e.preventDefault();
    e.stopPropagation();
    if (which === "start") {
      const s = snap(clamp(p.start + dir * step, Math.max(0, p.end - p.max), p.end - p.min));
      p.onRange(s, p.end, "start");
    } else {
      const en = snap(clamp(p.end + dir * step, p.start + p.min, Math.min(p.duration, p.start + p.max)));
      p.onRange(p.start, en, "end");
    }
  }

  // Graduations : un pas qui laisse au moins ~90 px entre deux libellés.
  const ticks = useMemo(() => {
    const pxPerSec = width / visible;
    const step = TICK_STEPS.find((s) => s * pxPerSec >= 90) ?? 600;
    const minor = step / 5;
    const out: { t: number; major: boolean }[] = [];
    const first = Math.floor(p.viewStart / minor) * minor;
    for (let t = first; t <= p.viewStart + visible + minor; t += minor) {
      const major = Math.abs(t / step - Math.round(t / step)) < 1e-6;
      if (t >= 0 && t <= p.duration) out.push({ t, major });
    }
    return { out, step };
  }, [width, visible, p.viewStart, p.duration]);

  // Vignettes : un emplacement par largeur de tuile, chacun montre la vignette la plus proche de son instant.
  const fs = p.filmstrip;
  const tileW = (STRIP_H * fs.tile_w) / fs.tile_h;
  const slots = Math.ceil(width / tileW) + 1;
  const sprite = (idx: number, h: number) => {
    const w = (h * fs.tile_w) / fs.tile_h;
    const col = idx % fs.cols;
    const row = Math.floor(idx / fs.cols);
    return {
      backgroundImage: `url(${p.filmstripUrl})`,
      backgroundSize: `${fs.cols * w}px ${fs.rows * h}px`,
      backgroundPosition: `-${col * w}px -${row * h}px`,
    };
  };
  const thumbAt = (t: number) => clamp(Math.floor(t / fs.interval + 0.5), 0, fs.count - 1);

  const xs = toX(p.start);
  const xe = toX(p.end);
  const xc = toX(p.current);
  const len = p.end - p.start;
  const warn = p.limitHit !== null;
  const tickLabel = (t: number) => (ticks.step < 1 ? timecode(t, p.fps) : timecode(t, p.fps, false));

  return (
    <div className="relative select-none" style={{ paddingTop: RULER_H + 8 }}>
      {/* Badge de durée, au-dessus de la sélection */}
      <div
        className="pointer-events-none absolute top-0 z-30 -translate-x-1/2 transition-[left] duration-75"
        style={{ left: clamp((xs + xe) / 2, 70, width - 70) }}
      >
        <div
          className={cx(
            "flex items-center gap-1.5 rounded-full px-2.5 py-1 font-mono text-[11px] font-medium whitespace-nowrap shadow-[var(--shadow-float)] tabular",
            warn ? "bg-warn text-accent-ink" : "bg-accent text-accent-ink",
          )}
        >
          {warn ? (p.limitHit === "max" ? `Max ${p.max} s` : `Min ${p.min} s`) : seconds(len)}
          <span className="opacity-60">·</span>
          {Math.round(len * p.fps)} img
        </div>
      </div>

      <div
        ref={trackRef}
        className="relative touch-none"
        onPointerMove={onPointerMove}
        onPointerUp={endDrag}
        onPointerCancel={endDrag}
        onPointerLeave={() => !drag.current && setHover(null)}
      >
        {/* Règle (glisser = déplacer la tête de lecture) */}
        <div
          className="absolute inset-x-0 cursor-ew-resize"
          style={{ top: -RULER_H - 4, height: RULER_H }}
          onPointerDown={(e) => beginDrag(e, "scrub")}
        >
          {ticks.out.map(({ t, major }) => (
            <div key={t.toFixed(3)} className="absolute bottom-0" style={{ left: toX(t) }}>
              <div className={cx("w-px", major ? "h-2.5 bg-line-strong" : "h-1.5 bg-line")} />
              {major && (
                <div className="absolute bottom-3 -translate-x-1/2 font-mono text-[10px] whitespace-nowrap text-faint tabular">
                  {tickLabel(t)}
                </div>
              )}
            </div>
          ))}
        </div>

        {/* Filmstrip */}
        <div
          className="relative overflow-hidden rounded-xl bg-raised ring-1 ring-line"
          style={{ height: STRIP_H }}
          onPointerDown={(e) => beginDrag(e, "scrub")}
        >
          {Array.from({ length: slots }, (_, i) => {
            const t = p.viewStart + ((i + 0.5) * tileW * visible) / width;
            return (
              <div
                key={i}
                className="absolute top-0 bg-no-repeat"
                style={{ left: i * tileW, width: tileW + 0.5, height: STRIP_H, ...sprite(thumbAt(t), STRIP_H) }}
              />
            );
          })}
          {/* Hors sélection : assombri et désaturé */}
          <div className="absolute inset-y-0 left-0 bg-black/60 backdrop-grayscale" style={{ width: Math.max(0, xs) }} />
          <div className="absolute inset-y-0 right-0 bg-black/60 backdrop-grayscale" style={{ left: Math.min(width, xe) }} />
        </div>

        {/* Fenêtre de sélection */}
        <div
          className={cx(
            "absolute top-0 z-10 cursor-grab rounded-[10px] transition-shadow active:cursor-grabbing",
            warn ? "ring-2 ring-warn" : "ring-2 ring-accent",
            dragging === "move" && "shadow-[0_0_0_6px_rgb(200_255_61/0.12)]",
          )}
          style={{ left: xs, width: Math.max(0, xe - xs), height: STRIP_H }}
          onPointerDown={(e) => beginDrag(e, "move")}
        >
          <Handle
            side="start"
            warn={warn}
            active={dragging === "start"}
            label="Début du passage"
            value={p.start}
            min={0}
            max={p.duration}
            valueText={timecode(p.start, p.fps)}
            onPointerDown={(e) => beginDrag(e, "start")}
            onKeyDown={(e) => onHandleKey(e, "start")}
          />
          <Handle
            side="end"
            warn={warn}
            active={dragging === "end"}
            label="Fin du passage"
            value={p.end}
            min={0}
            max={p.duration}
            valueText={timecode(p.end, p.fps)}
            onPointerDown={(e) => beginDrag(e, "end")}
            onKeyDown={(e) => onHandleKey(e, "end")}
          />
        </div>

        {/* Tête de lecture */}
        {xc >= -1 && xc <= width + 1 && (
          <div className="pointer-events-none absolute z-20" style={{ left: xc, top: -RULER_H - 4, bottom: 0 }}>
            <div className="absolute top-0 left-1/2 h-3 w-2.5 -translate-x-1/2 rounded-b-[4px] rounded-t-[2px] bg-fg shadow" />
            <div className="absolute top-2 bottom-0 left-1/2 w-0.5 -translate-x-1/2 bg-fg shadow-[0_0_8px_rgb(0_0_0/0.8)]" />
          </div>
        )}

        {/* Survol : aperçu de la vignette + timecode */}
        {hover && !dragging && (
          <>
            <div className="pointer-events-none absolute top-0 z-10 w-px bg-fg/50" style={{ left: hover.x, height: STRIP_H }} />
            <div
              className="pointer-events-none absolute z-40 -translate-x-1/2 overflow-hidden rounded-lg bg-overlay shadow-[var(--shadow-float)]"
              style={{ left: clamp(hover.x, 70, width - 70), bottom: STRIP_H + RULER_H + 16 }}
            >
              <div className="bg-no-repeat" style={{ width: 128, height: 72, ...sprite(thumbAt(hover.t), 72) }} />
              <div className="px-2 py-1 text-center font-mono text-[11px] text-fg tabular">{timecode(hover.t, p.fps)}</div>
            </div>
          </>
        )}
      </div>
    </div>
  );
}

function Handle(props: {
  side: "start" | "end";
  warn: boolean;
  active: boolean;
  label: string;
  value: number;
  min: number;
  max: number;
  valueText: string;
  onPointerDown: (e: RPointerEvent) => void;
  onKeyDown: (e: React.KeyboardEvent) => void;
}) {
  const left = props.side === "start";
  return (
    <div
      role="slider"
      tabIndex={0}
      aria-label={props.label}
      aria-valuemin={props.min}
      aria-valuemax={props.max}
      aria-valuenow={Number(props.value.toFixed(2))}
      aria-valuetext={props.valueText}
      onPointerDown={props.onPointerDown}
      onKeyDown={props.onKeyDown}
      className={cx(
        "group absolute inset-y-[-2px] z-10 flex w-4 cursor-ew-resize items-center justify-center outline-none",
        left ? "-left-4 rounded-l-[10px]" : "-right-4 rounded-r-[10px]",
        props.warn ? "bg-warn" : "bg-accent",
        "focus-visible:ring-2 focus-visible:ring-fg",
      )}
    >
      <div className={cx("flex gap-[3px] transition-transform", props.active && "scale-y-125")}>
        <span className="h-5 w-[2px] rounded-full bg-accent-ink/70" />
        <span className="h-5 w-[2px] rounded-full bg-accent-ink/70" />
      </div>
    </div>
  );
}
