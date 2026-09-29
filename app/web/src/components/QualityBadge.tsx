import { MonitorPlay } from "lucide-react";
import type { VideoInfo } from "../lib/api";
import { cx } from "./ui";

/** Qualité en « p » comme YouTube : le petit côté de l'image (un Short 720p fait 720×1280). */
export function resolutionP(info: Pick<VideoInfo, "width" | "height">): number {
  return Math.min(info.width, info.height);
}

function tier(p: number): string {
  if (p >= 2160) return "4K";
  if (p >= 1440) return "2K";
  if (p >= 1080) return "Full HD";
  if (p >= 720) return "HD";
  return "SD";
}

/**
 * Qualité de la vidéo et, si `maxRes` est donné, celle du passage rendu (la source réduite au-delà de maxRes).
 * En orange sous 720p : le rendu ne peut pas être plus net que la source.
 */
export function QualityBadge({ info, maxRes, compact, className }: {
  info: Pick<VideoInfo, "width" | "height">;
  maxRes?: number;
  compact?: boolean;
  className?: string;
}) {
  const src = resolutionP(info);
  const out = maxRes ? Math.min(src, maxRes) : src;
  const low = src < 720;
  const size = `${info.width}×${info.height}`;
  const title = !maxRes
    ? `Qualité ${src}p (${size})`
    : out < src
      ? `Source ${src}p (${size}), réduite à ${out}p pour le rendu du passage (render.max_height dans config.yaml).`
      : `Source ${src}p (${size}) : le passage sera rendu en ${out}p.` +
        (low ? " Qualité basse : réimporte la vidéo en meilleure qualité si elle existe." : "");
  return (
    <span
      title={title}
      className={cx(
        "flex items-center gap-1 font-mono text-[11px] tabular",
        compact ? "rounded-md bg-black/75 px-1.5 py-0.5" : "rounded-full bg-black/55 px-2.5 py-1 backdrop-blur-md",
        low ? "text-warn" : "text-fg/90",
        className,
      )}
    >
      {!compact && <MonitorPlay className="size-3.5" />}
      {out < src ? `${src}p → ${out}p` : `${src}p`}
      {!compact && <span className="text-fg/55">· {tier(out)}</span>}
    </span>
  );
}
