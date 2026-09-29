import { ArrowRight, Cpu } from "lucide-react";
import type { Status, Video } from "../lib/api";
import { cappedFps, duration } from "../lib/time";
import { Button, SectionTitle } from "../components/ui";
import { QualityBadge } from "../components/QualityBadge";
import { Trimmer, type Selection } from "../components/trimmer/Trimmer";

type Props = {
  video: Video;
  status?: Status;
  selection: Selection;
  onChange: (s: Selection) => void;
  onNext: () => void;
};

export function SegmentStep({ video, status, selection, onChange, onNext }: Props) {
  const min = status?.segment.min_s ?? 5;
  const max = status?.segment.max_s ?? 60;
  // Même hypothèse que l'option cochée par défaut à l'étape Rendu : 30 i/s max.
  const fps = cappedFps(video.info!.fps, status?.fps_cap ?? 30);
  const frames = Math.round((selection.end - selection.start) * fps);
  const estimate = frames * (status?.sec_per_frame ?? 2.2) + 15;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 2 · Passage"
        title="Choisis le passage à transformer"
        subtitle={<span className="line-clamp-1">{video.title}</span>}
        right={
          <div className="flex items-center gap-4">
            <div className="text-right">
              <div className="flex items-center justify-end gap-1.5 text-[13px] text-muted">
                <Cpu className="size-3.5" /> Calcul estimé · 1 visage · {status?.engine.label ?? "CPU"}
              </div>
              <div className="font-mono text-lg tabular">≈ {duration(estimate)}</div>
            </div>
            <Button variant="primary" size="lg" onClick={onNext} icon={<ArrowRight className="size-4" />}>
              Continuer
            </Button>
          </div>
        }
      />
      <Trimmer
        video={video}
        selection={selection}
        onChange={onChange}
        min={min}
        max={max}
        badge={<QualityBadge info={video.info!} maxRes={status?.render_max_res ?? 1080} />}
      />
    </div>
  );
}
