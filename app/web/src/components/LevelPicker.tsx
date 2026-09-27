import { useMutation, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, Check, Cpu, Download, FlaskConical, Palette, PersonStanding, ScanFace, UserRound, Zap } from "lucide-react";
import { useState, type ReactNode } from "react";
import { api, type Level, type Status } from "../lib/api";
import { LEVELS, levelInfo } from "../lib/levels";
import { Button, Notice, ProgressBar, cx } from "./ui";

const ICONS: Record<Level, ReactNode> = {
  face: <ScanFace className="size-5" />,
  face_tone: <Palette className="size-5" />,
  head: <UserRound className="size-5" />,
  character: <PersonStanding className="size-5" />,
};

const COMING: Partial<Record<Level, string>> = {
  head: "Pas encore disponible : la tête complète arrive dans une prochaine étape.",
  character: "Pas encore disponible : arrive juste après le branchement du GPU ZeroGPU, qui est indispensable à ce niveau.",
};

export function LevelPicker({ value, onChange, status }: { value: Level; onChange: (l: Level) => void; status?: Status }) {
  const qc = useQueryClient();
  const install = useMutation({
    mutationFn: (groups: string[]) => Promise.all(groups.map((g) => api.installModels(g))),
    onSettled: () => qc.invalidateQueries({ queryKey: ["status"] }),
  });
  // Un niveau pas encore livré peut être consulté (détail + alertes) sans être sélectionné pour le rendu.
  const [preview, setPreview] = useState<Level | null>(null);
  const shown = preview ?? value;
  const selected = levelInfo(shown);
  const st = status?.levels[shown];
  const locked = !!st && !st.available;

  return (
    <section className="mb-6">
      <div className="mb-3 flex items-baseline justify-between">
        <h2 className="text-[13px] font-medium tracking-wide text-faint uppercase">Niveau de transformation</h2>
        <span className="text-[12px] text-muted">Tout tourne sur ton CPU, sauf le niveau 4</span>
      </div>
      <div role="radiogroup" aria-label="Niveau de transformation" className="grid grid-cols-2 gap-3 lg:grid-cols-4">
        {LEVELS.map((lvl) => {
          const s = status?.levels[lvl.id];
          const available = s?.available ?? lvl.id === "face";
          const on = value === lvl.id;
          const previewing = preview === lvl.id;
          return (
            <button
              key={lvl.id}
              role="radio"
              aria-checked={on}
              aria-disabled={!available}
              onClick={() => {
                if (available) {
                  setPreview(null);
                  onChange(lvl.id);
                } else {
                  setPreview(previewing ? null : lvl.id);
                }
              }}
              className={cx(
                "group relative flex flex-col gap-2 rounded-2xl p-4 text-left ring-1 transition-[box-shadow,background]",
                on ? "bg-accent-soft ring-2 ring-accent" : "bg-surface ring-line hover:ring-line-strong",
                !available && "border border-dashed border-line-strong opacity-70 ring-0",
                previewing && "opacity-100 ring-1 ring-faint",
              )}
            >
              <div className="flex items-center justify-between">
                <span className={cx("flex size-9 items-center justify-center rounded-xl", on ? "bg-accent text-accent-ink" : "bg-raised text-muted group-hover:text-fg")}>
                  {ICONS[lvl.id]}
                </span>
                <span className="font-mono text-[11px] text-faint">Niv. {lvl.step}</span>
              </div>
              <div>
                <div className="font-medium">{lvl.title}</div>
                <div className="mt-0.5 text-[13px] leading-snug text-muted">{lvl.tagline}</div>
              </div>
              <div className="mt-auto flex flex-wrap gap-1.5 pt-1">
                {lvl.gpuOnly ? (
                  <Badge tone="warn" icon={<Zap className="size-3" />}>
                    GPU requis
                  </Badge>
                ) : (
                  <Badge tone="ok" icon={<Cpu className="size-3" />}>
                    CPU ✓{s ? ` · ${s.sec_per_frame.toFixed(1).replace(".", ",")} s/img` : ""}
                  </Badge>
                )}
                {lvl.experimental && (
                  <Badge tone="muted" icon={<FlaskConical className="size-3" />}>
                    Expérimental
                  </Badge>
                )}
                {!available && <Badge tone="muted">Bientôt</Badge>}
              </div>
              {on && (
                <span className="absolute top-3 right-3 hidden size-5 items-center justify-center rounded-full bg-accent text-accent-ink sm:flex">
                  <Check className="size-3.5" strokeWidth={3} />
                </span>
              )}
            </button>
          );
        })}
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={shown}
          initial={{ opacity: 0, y: 4 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.15 }}
          className="mt-3 grid gap-4 rounded-2xl bg-surface p-4 ring-1 ring-line sm:grid-cols-[minmax(0,1fr)_minmax(0,1.4fr)] sm:p-5"
        >
          {locked && (
            <Notice tone="info" className="sm:col-span-2">
              <span className="font-medium text-fg">Niveau {selected.step} · {selected.title}.</span> {COMING[shown] ?? "Pas encore disponible."}{" "}
              Tu restes sur le niveau {levelInfo(value).step} pour tes rendus.
            </Notice>
          )}
          <div>
            <div className="mb-2 text-[12px] font-medium tracking-wide text-faint uppercase">Ce qui change</div>
            <ul className="space-y-1.5 text-[13px]">
              {selected.changes.map((c) => (
                <li key={c} className="flex gap-2">
                  <Check className="mt-0.5 size-3.5 shrink-0 text-accent" strokeWidth={3} /> {c}
                </li>
              ))}
            </ul>
          </div>
          <div>
            <div className="mb-2 text-[12px] font-medium tracking-wide text-faint uppercase">À savoir</div>
            <ul className="space-y-1.5 text-[13px] text-muted">
              {selected.alerts.map((a) => (
                <li key={a} className="flex gap-2">
                  <AlertTriangle className="mt-0.5 size-3.5 shrink-0 text-warn" /> {a}
                </li>
              ))}
            </ul>
          </div>
          {st && st.available && !st.ready && (
            <div className="sm:col-span-2">
              {st.installing ? (
                <div>
                  <div className="mb-1.5 flex justify-between text-[13px]">
                    <span>Installation des modèles…</span>
                    <span className="font-mono text-muted tabular">{Math.round(st.installing.progress * 100)} %</span>
                  </div>
                  <ProgressBar value={st.installing.progress} />
                </div>
              ) : (
                <div className="flex flex-wrap items-center justify-between gap-3 rounded-xl bg-raised p-3 ring-1 ring-line">
                  <span className="text-[13px]">Ce niveau a besoin d'un modèle en plus ({st.install_mb} Mo, une seule fois).</span>
                  <Button variant="primary" size="sm" icon={<Download className="size-3.5" />} loading={install.isPending} onClick={() => install.mutate(st.missing_groups)}>
                    Installer
                  </Button>
                </div>
              )}
              {st.install_error && (
                <Notice tone="danger" className="mt-3">
                  {st.install_error}
                </Notice>
              )}
            </div>
          )}
        </motion.div>
      </AnimatePresence>
    </section>
  );
}

function Badge({ children, icon, tone }: { children: ReactNode; icon?: ReactNode; tone: "ok" | "warn" | "muted" }) {
  return (
    <span
      className={cx(
        "inline-flex items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-medium whitespace-nowrap",
        tone === "ok" && "bg-accent-soft text-accent",
        tone === "warn" && "bg-warn-soft text-warn",
        tone === "muted" && "bg-overlay text-muted",
      )}
    >
      {icon}
      {children}
    </span>
  );
}
