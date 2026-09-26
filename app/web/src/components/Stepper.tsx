import { Check } from "lucide-react";
import { cx } from "./ui";

export const STEPS = ["Vidéo", "Passage", "Visages", "Rendu"] as const;

export function Stepper({ current, reachable, onGo }: { current: number; reachable: (i: number) => boolean; onGo: (i: number) => void }) {
  return (
    <nav aria-label="Étapes" className="flex items-center gap-1 overflow-x-auto">
      {STEPS.map((label, i) => {
        const done = i < current;
        const active = i === current;
        const ok = reachable(i);
        return (
          <div key={label} className="flex items-center gap-1">
            {i > 0 && <span className={cx("h-px w-4 sm:w-8", i <= current ? "bg-accent/50" : "bg-line")} />}
            <button
              disabled={!ok}
              onClick={() => onGo(i)}
              aria-current={active ? "step" : undefined}
              className={cx(
                "flex items-center gap-2 rounded-full py-1 pr-3 pl-1 text-[13px] font-medium whitespace-nowrap transition-colors disabled:cursor-not-allowed",
                active ? "bg-raised text-fg ring-1 ring-line-strong" : ok ? "text-muted hover:text-fg" : "text-faint",
              )}
            >
              <span
                className={cx(
                  "flex size-6 items-center justify-center rounded-full font-mono text-[11px]",
                  active && "bg-accent text-accent-ink",
                  done && !active && "bg-accent-soft text-accent",
                  !done && !active && "bg-overlay",
                )}
              >
                {done && !active ? <Check className="size-3.5" strokeWidth={3} /> : i + 1}
              </span>
              <span className="hidden sm:inline">{label}</span>
            </button>
          </div>
        );
      })}
    </nav>
  );
}
