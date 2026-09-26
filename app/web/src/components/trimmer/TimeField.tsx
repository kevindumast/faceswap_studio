import { useEffect, useState } from "react";
import { parseTimecode, timecode } from "../../lib/time";
import { cx } from "../ui";

/** Champ timecode éditable : affiche mm:ss.ff, accepte aussi « 83.5 » ou « 1:23 ». */
export function TimeField({
  label,
  value,
  fps,
  onCommit,
  accent,
}: {
  label: string;
  value: number;
  fps: number;
  onCommit: (t: number) => void;
  accent?: boolean;
}) {
  const [draft, setDraft] = useState<string | null>(null);
  const [invalid, setInvalid] = useState(false);

  useEffect(() => {
    if (!invalid) return;
    const id = setTimeout(() => setInvalid(false), 900);
    return () => clearTimeout(id);
  }, [invalid]);

  function commit() {
    if (draft === null) return;
    const t = parseTimecode(draft, fps);
    if (t === null) setInvalid(true);
    else onCommit(t);
    setDraft(null);
  }

  return (
    <label className="flex flex-col gap-1">
      <span className="text-[11px] font-medium tracking-wide text-faint uppercase">{label}</span>
      <input
        value={draft ?? timecode(value, fps)}
        onFocus={(e) => {
          setDraft(timecode(value, fps));
          requestAnimationFrame(() => e.target.select());
        }}
        onChange={(e) => setDraft(e.target.value)}
        onBlur={commit}
        onKeyDown={(e) => {
          if (e.key === "Enter") (e.target as HTMLInputElement).blur();
          if (e.key === "Escape") {
            setDraft(null);
            (e.target as HTMLInputElement).blur();
          }
        }}
        spellCheck={false}
        inputMode="decimal"
        className={cx(
          "h-10 w-[7.5rem] rounded-lg bg-bg px-3 font-mono text-[15px] tabular ring-1 transition-colors outline-none focus:ring-2",
          accent ? "text-accent" : "text-fg",
          invalid ? "ring-danger" : "ring-line focus:ring-accent",
        )}
      />
    </label>
  );
}
