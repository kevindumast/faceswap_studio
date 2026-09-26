// Timecodes « mm:ss.ff » (ff = frame dans la seconde), lisibles et précis à la frame.

export function timecode(t: number, fps = 25, withFrames = true): string {
  const safe = Math.max(0, t);
  const totalFrames = Math.round(safe * fps);
  const frames = totalFrames % Math.round(fps);
  const secs = Math.floor(totalFrames / Math.round(fps));
  const m = Math.floor(secs / 60);
  const s = secs % 60;
  const base = `${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
  return withFrames ? `${base}.${String(frames).padStart(2, "0")}` : base;
}

/** Accepte « 83 », « 83.5 », « 1:23 », « 01:23.12 » (frames), « 1:23,5 ». Renvoie null si illisible. */
export function parseTimecode(input: string, fps = 25): number | null {
  const txt = input.trim().replace(",", ".");
  if (!txt) return null;
  if (/^\d+(\.\d+)?$/.test(txt)) return parseFloat(txt);
  const m = txt.match(/^(\d+):(\d{1,2})(?:\.(\d{1,2}))?$/);
  if (!m) return null;
  const [, mm, ss, ff] = m;
  const secs = parseInt(mm, 10) * 60 + parseInt(ss, 10);
  return secs + (ff ? parseInt(ff, 10) / fps : 0);
}

export function seconds(t: number, digits = 1): string {
  return `${t.toFixed(digits).replace(".", ",")} s`;
}

export function duration(t: number): string {
  if (t < 60) return `${Math.round(t)} s`;
  const m = Math.floor(t / 60);
  const s = Math.round(t % 60);
  return s ? `${m} min ${String(s).padStart(2, "0")}` : `${m} min`;
}

export function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
