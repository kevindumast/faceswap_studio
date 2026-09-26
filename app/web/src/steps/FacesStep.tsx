import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import {
  AlertTriangle,
  ArrowLeftRight,
  ArrowRight,
  Check,
  ChevronDown,
  ImagePlus,
  Loader2,
  MoveRight,
  ScanFace,
  ShieldCheck,
  Sparkles,
  Sun,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type FaceSet, type FramesFaces, type Mapping, type Person, type Photo, type Status, type Video } from "../lib/api";
import { faceIndex, mappingsEqual, personColor, reconcile } from "../lib/people";
import { timecode } from "../lib/time";
import { Button, Card, Menu, MenuItem, Notice, SectionTitle, SegmentedControl, cx } from "../components/ui";
import type { Selection } from "../components/trimmer/Trimmer";

type Props = {
  video: Video;
  status?: Status;
  selection: Selection;
  faceSetId: string | null;
  onFaceSet: (id: string | null) => void;
  consent: boolean;
  onConsent: (v: boolean) => void;
  mappings: Mapping[];
  onMappings: (m: Mapping[]) => void;
  onNext: () => void;
};

export function FacesStep(p: Props) {
  const faceSet = useQuery({
    queryKey: ["faceset", p.faceSetId],
    queryFn: () => api.faceSet(p.faceSetId!),
    enabled: !!p.faceSetId,
    retry: false,
  });
  // Set supprimé (rétention) : on repart de zéro.
  const { onFaceSet } = p;
  useEffect(() => {
    if (faceSet.error) onFaceSet(null);
  }, [faceSet.error, onFaceSet]);

  const persons = faceSet.data?.persons ?? [];
  const active = p.mappings.filter((m) => m.person).length;
  const canGo = p.consent && persons.length > 0 && active > 0;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 3 · Visages"
        title="Qui remplace qui ?"
        subtitle="Importe les photos : elles sont rangées automatiquement par personne. Chaque personne est ensuite associée à un visage du clip, et tu peux tout changer."
        right={
          <Button variant="primary" size="lg" disabled={!canGo} onClick={p.onNext} icon={<ArrowRight className="size-4" />}>
            Continuer
          </Button>
        }
      />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <SourcePanel {...p} data={faceSet.data} />
        <TargetPanel video={p.video} selection={p.selection} persons={persons} mappings={p.mappings} onMappings={p.onMappings} />
      </div>
    </div>
  );
}

/* ───────────────────────── Photos source, rangées par personne ───────────────────────── */

function SourcePanel(p: Props & { data?: FaceSet }) {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState(0);
  const max = p.status?.photos_max ?? 20;
  const photos = p.data?.photos ?? [];
  const persons = p.data?.persons ?? [];
  const noFace = photos.filter((ph) => !ph.ok);

  const add = useMutation({
    mutationFn: (files: File[]) => (p.faceSetId ? api.addPhotos(p.faceSetId, files) : api.createFaceSet(files)),
    onMutate: (files) => setUploading(files.length),
    onSettled: () => setUploading(0),
    onSuccess: (set) => {
      qc.setQueryData(["faceset", set.id], set);
      p.onFaceSet(set.id);
    },
  });
  const update = (set: FaceSet) => qc.setQueryData(["faceset", set.id], set);
  const remove = useMutation({ mutationFn: (photoId: string) => api.deletePhoto(p.faceSetId!, photoId), onSuccess: update });
  const move = useMutation({ mutationFn: ({ id, person }: { id: string; person: string }) => api.movePhoto(p.faceSetId!, id, person), onSuccess: update });

  const pick = (list: FileList | null) => {
    if (!list || !p.consent) return;
    const files = Array.from(list).filter((f) => f.type.startsWith("image/")).slice(0, max - photos.length);
    if (files.length) add.mutate(files);
  };

  const dropProps = {
    onDragOver: (e: React.DragEvent) => {
      e.preventDefault();
      setOver(true);
    },
    onDragLeave: () => setOver(false),
    onDrop: (e: React.DragEvent) => {
      e.preventDefault();
      setOver(false);
      pick(e.dataTransfer.files);
    },
  };

  return (
    <Card className="flex flex-col p-5 sm:p-6">
      <div className="mb-4 flex items-center justify-between">
        <h2 className="flex items-center gap-2 font-medium">
          <Users className="size-4.5 text-accent" /> Visages source
        </h2>
        <span className="font-mono text-[12px] text-muted tabular">
          {persons.length} personne{persons.length > 1 ? "s" : ""} · {photos.length}/{max} photos
        </span>
      </div>

      <ConsentBox checked={p.consent} onChange={p.onConsent} />

      {photos.length === 0 && uploading === 0 ? (
        <button
          disabled={!p.consent}
          onClick={() => input.current?.click()}
          {...dropProps}
          className={cx(
            "flex flex-col items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-10 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-40",
            over ? "border-accent bg-accent-soft" : "border-line-strong hover:border-faint",
          )}
        >
          <ImagePlus className="size-7 text-muted" />
          <span className="text-sm font-medium">Ajoute les photos de toi et/ou de ton pote</span>
          <span className="max-w-xs text-[13px] text-muted">Mélange-les sans souci : chaque visage est reconnu et rangé dans sa propre personne.</span>
        </button>
      ) : (
        <div className="flex flex-col gap-3">
          <AnimatePresence initial={false}>
            {persons.map((person) => (
              <motion.div key={person.id} layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
                <PersonGroup
                  person={person}
                  photos={photos.filter((ph) => ph.ok && ph.person === person.id)}
                  others={persons.filter((o) => o.id !== person.id)}
                  onRemove={(id) => remove.mutate(id)}
                  onMove={(id, to) => move.mutate({ id, person: to })}
                />
              </motion.div>
            ))}
          </AnimatePresence>

          {noFace.length > 0 && (
            <div className="rounded-xl bg-warn-soft/50 p-3 ring-1 ring-warn/20">
              <div className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-warn">
                <AlertTriangle className="size-3.5" /> Aucun visage détecté (ignorées)
              </div>
              <div className="flex flex-wrap gap-2">
                {noFace.map((ph) => (
                  <PhotoTile key={ph.id} photo={ph} color={null} onRemove={() => remove.mutate(ph.id)} />
                ))}
              </div>
            </div>
          )}

          {photos.length + uploading < max && (
            <button
              disabled={!p.consent}
              onClick={() => input.current?.click()}
              {...dropProps}
              className={cx(
                "flex items-center justify-center gap-2 rounded-xl border-2 border-dashed px-4 py-3 text-[13px] transition-colors disabled:opacity-40",
                over ? "border-accent bg-accent-soft text-accent" : "border-line-strong text-muted hover:border-faint hover:text-fg",
              )}
            >
              {uploading ? <Loader2 className="size-4 animate-spin" /> : <ImagePlus className="size-4" />}
              {uploading ? `Analyse de ${uploading} photo${uploading > 1 ? "s" : ""}…` : "Ajouter des photos (rangement automatique)"}
            </button>
          )}
        </div>
      )}
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" multiple className="hidden" onChange={(e) => (pick(e.target.files), (e.target.value = ""))} />

      {(add.error || move.error) && (
        <Notice tone="danger" className="mt-4">
          {(add.error ?? move.error)!.message}
        </Notice>
      )}

      <ul className="mt-auto grid gap-2 pt-5 text-[13px] text-muted sm:grid-cols-2">
        <li className="flex gap-2">
          <ScanFace className="mt-0.5 size-4 shrink-0 text-faint" /> 3 à 10 photos par personne, de face et de trois quarts
        </li>
        <li className="flex gap-2">
          <Sun className="mt-0.5 size-4 shrink-0 text-faint" /> Lumière variée, sans lunettes de soleil
        </li>
      </ul>
    </Card>
  );
}

function ConsentBox({ checked, onChange }: { checked: boolean; onChange: (v: boolean) => void }) {
  return (
    <label
      className={cx(
        "mb-4 flex cursor-pointer items-start gap-3 rounded-xl p-3 ring-1 transition-colors",
        checked ? "bg-accent-soft ring-accent/30" : "bg-raised ring-line hover:ring-line-strong",
      )}
    >
      <input type="checkbox" checked={checked} onChange={(e) => onChange(e.target.checked)} className="peer sr-only" />
      <span
        className={cx(
          "mt-0.5 flex size-5 shrink-0 items-center justify-center rounded-md ring-1 transition-colors peer-focus-visible:ring-2 peer-focus-visible:ring-accent",
          checked ? "bg-accent text-accent-ink ring-accent" : "bg-bg ring-line-strong",
        )}
      >
        {checked && <Check className="size-3.5" strokeWidth={3} />}
      </span>
      <span className="text-[13px] leading-relaxed">
        <span className="flex items-center gap-1.5 font-medium">
          <ShieldCheck className="size-3.5" /> Consentement
        </span>
        <span className="text-muted">Ces photos sont de moi, ou de personnes qui ont donné leur accord pour ce montage.</span>
      </span>
    </label>
  );
}

function PersonGroup({
  person,
  photos,
  others,
  onRemove,
  onMove,
}: {
  person: Person;
  photos: Photo[];
  others: Person[];
  onRemove: (id: string) => void;
  onMove: (id: string, to: string) => void;
}) {
  const color = personColor(person.id);
  return (
    <div className="rounded-xl bg-raised p-3 ring-1 ring-line" style={{ boxShadow: `inset 3px 0 0 ${color}` }}>
      <div className="mb-2.5 flex items-center gap-2 pl-1">
        <PersonBadge id={person.id} />
        <span className="text-sm font-medium">{person.name}</span>
        <span className="text-[12px] text-muted">
          {person.count} photo{person.count > 1 ? "s" : ""}
        </span>
      </div>
      <div className="flex flex-wrap gap-2 pl-1">
        {photos.map((ph) => (
          <PhotoTile key={ph.id} photo={ph} color={color} onRemove={() => onRemove(ph.id)} moveTargets={others} onMove={(to) => onMove(ph.id, to)} />
        ))}
      </div>
    </div>
  );
}

function PhotoTile({
  photo,
  color,
  onRemove,
  moveTargets,
  onMove,
}: {
  photo: Photo;
  color: string | null;
  onRemove: () => void;
  moveTargets?: Person[];
  onMove?: (to: string) => void;
}) {
  const [menu, setMenu] = useState(false);
  return (
    <div className="group relative">
      <div className="size-16 overflow-hidden rounded-lg ring-2" style={{ ["--tw-ring-color" as string]: color ?? "transparent" }} title={photo.name}>
        <img src={photo.crop_url ?? photo.photo_url} alt={photo.name} className={cx("size-full object-cover", !photo.ok && "opacity-40 grayscale")} />
      </div>
      <button
        onClick={onRemove}
        aria-label={`Retirer ${photo.name}`}
        className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-overlay text-fg opacity-0 ring-1 ring-line-strong transition-opacity group-hover:opacity-100 hover:bg-danger hover:text-accent-ink focus-visible:opacity-100"
      >
        <X className="size-3" />
      </button>
      {onMove && (
        <div className="absolute -right-1.5 -bottom-1.5">
          <button
            onClick={() => setMenu((m) => !m)}
            aria-label="Changer de personne"
            aria-expanded={menu}
            className={cx(
              "flex size-5 items-center justify-center rounded-full bg-overlay text-fg ring-1 ring-line-strong transition-opacity hover:bg-fg hover:text-accent-ink focus-visible:opacity-100",
              menu ? "opacity-100" : "opacity-0 group-hover:opacity-100",
            )}
          >
            <MoveRight className="size-3" />
          </button>
          <Menu open={menu} onClose={() => setMenu(false)} className="w-56">
            <div className="px-2.5 pt-1 pb-1.5 text-[11px] font-medium tracking-wide text-faint uppercase">Ce visage est plutôt…</div>
            {moveTargets?.map((o) => (
              <MenuItem
                key={o.id}
                onClick={() => {
                  setMenu(false);
                  onMove(o.id);
                }}
              >
                <PersonBadge id={o.id} /> {o.name}
              </MenuItem>
            ))}
            <MenuItem
              onClick={() => {
                setMenu(false);
                onMove("new");
              }}
            >
              <span className="flex size-5 items-center justify-center rounded-full bg-raised ring-1 ring-line-strong">
                <UserPlus className="size-3" />
              </span>
              Une nouvelle personne
            </MenuItem>
          </Menu>
        </div>
      )}
    </div>
  );
}

export function PersonBadge({ id, size = "sm" }: { id: string | null; size?: "sm" | "md" }) {
  return (
    <span
      className={cx(
        "inline-flex shrink-0 items-center justify-center rounded-full font-mono font-semibold text-accent-ink",
        size === "sm" ? "size-5 text-[10px]" : "size-7 text-[12px]",
      )}
      style={{ background: personColor(id) }}
    >
      {id ?? "–"}
    </span>
  );
}

/* ───────────────────────── Visages du clip et associations ───────────────────────── */

function TargetPanel({
  video,
  selection,
  persons,
  mappings,
  onMappings,
}: {
  video: Video;
  selection: Selection;
  persons: Person[];
  mappings: Mapping[];
  onMappings: (m: Mapping[]) => void;
}) {
  const fps = video.info!.fps;
  const moments = { start: selection.start + 0.1, mid: (selection.start + selection.end) / 2, end: selection.end - 0.2 };
  const [moment, setMoment] = useState<keyof typeof moments>(() => {
    const t = mappings[0]?.t;
    const found = (Object.keys(moments) as (keyof typeof moments)[]).find((k) => t !== undefined && Math.abs(Math.round(moments[k] * fps) / fps - t) < 0.01);
    return found ?? "start";
  });
  const [openMenu, setOpenMenu] = useState<number | null>(null);
  const t = Math.round(moments[moment] * fps) / fps;
  const { data, isFetching, error } = useQuery({
    queryKey: ["faces", video.id, t],
    queryFn: () => api.facesAt(video.id, t),
    staleTime: Infinity,
  });

  // Association automatique (et maintenue quand on ajoute une personne ou qu'on change d'image).
  useEffect(() => {
    if (!data) return;
    const next = reconcile(mappings, data, persons);
    if (!mappingsEqual(next, mappings)) onMappings(next);
  }, [data, persons, mappings, onMappings]);

  const personOf = (i: number): string | null | undefined => {
    if (!data) return undefined;
    return mappings.find((m) => faceIndex(data, m) === i)?.person;
  };

  const assign = (i: number, person: string | null) => {
    if (!data) return;
    const current = personOf(i) ?? null;
    const next = mappings
      .filter((m) => faceIndex(data, m) >= 0)
      .map((m) => {
        // Personne déjà utilisée ailleurs : on échange les deux visages (plus naturel qu'un doublon).
        if (person && m.person === person && faceIndex(data, m) !== i) return { ...m, person: current };
        return m;
      })
      .filter((m) => faceIndex(data, m) !== i);
    next.push({ t: data.t, box: data.faces[i].box, person });
    onMappings(next);
    setOpenMenu(null);
  };

  const swapTwo = () => {
    if (!data) return;
    const assigned = mappings.filter((m) => faceIndex(data, m) >= 0);
    if (assigned.length !== 2) return;
    onMappings([{ ...assigned[0], person: assigned[1].person }, { ...assigned[1], person: assigned[0].person }]);
  };

  const auto = () => data && onMappings(reconcile([], data, persons));
  const canSwap = !!data && mappings.filter((m) => faceIndex(data, m) >= 0).length === 2;

  return (
    <Card className="flex flex-col p-5 sm:p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 font-medium">
          <ScanFace className="size-4.5 text-accent" /> Visages dans le clip
        </h2>
        <SegmentedControl
          value={moment}
          onChange={setMoment}
          options={[
            { value: "start", label: "Début" },
            { value: "mid", label: "Milieu" },
            { value: "end", label: "Fin" },
          ]}
        />
      </div>

      <FrameWithFaces video={video} data={data} loading={isFetching} t={t} fps={fps} personOf={personOf} onPick={(i) => setOpenMenu(i)} />

      {error && (
        <Notice tone="danger" className="mt-4">
          {error.message}
        </Notice>
      )}
      {data && !data.faces.length && (
        <Notice tone="warn" className="mt-4">
          Aucun visage détecté à cet instant. Essaie « Milieu » ou « Fin », ou ajuste le passage.
        </Notice>
      )}

      {data && data.faces.length > 0 && (
        <div className="mt-4">
          <div className="mb-2 flex items-center justify-between">
            <span className="text-[13px] font-medium">Associations</span>
            <div className="flex gap-1">
              {canSwap && (
                <Button variant="ghost" size="sm" onClick={swapTwo} icon={<ArrowLeftRight className="size-3.5" />}>
                  Inverser
                </Button>
              )}
              <Button variant="ghost" size="sm" onClick={auto} disabled={!persons.length} icon={<Sparkles className="size-3.5" />}>
                Auto
              </Button>
            </div>
          </div>
          <ul className="space-y-2">
            {data.faces.map((f, i) => {
              const person = personOf(i);
              const p = persons.find((x) => x.id === person);
              return (
                <li key={i} className="flex items-center gap-3 rounded-xl bg-raised p-2 pr-2.5 ring-1 ring-line">
                  <img src={f.crop} alt="" className="size-11 rounded-lg object-cover" />
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium">Visage {i + 1}</div>
                    <div className="text-[12px] text-muted">{i === 0 ? "le plus grand" : "dans le clip"}</div>
                  </div>
                  <ArrowRight className="ml-auto size-4 shrink-0 text-faint" />
                  <div className="relative">
                    <button
                      onClick={() => setOpenMenu(openMenu === i ? null : i)}
                      aria-expanded={openMenu === i}
                      className={cx(
                        "flex h-11 min-w-44 items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] ring-1 transition-colors",
                        p ? "bg-bg ring-line-strong hover:ring-faint" : "border border-dashed border-line-strong bg-transparent text-muted ring-transparent hover:text-fg",
                      )}
                    >
                      {p ? (
                        <>
                          {p.cover_url ? (
                            <img src={p.cover_url} alt="" className="size-7 rounded-full object-cover ring-2" style={{ ["--tw-ring-color" as string]: personColor(p.id) }} />
                          ) : (
                            <PersonBadge id={p.id} size="md" />
                          )}
                          <span className="flex-1">{p.name}</span>
                        </>
                      ) : (
                        <span className="flex-1">{persons.length ? "Ne pas remplacer" : "Ajoute des photos"}</span>
                      )}
                      <ChevronDown className="size-4 text-faint" />
                    </button>
                    <Menu open={openMenu === i} onClose={() => setOpenMenu(null)} align="right" className="w-60">
                      <div className="px-2.5 pt-1 pb-1.5 text-[11px] font-medium tracking-wide text-faint uppercase">Remplacer le visage {i + 1} par</div>
                      {persons.map((o) => (
                        <MenuItem key={o.id} onClick={() => assign(i, o.id)} active={o.id === person}>
                          {o.cover_url ? <img src={o.cover_url} alt="" className="size-6 rounded-full object-cover" /> : <PersonBadge id={o.id} />}
                          <span className="flex-1">{o.name}</span>
                          <PersonBadge id={o.id} />
                        </MenuItem>
                      ))}
                      <div className="my-1 h-px bg-line" />
                      <MenuItem onClick={() => assign(i, null)} active={person === null}>
                        <span className="flex size-6 items-center justify-center rounded-full border border-dashed border-line-strong">
                          <X className="size-3" />
                        </span>
                        Ne pas remplacer
                      </MenuItem>
                    </Menu>
                  </div>
                </li>
              );
            })}
          </ul>
          {!persons.length && <p className="mt-3 text-[13px] text-muted">Ajoute des photos à gauche : chaque personne sera associée automatiquement à un visage.</p>}
        </div>
      )}
    </Card>
  );
}

function FrameWithFaces({
  video,
  data,
  loading,
  t,
  fps,
  personOf,
  onPick,
}: {
  video: Video;
  data?: FramesFaces;
  loading: boolean;
  t: number;
  fps: number;
  personOf: (i: number) => string | null | undefined;
  onPick: (i: number) => void;
}) {
  return (
    <div className="relative overflow-hidden rounded-xl bg-black ring-1 ring-line" style={{ aspectRatio: `${video.info!.width} / ${video.info!.height}` }}>
      {data && <img src={data.frame} alt="" className="absolute inset-0 size-full object-contain" />}
      {data?.faces.map((f, i) => {
        const [x1, y1, x2, y2] = f.box;
        const person = personOf(i);
        const color = person ? personColor(person) : null;
        return (
          <button
            key={i}
            onClick={() => onPick(i)}
            aria-label={`Visage ${i + 1}`}
            className={cx(
              "absolute rounded-lg transition-[background] duration-150 hover:bg-white/10",
              !color && "outline-2 outline-white/60 outline-dashed",
            )}
            style={{
              left: `${x1 * 100}%`,
              top: `${y1 * 100}%`,
              width: `${(x2 - x1) * 100}%`,
              height: `${(y2 - y1) * 100}%`,
              boxShadow: color ? `0 0 0 3px ${color}, 0 0 24px ${color}55` : undefined,
            }}
          >
            <span
              className="absolute -top-7 left-1/2 flex -translate-x-1/2 items-center gap-1 rounded-md px-1.5 py-0.5 text-[11px] font-semibold whitespace-nowrap"
              style={{ background: color ?? "rgb(0 0 0 / 0.7)", color: color ? "#0a0a0c" : "#ededf0" }}
            >
              {i + 1}
              {person ? ` → ${person}` : person === null ? " · original" : ""}
            </span>
          </button>
        );
      })}
      {loading && (
        <div className="absolute inset-0 flex items-center justify-center bg-black/40 backdrop-blur-[2px]">
          <Loader2 className="size-6 animate-spin text-fg" />
        </div>
      )}
      <span className="absolute bottom-2 left-2 rounded-md bg-black/70 px-2 py-0.5 font-mono text-[11px] tabular">{timecode(t, fps)}</span>
    </div>
  );
}
