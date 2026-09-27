import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import {
  AlertTriangle,
  ArrowLeftRight,
  ArrowRight,
  BookUser,
  Check,
  ChevronDown,
  ImagePlus,
  Loader2,
  MoveRight,
  Pencil,
  ScanFace,
  Search,
  ShieldCheck,
  Sparkles,
  Sun,
  UserMinus,
  UserPlus,
  Users,
  X,
} from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type FaceSet, type ImportResult, type Level, type Mapping, type PassageScan, type Person, type Photo, type Status, type Video } from "../lib/api";
import { faceIndex, mappingsEqual, reconcile, styleAt, styleOf, type PersonStyle } from "../lib/people";
import { timecode } from "../lib/time";
import { LevelPicker } from "../components/LevelPicker";
import { Button, Card, Menu, MenuItem, Notice, SectionTitle, cx } from "../components/ui";
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
  level: Level;
  onLevel: (l: Level) => void;
  onNext: () => void;
};

export function FacesStep(p: Props) {
  const faceSet = useQuery({
    queryKey: ["faceset", p.faceSetId],
    queryFn: () => api.faceSet(p.faceSetId!),
    enabled: !!p.faceSetId,
    retry: false,
  });
  // Session supprimée (nettoyage automatique) : on repart d'une session vide. La bibliothèque, elle, reste.
  const { onFaceSet } = p;
  useEffect(() => {
    if (faceSet.error) onFaceSet(null);
  }, [faceSet.error, onFaceSet]);

  const persons = faceSet.data?.persons ?? [];
  const active = p.mappings.filter((m) => m.person).length;
  const levelReady = p.status?.levels[p.level]?.ready ?? p.level === "face";
  const canGo = p.consent && persons.length > 0 && active > 0 && levelReady;

  return (
    <div>
      <SectionTitle
        eyebrow="Étape 3 · Visages"
        title="Qui remplace qui ?"
        subtitle="Choisis jusqu'où va la transformation, puis les personnes de ta bibliothèque (ou de nouvelles photos). Elles sont associées aux personnes trouvées dans le passage."
        right={
          <Button variant="primary" size="lg" disabled={!canGo} onClick={p.onNext} icon={<ArrowRight className="size-4" />}>
            Continuer
          </Button>
        }
      />
      <LevelPicker value={p.level} onChange={p.onLevel} status={p.status} />
      <div className="grid gap-5 lg:grid-cols-[minmax(0,1fr)_minmax(0,1.1fr)]">
        <SourcePanel {...p} data={faceSet.data} />
        <TargetPanel video={p.video} selection={p.selection} persons={persons} mappings={p.mappings} onMappings={p.onMappings} />
      </div>
    </div>
  );
}

/* ───────────────────────── Personnes de la vidéo (issues de la bibliothèque) ───────────────────────── */

function SourcePanel(p: Props & { data?: FaceSet }) {
  const qc = useQueryClient();
  const input = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);
  const [uploading, setUploading] = useState(0);
  const [imported, setImported] = useState<ImportResult[] | null>(null);
  const [editing, setEditing] = useState<string | null>(null);
  const max = p.status?.photos_max ?? 20;
  const persons = p.data?.persons ?? [];
  const rejected = p.data?.rejected ?? [];

  const applySet = (set: FaceSet) => {
    qc.setQueryData(["faceset", set.id], set);
    qc.invalidateQueries({ queryKey: ["people"] });
    p.onFaceSet(set.id);
  };
  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["faceset", p.faceSetId] });
    qc.invalidateQueries({ queryKey: ["people"] });
  };
  const sessionId = async () => p.faceSetId ?? (await api.createFaceSet()).id;

  const upload = useMutation({
    mutationFn: async (files: File[]) => (p.faceSetId ? api.addPhotos(p.faceSetId, files) : api.createFaceSet(files)),
    onMutate: (files) => setUploading(files.length),
    onSettled: () => setUploading(0),
    onSuccess: (set) => {
      applySet(set);
      const results = set.imported ?? [];
      setImported(results);
      const firstNew = results.find((r) => r.ok && r.created);
      if (firstNew?.person_id) setEditing(firstNew.person_id); // nouvelle personne : on propose tout de suite de la nommer
    },
  });
  const addFromLibrary = useMutation({ mutationFn: async (pid: string) => api.addPersonToSet(await sessionId(), pid), onSuccess: applySet });
  const removeFromVideo = useMutation({ mutationFn: (pid: string) => api.removePersonFromSet(p.faceSetId!, pid), onSuccess: applySet });
  const rename = useMutation({
    mutationFn: ({ pid, name }: { pid: string; name: string }) => api.renamePerson(pid, name),
    onSuccess: () => {
      setEditing(null);
      refresh();
    },
  });
  const move = useMutation({
    mutationFn: async ({ pid, photo, to }: { pid: string; photo: string; to: string }) => {
      const r = await api.movePhoto(pid, photo, to);
      if (to === "new" && p.faceSetId) await api.addPersonToSet(p.faceSetId, r.moved_to);
      return r;
    },
    onSuccess: refresh,
  });
  const deleteRejected = useMutation({ mutationFn: (rid: string) => api.deleteRejected(p.faceSetId!, rid), onSuccess: applySet });

  const pick = (list: FileList | null) => {
    if (!list || !p.consent) return;
    const files = Array.from(list).filter((f) => f.type.startsWith("image/")).slice(0, max);
    if (files.length) upload.mutate(files);
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
  const error = upload.error ?? addFromLibrary.error ?? rename.error ?? move.error;

  return (
    <Card className="flex flex-col p-5 sm:p-6">
      <div className="mb-4 flex items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 font-medium">
          <Users className="size-4.5 text-accent" /> Pour cette vidéo
        </h2>
        <LibraryPicker exclude={new Set(persons.map((x) => x.id))} onPick={(pid) => addFromLibrary.mutate(pid)} loading={addFromLibrary.isPending} />
      </div>

      <ConsentBox checked={p.consent} onChange={p.onConsent} />

      <div className="flex flex-col gap-3">
        <AnimatePresence initial={false}>
          {persons.map((person, i) => (
            <motion.div key={person.id} layout initial={{ opacity: 0, y: 6 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0 }}>
              <PersonCard
                person={person}
                style={styleAt(i)}
                others={persons.filter((o) => o.id !== person.id)}
                editing={editing === person.id}
                onEdit={() => setEditing(person.id)}
                onCancelEdit={() => setEditing(null)}
                onRename={(name) => rename.mutate({ pid: person.id, name })}
                onRemove={() => removeFromVideo.mutate(person.id)}
                onMove={(photo, to) => move.mutate({ pid: person.id, photo, to })}
                onDeletePhoto={() => removeFromVideo.mutate(person.id)}
              />
            </motion.div>
          ))}
        </AnimatePresence>

        {imported && imported.length > 0 && <ImportSummary results={imported} onClose={() => setImported(null)} />}

        {rejected.length > 0 && (
          <div className="rounded-xl bg-warn-soft/50 p-3 ring-1 ring-warn/20">
            <div className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-warn">
              <AlertTriangle className="size-3.5" /> Aucun visage détecté (ignorées)
            </div>
            <div className="flex flex-wrap gap-2">
              {rejected.map((r) => (
                <div key={r.id} className="group relative size-16 overflow-hidden rounded-lg" title={r.name}>
                  <img src={r.photo_url} alt={r.name} className="size-full object-cover opacity-40 grayscale" />
                  <button
                    onClick={() => deleteRejected.mutate(r.id)}
                    aria-label={`Retirer ${r.name}`}
                    className="absolute top-1 right-1 flex size-5 items-center justify-center rounded-full bg-overlay opacity-0 ring-1 ring-line-strong group-hover:opacity-100"
                  >
                    <X className="size-3" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        <button
          disabled={!p.consent || uploading > 0}
          onClick={() => input.current?.click()}
          {...dropProps}
          className={cx(
            "flex flex-col items-center justify-center gap-1.5 rounded-xl border-2 border-dashed px-4 text-center transition-colors disabled:cursor-not-allowed disabled:opacity-40",
            persons.length === 0 ? "py-8" : "py-3",
            over ? "border-accent bg-accent-soft text-accent" : "border-line-strong text-muted hover:border-faint hover:text-fg",
          )}
        >
          {uploading ? <Loader2 className="size-5 animate-spin" /> : <ImagePlus className="size-5" />}
          <span className="text-[13px] font-medium">
            {uploading ? `Analyse de ${uploading} photo${uploading > 1 ? "s" : ""}…` : "Ajouter des photos"}
          </span>
          {persons.length === 0 && !uploading && (
            <span className="max-w-xs text-[12px] text-muted">
              Les visages déjà connus sont reconnus et rangés dans leur fiche ; les nouveaux créent une personne dans ta bibliothèque.
            </span>
          )}
        </button>
      </div>
      <input ref={input} type="file" accept="image/jpeg,image/png,image/webp" multiple className="hidden" onChange={(e) => (pick(e.target.files), (e.target.value = ""))} />

      {error && (
        <Notice tone="danger" className="mt-4">
          {error.message}
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

function ImportSummary({ results, onClose }: { results: ImportResult[]; onClose: () => void }) {
  const known = new Map<string, number>();
  const created = new Map<string, number>();
  let none = 0;
  for (const r of results) {
    if (!r.ok) none++;
    else (r.created ? created : known).set(r.person_name!, ((r.created ? created : known).get(r.person_name!) ?? 0) + 1);
  }
  const plural = (n: number) => `${n} photo${n > 1 ? "s" : ""}`;
  return (
    <div className="flex items-start gap-3 rounded-xl bg-raised p-3 text-[13px] ring-1 ring-line">
      <Sparkles className="mt-0.5 size-4 shrink-0 text-accent" />
      <ul className="flex-1 space-y-0.5">
        {[...known].map(([name, n]) => (
          <li key={`k${name}`}>
            <span className="text-accent">Reconnu</span> : {name} (+{plural(n)})
          </li>
        ))}
        {[...created].map(([name, n]) => (
          <li key={`c${name}`}>
            <span className="text-fg">Nouvelle personne</span> : {name} ({plural(n)}) <span className="text-muted">— donne-lui un nom</span>
          </li>
        ))}
        {none > 0 && <li className="text-warn">{plural(none)} sans visage détecté</li>}
      </ul>
      <button onClick={onClose} aria-label="Fermer" className="text-muted hover:text-fg">
        <X className="size-4" />
      </button>
    </div>
  );
}

/** Ajouter à la vidéo quelqu'un de la bibliothèque, avec recherche par nom. */
function LibraryPicker({ exclude, onPick, loading }: { exclude: Set<string>; onPick: (pid: string) => void; loading: boolean }) {
  const [open, setOpen] = useState(false);
  const [q, setQ] = useState("");
  const { data, isFetching } = useQuery({ queryKey: ["people", q], queryFn: () => api.people(q), enabled: open });
  const list = (data ?? []).filter((p) => !exclude.has(p.id));
  return (
    <div className="relative">
      <Button variant="secondary" size="sm" loading={loading} onClick={() => setOpen((o) => !o)} aria-expanded={open} icon={<BookUser className="size-3.5" />}>
        Depuis ma bibliothèque
      </Button>
      <Menu open={open} onClose={() => setOpen(false)} align="right" className="w-80">
        <div className="relative mb-1.5">
          <Search className="pointer-events-none absolute top-1/2 left-2.5 size-3.5 -translate-y-1/2 text-faint" />
          <input
            autoFocus
            value={q}
            onChange={(e) => setQ(e.target.value)}
            placeholder="Chercher un nom…"
            className="h-9 w-full rounded-lg bg-bg pr-2 pl-8 text-[13px] ring-1 ring-line outline-none placeholder:text-faint focus:ring-accent"
          />
        </div>
        <div className="max-h-72 overflow-y-auto">
          {isFetching && !data && <Loader2 className="mx-auto my-4 size-4 animate-spin text-muted" />}
          {data && !list.length && (
            <p className="px-2.5 py-3 text-[13px] text-muted">
              {q ? "Personne à ce nom." : data.length ? "Tout le monde est déjà dans cette vidéo." : "Ta bibliothèque est vide : ajoute des photos."}
            </p>
          )}
          {list.map((person) => (
            <MenuItem
              key={person.id}
              onClick={() => {
                setOpen(false);
                setQ("");
                onPick(person.id);
              }}
            >
              {person.cover_url ? <img src={person.cover_url} alt="" className="size-8 rounded-full object-cover" /> : <span className="size-8 rounded-full bg-raised" />}
              <span className="min-w-0 flex-1">
                <span className="block truncate font-medium">{person.name}</span>
                <span className="text-[11px] text-muted">
                  {person.count} photo{person.count > 1 ? "s" : ""}
                </span>
              </span>
              <UserPlus className="size-3.5 text-faint" />
            </MenuItem>
          ))}
        </div>
      </Menu>
    </div>
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

function PersonCard(p: {
  person: Person;
  style: PersonStyle;
  others: Person[];
  editing: boolean;
  onEdit: () => void;
  onCancelEdit: () => void;
  onRename: (name: string) => void;
  onRemove: () => void;
  onMove: (photo: string, to: string) => void;
  onDeletePhoto: (photo: string) => void;
}) {
  return (
    <div className="rounded-xl bg-raised p-3 ring-1 ring-line" style={{ boxShadow: `inset 3px 0 0 ${p.style.color}` }}>
      <div className="mb-2.5 flex items-center gap-2 pl-1">
        <PersonBadge style={p.style} />
        {p.editing ? (
          <NameInput initial={p.person.name} onSave={p.onRename} onCancel={p.onCancelEdit} />
        ) : (
          <button onClick={p.onEdit} className="group flex min-w-0 items-center gap-1.5 text-left" title="Renommer">
            <span className="truncate text-sm font-medium">{p.person.name}</span>
            <Pencil className="size-3 shrink-0 text-faint opacity-0 transition-opacity group-hover:opacity-100" />
          </button>
        )}
        <span className="shrink-0 text-[12px] text-muted">
          {p.person.count} photo{p.person.count > 1 ? "s" : ""}
        </span>
        <button
          onClick={p.onRemove}
          title="Retirer de cette vidéo (reste dans ta bibliothèque)"
          aria-label={`Retirer ${p.person.name} de cette vidéo`}
          className="ml-auto flex size-7 items-center justify-center rounded-lg text-faint transition-colors hover:bg-overlay hover:text-fg"
        >
          <UserMinus className="size-3.5" />
        </button>
      </div>
      <div className="flex flex-wrap gap-2 pl-1">
        {p.person.photos.map((ph) => (
          <PhotoTile key={ph.id} photo={ph} color={p.style.color} moveTargets={p.others} onMove={(to) => p.onMove(ph.id, to)} onDelete={() => p.onDeletePhoto(ph.id)} />
        ))}
      </div>
    </div>
  );
}

function NameInput({ initial, onSave, onCancel }: { initial: string; onSave: (name: string) => void; onCancel: () => void }) {
  const [value, setValue] = useState(initial);
  const done = useRef(false);
  const commit = () => {
    if (done.current) return;
    done.current = true;
    const name = value.trim();
    if (name && name !== initial) onSave(name);
    else onCancel();
  };
  return (
    <input
      autoFocus
      value={value}
      maxLength={40}
      onFocus={(e) => e.target.select()}
      onChange={(e) => setValue(e.target.value)}
      onBlur={commit}
      onKeyDown={(e) => {
        if (e.key === "Enter") commit();
        if (e.key === "Escape") {
          done.current = true;
          onCancel();
        }
      }}
      aria-label="Nom de la personne"
      className="h-7 min-w-0 flex-1 rounded-md bg-bg px-2 text-sm font-medium ring-1 ring-accent outline-none"
    />
  );
}

function PhotoTile({ photo, color, moveTargets, onMove, onDelete }: { photo: Photo; color: string; moveTargets: Person[]; onMove: (to: string) => void; onDelete: () => void }) {
  const [menu, setMenu] = useState(false);
  return (
    <div className="group relative">
      <div className="size-16 overflow-hidden rounded-lg ring-2" style={{ ["--tw-ring-color" as string]: color }} title={photo.name}>
        <img src={photo.crop_url} alt={photo.name} className="size-full object-cover" />
      </div>
      <button
        onClick={() => window.confirm("Retirer cette personne de cette vidéo ? Elle reste dans ta bibliothèque.") && onDelete()}
        aria-label="Retirer de cette vidéo"
        className="absolute -top-1.5 -right-1.5 flex size-5 items-center justify-center rounded-full bg-overlay text-fg opacity-0 ring-1 ring-line-strong transition-opacity group-hover:opacity-100 hover:bg-danger hover:text-accent-ink focus-visible:opacity-100"
      >
        <X className="size-3" />
      </button>
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
          {moveTargets.map((o) => (
            <MenuItem
              key={o.id}
              onClick={() => {
                setMenu(false);
                onMove(o.id);
              }}
            >
              {o.cover_url ? <img src={o.cover_url} alt="" className="size-5 rounded-full object-cover" /> : null}
              {o.name}
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
    </div>
  );
}

export function PersonBadge({ style, size = "sm" }: { style: PersonStyle; size?: "sm" | "md" }) {
  return (
    <span
      className={cx(
        "inline-flex shrink-0 items-center justify-center rounded-full font-mono font-semibold text-accent-ink",
        size === "sm" ? "size-5 text-[10px]" : "size-7 text-[12px]",
      )}
      style={{ background: style.color }}
    >
      {style.letter}
    </span>
  );
}

/* ───────────────────────── Personnes trouvées dans le passage ───────────────────────── */

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
  const [focus, setFocus] = useState(0);
  const [openMenu, setOpenMenu] = useState<number | null>(null);
  const { data: scan, isFetching, error } = useQuery({
    queryKey: ["scan", video.id, selection.start, selection.end],
    queryFn: () => api.scan(video.id, selection.start, selection.end),
    staleTime: Infinity,
    retry: false,
  });

  // Association automatique (et maintenue quand on ajoute une personne ou qu'on change de passage).
  useEffect(() => {
    if (!scan) return;
    const next = reconcile(mappings, scan, persons);
    if (!mappingsEqual(next, mappings)) onMappings(next);
  }, [scan, persons, mappings, onMappings]);

  const personOf = (i: number): string | null | undefined => (scan ? mappings.find((m) => faceIndex(scan, m) === i)?.person : undefined);

  // Une même personne peut remplacer plusieurs visages : la recherche coupe parfois un acteur en deux
  // (profil, éclairage…), et il faut pouvoir lui donner la même personne aux deux endroits.
  const assign = (i: number, person: string | null) => {
    if (!scan) return;
    const next = mappings.filter((m) => {
      const k = faceIndex(scan, m);
      return k >= 0 && k !== i;
    });
    next.push({ t: scan.faces[i].t, box: scan.faces[i].box, person });
    onMappings(next);
    setOpenMenu(null);
  };
  const assigned = scan ? mappings.filter((m) => faceIndex(scan, m) >= 0 && m.person) : [];
  const canSwap = assigned.length === 2 && assigned[0].person !== assigned[1].person;
  const swapTwo = () => {
    if (!canSwap) return;
    const others = mappings.filter((m) => !assigned.includes(m));
    onMappings([...others, { ...assigned[0], person: assigned[1].person }, { ...assigned[1], person: assigned[0].person }]);
  };
  const auto = () => scan && onMappings(reconcile([], scan, persons));

  const focused = scan?.faces[Math.min(focus, (scan?.faces.length ?? 1) - 1)];

  return (
    <Card className="flex flex-col p-5 sm:p-6">
      <div className="mb-4 flex flex-wrap items-center justify-between gap-3">
        <h2 className="flex items-center gap-2 font-medium">
          <ScanFace className="size-4.5 text-accent" /> Dans le passage
        </h2>
        {scan && (
          <span className="text-[12px] text-muted">
            {scan.faces.length} personne{scan.faces.length > 1 ? "s" : ""} · {scan.samples} images analysées
          </span>
        )}
      </div>

      <div className="relative overflow-hidden rounded-xl bg-black ring-1 ring-line" style={{ aspectRatio: `${video.info!.width} / ${video.info!.height}` }}>
        {scan && focused && <PassageFrame scan={scan} focusedT={focused.t} persons={persons} personOf={personOf} onPick={(i) => (setFocus(i), setOpenMenu(i))} />}
        {isFetching && (
          <div className="absolute inset-0 flex flex-col items-center justify-center gap-2 bg-black/50 text-[13px] backdrop-blur-[2px]">
            <Loader2 className="size-6 animate-spin text-fg" />
            Recherche des personnes dans tout le passage…
          </div>
        )}
        {focused && <span className="absolute bottom-2 left-2 rounded-md bg-black/70 px-2 py-0.5 font-mono text-[11px] tabular">{timecode(focused.t, fps)}</span>}
      </div>

      {error && (
        <Notice tone="danger" className="mt-4">
          {error.message}
        </Notice>
      )}
      {scan && !scan.faces.length && (
        <Notice tone="warn" className="mt-4">
          Aucun visage trouvé dans ce passage, même sur {scan.samples} images. Choisis un autre passage à l'étape 2.
        </Notice>
      )}

      {scan && scan.faces.length > 0 && (
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
            {scan.faces.map((f, i) => {
              const person = personOf(i);
              const pr = persons.find((x) => x.id === person);
              const st = styleOf(persons, person);
              return (
                <li
                  key={i}
                  onClick={() => setFocus(i)}
                  className={cx("flex cursor-pointer items-center gap-3 rounded-xl bg-raised p-2 pr-2.5 ring-1 transition-colors", focus === i ? "ring-line-strong" : "ring-line")}
                >
                  <img src={f.crop} alt="" className="size-11 rounded-lg object-cover" />
                  <div className="min-w-0">
                    <div className="text-[13px] font-medium">Visage {i + 1}</div>
                    <div className="text-[12px] text-muted">
                      {f.maybe_same !== null ? (
                        <span className="text-warn">Vu 1 fois · sans doute le visage {f.maybe_same + 1} de profil</span>
                      ) : (
                        `vu sur ${f.seen}/${scan.samples} images`
                      )}
                    </div>
                  </div>
                  <ArrowRight className="ml-auto size-4 shrink-0 text-faint" />
                  <div className="relative" onClick={(e) => e.stopPropagation()}>
                    <button
                      onClick={() => setOpenMenu(openMenu === i ? null : i)}
                      aria-expanded={openMenu === i}
                      className={cx(
                        "flex h-11 min-w-44 items-center gap-2.5 rounded-lg px-2.5 text-left text-[13px] ring-1 transition-colors",
                        pr ? "bg-bg ring-line-strong hover:ring-faint" : "border border-dashed border-line-strong bg-transparent text-muted ring-transparent hover:text-fg",
                      )}
                    >
                      {pr ? (
                        <>
                          {pr.cover_url ? (
                            <img src={pr.cover_url} alt="" className="size-7 rounded-full object-cover ring-2" style={{ ["--tw-ring-color" as string]: st.color }} />
                          ) : (
                            <PersonBadge style={st} size="md" />
                          )}
                          <span className="min-w-0 flex-1 truncate">{pr.name}</span>
                        </>
                      ) : (
                        <span className="flex-1">{persons.length ? "Ne pas remplacer" : "Ajoute des personnes"}</span>
                      )}
                      <ChevronDown className="size-4 text-faint" />
                    </button>
                    <Menu open={openMenu === i} onClose={() => setOpenMenu(null)} align="right" className="w-64">
                      <div className="px-2.5 pt-1 pb-1.5 text-[11px] font-medium tracking-wide text-faint uppercase">Remplacer le visage {i + 1} par</div>
                      {persons.map((o, k) => (
                        <MenuItem key={o.id} onClick={() => assign(i, o.id)} active={o.id === person}>
                          {o.cover_url ? <img src={o.cover_url} alt="" className="size-6 rounded-full object-cover" /> : <PersonBadge style={styleAt(k)} />}
                          <span className="min-w-0 flex-1 truncate">{o.name}</span>
                          <PersonBadge style={styleAt(k)} />
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
          {!persons.length && <p className="mt-3 text-[13px] text-muted">Ajoute des personnes à gauche : elles seront associées automatiquement aux visages trouvés.</p>}
        </div>
      )}
    </Card>
  );
}

/** Image où la personne sélectionnée apparaît le mieux, avec les visages trouvés sur cette même image. */
function PassageFrame({
  scan,
  focusedT,
  persons,
  personOf,
  onPick,
}: {
  scan: PassageScan;
  focusedT: number;
  persons: Person[];
  personOf: (i: number) => string | null | undefined;
  onPick: (i: number) => void;
}) {
  const frame = scan.frames[focusedT.toFixed(3)];
  return (
    <>
      {frame && <img src={frame} alt="" className="absolute inset-0 size-full object-contain" />}
      {scan.faces.map((f, i) => {
        if (Math.abs(f.t - focusedT) > 0.001) return null;
        const [x1, y1, x2, y2] = f.box;
        const person = personOf(i);
        const st = styleOf(persons, person);
        const color = person ? st.color : null;
        return (
          <button
            key={i}
            onClick={() => onPick(i)}
            aria-label={`Visage ${i + 1}`}
            className={cx("absolute rounded-lg transition-[background] duration-150 hover:bg-white/10", !color && "outline-2 outline-white/60 outline-dashed")}
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
              {person ? ` → ${st.letter}` : person === null ? " · original" : ""}
            </span>
          </button>
        );
      })}
    </>
  );
}
