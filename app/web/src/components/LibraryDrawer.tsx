import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { AlertTriangle, BookUser, Check, ChevronDown, ImagePlus, Loader2, Pencil, PersonStanding, Search, ShieldCheck, Trash2, UserPlus, X } from "lucide-react";
import { useEffect, useRef, useState } from "react";
import { api, type ImportResult, type PendingPhoto, type Person } from "../lib/api";
import { FRAMING_TEXT, PhotoViewer } from "./PhotoViewer";
import { Button, Notice, cx } from "./ui";

/** Bibliothèque de personnes : toutes les têtes importées, nommées, réutilisables dans chaque vidéo. */
export function LibraryDrawer({ open, onClose }: { open: boolean; onClose: () => void }) {
  const qc = useQueryClient();
  const [q, setQ] = useState("");
  const [expanded, setExpanded] = useState<string | null>(null);
  const [consent, setConsent] = useState(false);
  const [lastImport, setLastImport] = useState<ImportResult[] | null>(null);
  const input = useRef<HTMLInputElement>(null);
  const { data, isLoading } = useQuery({ queryKey: ["people", q], queryFn: () => api.people(q), enabled: open });
  const { data: pending } = useQuery({ queryKey: ["people-pending"], queryFn: api.pendingPhotos, enabled: open });

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["people"] });
    qc.invalidateQueries({ queryKey: ["faceset"] });
  };
  const refreshPending = () => qc.invalidateQueries({ queryKey: ["people-pending"] });
  const importPhotos = useMutation({
    mutationFn: (files: File[]) => api.importToLibrary(files),
    onSuccess: (r) => {
      setLastImport(r.imported);
      refresh();
      refreshPending();
    },
  });
  const rename = useMutation({ mutationFn: ({ pid, name }: { pid: string; name: string }) => api.renamePerson(pid, name), onSuccess: refresh });
  const deletePerson = useMutation({ mutationFn: api.deletePerson, onSuccess: refresh });
  const deletePhoto = useMutation({ mutationFn: ({ pid, photo }: { pid: string; photo: string }) => api.deletePhoto(pid, photo), onSuccess: refresh });
  const deletePending = useMutation({ mutationFn: (rid: string) => api.deletePending(rid), onSuccess: refreshPending });
  const namePending = useMutation({
    mutationFn: ({ rid, name }: { rid: string; name: string }) => api.assignPending(rid, "new", name),
    onSuccess: () => {
      refresh();
      refreshPending();
    },
  });

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const people = data ?? [];
  const error = importPhotos.error ?? rename.error ?? deletePerson.error ?? deletePhoto.error ?? namePending.error;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside
            role="dialog"
            aria-label="Bibliothèque de personnes"
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col bg-surface shadow-2xl ring-1 ring-line"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 380, damping: 38 }}
          >
            <header className="flex items-center justify-between border-b border-line px-5 py-4">
              <h2 className="flex items-center gap-2 font-medium">
                <BookUser className="size-4.5 text-accent" /> Personnes
                {data && <span className="text-[12px] font-normal text-muted">· {people.length}</span>}
              </h2>
              <Button variant="ghost" size="sm" onClick={onClose} aria-label="Fermer" className="w-8 px-0">
                <X className="size-4" />
              </Button>
            </header>

            <div className="space-y-3 border-b border-line p-4">
              <div className="relative">
                <Search className="pointer-events-none absolute top-1/2 left-3 size-4 -translate-y-1/2 text-faint" />
                <input
                  value={q}
                  onChange={(e) => setQ(e.target.value)}
                  placeholder="Chercher un nom…"
                  className="h-10 w-full rounded-xl bg-bg pr-3 pl-9 text-sm ring-1 ring-line outline-none placeholder:text-faint focus:ring-accent"
                />
              </div>
              <div className="flex items-center gap-3">
                <label className="flex flex-1 cursor-pointer items-center gap-2 text-[12px] text-muted">
                  <input type="checkbox" checked={consent} onChange={(e) => setConsent(e.target.checked)} className="peer sr-only" />
                  <span className={cx("flex size-4 shrink-0 items-center justify-center rounded ring-1", consent ? "bg-accent text-accent-ink ring-accent" : "bg-bg ring-line-strong")}>
                    {consent && <Check className="size-3" strokeWidth={3} />}
                  </span>
                  <ShieldCheck className="size-3.5 shrink-0" /> Moi ou des personnes d'accord
                </label>
                <Button
                  variant="secondary"
                  size="sm"
                  disabled={!consent}
                  loading={importPhotos.isPending}
                  onClick={() => input.current?.click()}
                  icon={<ImagePlus className="size-3.5" />}
                >
                  Ajouter des photos
                </Button>
                <input
                  ref={input}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  multiple
                  className="hidden"
                  onChange={(e) => {
                    const files = Array.from(e.target.files ?? []);
                    if (files.length) importPhotos.mutate(files);
                    e.target.value = "";
                  }}
                />
              </div>
              {lastImport && (
                <p className="text-[12px] text-muted">
                  {lastImport.filter((r) => r.ok && !r.created).length} reconnue(s), {lastImport.filter((r) => r.ok && r.created).length} nouvelle(s),{" "}
                  {lastImport.filter((r) => !r.ok).length} sans visage.
                </p>
              )}
              {error && <Notice tone="danger">{error.message}</Notice>}
            </div>

            <div className="flex-1 overflow-y-auto p-3">
              {!!pending?.length && (
                <div className="mb-3 rounded-xl bg-warn-soft/50 p-3 ring-1 ring-warn/20">
                  <div className="mb-2 flex items-center gap-1.5 text-[12px] font-medium text-warn">
                    <AlertTriangle className="size-3.5" /> Aucun visage détecté automatiquement
                  </div>
                  <ul className="space-y-2">
                    {pending.map((ph) => (
                      <PendingRow
                        key={ph.id}
                        photo={ph}
                        loading={namePending.isPending && namePending.variables?.rid === ph.id}
                        onName={(name) => namePending.mutate({ rid: ph.id, name })}
                        onDelete={() => deletePending.mutate(ph.id)}
                      />
                    ))}
                  </ul>
                </div>
              )}
              {isLoading && <Loader2 className="mx-auto mt-10 size-5 animate-spin text-muted" />}
              {data && !people.length && (
                <p className="mt-10 px-6 text-center text-sm text-muted">
                  {q ? "Personne à ce nom." : "Ta bibliothèque est vide. Les photos importées à l'étape Visages y arrivent automatiquement."}
                </p>
              )}
              <ul className="space-y-2">
                {people.map((person) => (
                  <PersonRow
                    key={person.id}
                    person={person}
                    expanded={expanded === person.id}
                    onToggle={() => setExpanded(expanded === person.id ? null : person.id)}
                    onRename={(name) => rename.mutate({ pid: person.id, name })}
                    onDelete={() =>
                      window.confirm(`Supprimer ${person.name} et ses ${person.count} photo(s) de ta bibliothèque ?`) && deletePerson.mutate(person.id)
                    }
                    onDeletePhoto={(photo) => window.confirm("Supprimer cette photo ?") && deletePhoto.mutate({ pid: person.id, photo })}
                  />
                ))}
              </ul>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

/** Photo importée sans visage auto-détecté : on peut la retirer, ou la nommer pour en faire une personne. */
function PendingRow(p: { photo: PendingPhoto; loading: boolean; onName: (name: string) => void; onDelete: () => void }) {
  const [naming, setNaming] = useState(false);
  const [name, setName] = useState("");
  const submit = () => {
    const clean = name.trim();
    if (clean) p.onName(clean);
  };
  return (
    <li className="flex items-center gap-2.5">
      <img src={p.photo.photo_url} alt={p.photo.name} className="size-12 shrink-0 rounded-lg object-cover opacity-60 grayscale" />
      {naming ? (
        <form
          className="flex flex-1 gap-1.5"
          onSubmit={(e) => {
            e.preventDefault();
            submit();
          }}
        >
          <input
            autoFocus
            value={name}
            maxLength={40}
            onChange={(e) => setName(e.target.value)}
            onBlur={() => !name.trim() && setNaming(false)}
            placeholder="Nom de la personne…"
            className="h-8 min-w-0 flex-1 rounded-lg bg-bg px-2.5 text-[13px] ring-1 ring-accent outline-none"
          />
          <Button type="submit" size="sm" variant="primary" loading={p.loading} disabled={!name.trim()} icon={<Check className="size-3.5" />} />
        </form>
      ) : (
        <>
          <Button size="sm" variant="secondary" onClick={() => setNaming(true)} icon={<UserPlus className="size-3.5" />} className="flex-1">
            Nommer
          </Button>
          <button onClick={p.onDelete} aria-label={`Retirer ${p.photo.name}`} className="flex size-8 shrink-0 items-center justify-center rounded-lg text-muted hover:bg-danger-soft hover:text-danger">
            <X className="size-3.5" />
          </button>
        </>
      )}
    </li>
  );
}

function PersonRow(p: {
  person: Person;
  expanded: boolean;
  onToggle: () => void;
  onRename: (name: string) => void;
  onDelete: () => void;
  onDeletePhoto: (photo: string) => void;
}) {
  const [editing, setEditing] = useState(false);
  const [name, setName] = useState(p.person.name);
  const [viewing, setViewing] = useState<number | null>(null);
  // Cadrage des photos (en pied ou non), calculé une fois par photo côté serveur.
  const framing = useQuery({
    queryKey: ["framing", p.person.id, p.person.photos.map((ph) => ph.id).join(",")],
    queryFn: () => api.framing(p.person.id),
    enabled: p.expanded || viewing != null,
    staleTime: Infinity,
  });
  const updated = p.person.updated_at ? new Date(p.person.updated_at * 1000).toLocaleDateString("fr-FR", { day: "2-digit", month: "2-digit" }) : null;
  const save = () => {
    setEditing(false);
    const clean = name.trim();
    if (clean && clean !== p.person.name) p.onRename(clean);
    else setName(p.person.name);
  };
  return (
    <li className="rounded-xl bg-raised ring-1 ring-line">
      <div className="flex items-center gap-3 p-2.5">
        {p.person.cover_url ? <img src={p.person.cover_url} alt="" className="size-12 rounded-xl object-cover" /> : <span className="size-12 rounded-xl bg-overlay" />}
        <div className="min-w-0 flex-1">
          {editing ? (
            <input
              autoFocus
              value={name}
              maxLength={40}
              onFocus={(e) => e.target.select()}
              onChange={(e) => setName(e.target.value)}
              onBlur={save}
              onKeyDown={(e) => {
                if (e.key === "Enter") save();
                if (e.key === "Escape") {
                  setName(p.person.name);
                  setEditing(false);
                }
              }}
              aria-label="Nom"
              className="h-7 w-full rounded-md bg-bg px-2 text-sm font-medium ring-1 ring-accent outline-none"
            />
          ) : (
            <button onClick={() => setEditing(true)} className="group flex max-w-full items-center gap-1.5 text-left" title="Renommer">
              <span className="truncate text-sm font-medium">{p.person.name}</span>
              <Pencil className="size-3 shrink-0 text-faint opacity-0 group-hover:opacity-100" />
            </button>
          )}
          <div className="text-[12px] text-muted">
            {p.person.count} photo{p.person.count > 1 ? "s" : ""}
            {updated ? ` · modifiée le ${updated}` : ""}
          </div>
        </div>
        <button onClick={p.onToggle} aria-expanded={p.expanded} aria-label="Voir les photos" className="flex size-8 items-center justify-center rounded-lg text-muted hover:bg-overlay hover:text-fg">
          <ChevronDown className={cx("size-4 transition-transform", p.expanded && "rotate-180")} />
        </button>
        <button onClick={p.onDelete} aria-label={`Supprimer ${p.person.name}`} className="flex size-8 items-center justify-center rounded-lg text-muted hover:bg-danger-soft hover:text-danger">
          <Trash2 className="size-3.5" />
        </button>
      </div>
      <AnimatePresence initial={false}>
        {p.expanded && (
          <motion.div initial={{ height: 0, opacity: 0 }} animate={{ height: "auto", opacity: 1 }} exit={{ height: 0, opacity: 0 }} className="overflow-hidden">
            <div className="flex flex-wrap gap-2 border-t border-line p-2.5">
              {p.person.photos.map((ph, i) => {
                const info = framing.data?.photos[ph.id];
                const isReference = framing.data?.reference === ph.id;
                return (
                  <div
                    key={ph.id}
                    className={cx("group relative size-16 overflow-hidden rounded-lg", isReference && "ring-2 ring-accent")}
                    title={`${ph.name}${info ? ` · ${FRAMING_TEXT[info.framing].label}` : ""}${isReference ? " · utilisée au niveau 4" : ""}`}
                  >
                    <button onClick={() => setViewing(i)} aria-label={`Voir ${ph.name} en grand`} className="size-full cursor-zoom-in">
                      <img src={ph.crop_url} alt={ph.name} className="size-full object-cover" />
                    </button>
                    {info && (info.framing === "full" || info.framing === "half") && (
                      <span
                        className={cx(
                          "pointer-events-none absolute bottom-1 left-1 flex size-5 items-center justify-center rounded-full ring-1 ring-black/20",
                          info.framing === "full" ? "bg-accent text-accent-ink" : "bg-warn text-accent-ink",
                        )}
                      >
                        <PersonStanding className="size-3" />
                      </span>
                    )}
                    <button
                      onClick={() => p.onDeletePhoto(ph.id)}
                      aria-label={`Supprimer ${ph.name}`}
                      className="absolute top-1 right-1 flex size-5 items-center justify-center rounded-full bg-overlay opacity-0 ring-1 ring-line-strong group-hover:opacity-100 hover:bg-danger hover:text-accent-ink"
                    >
                      <X className="size-3" />
                    </button>
                  </div>
                );
              })}
            </div>
            <p className="px-2.5 pb-2.5 text-[11px] text-faint">
              Vignettes recadrées sur le visage : clique pour voir la photo entière. <PersonStanding className="inline size-3" /> = en pied (vert) ou à
              mi-corps (orange) ; la photo entourée est celle que le niveau 4 utilisera.
            </p>
          </motion.div>
        )}
      </AnimatePresence>
      <PhotoViewer photos={p.person.photos} index={viewing} framing={framing.data} onIndex={setViewing} onClose={() => setViewing(null)} />
    </li>
  );
}
