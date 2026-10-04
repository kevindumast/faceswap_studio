import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { Download, History, Loader2, Trash2, X } from "lucide-react";
import { useEffect } from "react";
import { api, type Job } from "../lib/api";
import { parseStage } from "../lib/stages";
import { seconds } from "../lib/time";
import { Button, cx } from "./ui";

const LABEL: Record<Job["status"], string> = {
  queued: "En attente",
  running: "En cours",
  cancelling: "Annulation",
  pausing: "Pause…",
  paused: "En pause",
  review: "À vérifier",
  cancelled: "Annulé",
  done: "Terminé",
  error: "Erreur",
};

export function HistoryDrawer({ open, onClose, onOpenJob }: { open: boolean; onClose: () => void; onOpenJob: (j: Job) => void }) {
  const qc = useQueryClient();
  const { data, isLoading } = useQuery({
    queryKey: ["jobs"],
    queryFn: api.jobs,
    enabled: open,
    refetchInterval: open ? 3000 : false,
  });
  const del = useMutation({ mutationFn: api.deleteJob, onSuccess: () => qc.invalidateQueries({ queryKey: ["jobs"] }) });

  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside
            role="dialog"
            aria-label="Historique des rendus"
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col bg-surface shadow-2xl ring-1 ring-line"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 380, damping: 38 }}
          >
            <header className="flex items-center justify-between border-b border-line px-5 py-4">
              <h2 className="flex items-center gap-2 font-medium">
                <History className="size-4.5 text-accent" /> Historique
              </h2>
              <Button variant="ghost" size="sm" onClick={onClose} aria-label="Fermer" className="w-8 px-0">
                <X className="size-4" />
              </Button>
            </header>
            <div className="flex-1 overflow-y-auto p-3">
              {isLoading && <Loader2 className="mx-auto mt-10 size-5 animate-spin text-muted" />}
              {data && !data.length && <p className="mt-10 text-center text-sm text-muted">Aucun rendu pour l'instant.</p>}
              <ul className="space-y-2">
                {data?.map((j) => (
                  <li key={j.id} className="group flex items-center gap-3 rounded-xl bg-raised p-2.5 ring-1 ring-line transition-colors hover:ring-line-strong">
                    <button
                      onClick={() => onOpenJob(j)}
                      disabled={j.video_missing}
                      title={j.video_missing ? "Vidéo source supprimée : ce rendu reste téléchargeable mais ne peut plus être rouvert." : undefined}
                      className="flex min-w-0 flex-1 items-center gap-3 text-left disabled:cursor-default"
                    >
                      <div className="relative aspect-video w-24 shrink-0 overflow-hidden rounded-lg bg-black">
                        <img
                          src={j.video_missing ? j.preview_url : `/api/videos/${j.video_id}/poster.jpg`}
                          alt=""
                          className="size-full object-cover"
                          onError={(e) => (e.currentTarget.style.visibility = "hidden")}
                        />
                      </div>
                      <div className="min-w-0">
                        <div className="truncate text-[13px] font-medium">{j.video_title ?? "Vidéo sans titre"}</div>
                        <div className="mt-0.5 font-mono text-[11px] text-muted">
                          {seconds(j.params.end - j.params.start)} · {j.params.output === "full" ? "complète" : "extrait"}
                          {j.video_missing && " · source supprimée"}
                        </div>
                        <span
                          className={cx(
                            "mt-1 inline-block rounded-md px-1.5 py-0.5 text-[10px] font-semibold uppercase",
                            j.status === "done" && "bg-accent-soft text-accent",
                            (j.status === "running" || j.status === "queued") && "bg-overlay text-fg",
                            j.status === "error" && "bg-danger-soft text-danger",
                            (j.status === "cancelled" || j.status === "cancelling") && "bg-overlay text-muted",
                            j.status === "review" && "bg-warn-soft text-warn",
                          )}
                        >
                          {LABEL[j.status]}
                          {j.status === "running" && parseStage(j.stage).key === "wake" ? " · réveil du Space" : ""}
                          {j.status === "running" && j.total && ["swap", "pose", "mask", "generate", "face"].includes(parseStage(j.stage).key ?? "")
                            ? ` ${Math.round((j.done / j.total) * 100)} %`
                            : ""}
                        </span>
                      </div>
                    </button>
                    <div className="flex flex-col gap-1">
                      {j.result_url && (
                        <a href={`${j.result_url}?download=1`} download aria-label="Télécharger" className="flex size-7 items-center justify-center rounded-lg text-muted hover:bg-overlay hover:text-fg">
                          <Download className="size-3.5" />
                        </a>
                      )}
                      {!["running", "cancelling", "pausing"].includes(j.status) && (
                        <button
                          onClick={() => window.confirm(`Supprimer ce rendu${j.video_title ? ` de « ${j.video_title} »` : ""} ? La vidéo finale sera effacée.`) && del.mutate(j.id)}
                          aria-label="Supprimer" className="flex size-7 items-center justify-center rounded-lg text-muted hover:bg-danger-soft hover:text-danger">
                          <Trash2 className="size-3.5" />
                        </button>
                      )}
                    </div>
                  </li>
                ))}
              </ul>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}
