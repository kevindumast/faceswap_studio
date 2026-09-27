import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { AnimatePresence, motion } from "motion/react";
import { Check, Copy, Cpu, ExternalLink, Loader2, PersonStanding, PlugZap, Trash2, X, Zap } from "lucide-react";
import { useEffect, useState } from "react";
import { api, type Status } from "../lib/api";
import { duration } from "../lib/time";
import { SpaceLive } from "./SpaceStatus";
import { Button, Notice, ProgressBar, cx } from "./ui";

const DEPLOY_CMD = String.raw`.venv\Scripts\python.exe scripts\deploy_space.py --space ton-pseudo/faceswap-gpu --save`;
const DEPLOY_CHARACTER_CMD = String.raw`.venv\Scripts\python.exe scripts\deploy_space.py --kind character --space ton-pseudo/faceswap-character --save`;

/** Ouvre le panneau depuis n'importe où (ex. le lien « Brancher ZeroGPU » de l'étape Rendu). */
export const openEngineSettings = () => window.dispatchEvent(new Event("open-engine-settings"));

/** Moteur de calcul : ce PC (utilisé par défaut) et le GPU distant ZeroGPU (case à cocher à chaque rendu). */
export function EngineSettings({ open, onClose, status }: { open: boolean; onClose: () => void; status?: Status }) {
  const qc = useQueryClient();
  const settings = useQuery({ queryKey: ["settings"], queryFn: api.settings, enabled: open });
  const z = settings.data?.zerogpu;
  const [space, setSpace] = useState("");
  const [token, setToken] = useState("");
  const [key, setKey] = useState("");
  const [characterSpace, setCharacterSpace] = useState("");
  const [copied, setCopied] = useState(false);

  useEffect(() => {
    if (z?.space) setSpace(z.space);
  }, [z?.space]);
  useEffect(() => {
    setCharacterSpace(z?.character_space ?? "");
  }, [z?.character_space]);
  useEffect(() => {
    if (!open) return;
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose();
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [open, onClose]);

  const refresh = () => {
    qc.invalidateQueries({ queryKey: ["settings"] });
    qc.invalidateQueries({ queryKey: ["status"] });
  };
  const save = useMutation({
    mutationFn: () =>
      api.saveZeroGPU({
        space: space.trim(),
        ...(token ? { token } : {}),
        ...(key ? { key } : {}),
        ...(characterSpace.trim() !== (z?.character_space ?? "") ? { character_space: characterSpace.trim() } : {}),
      }),
    onSuccess: () => {
      setToken("");
      setKey("");
      test.reset();
      testCharacter.reset();
      refresh();
    },
  });
  const clear = useMutation({ mutationFn: api.clearZeroGPU, onSuccess: refresh });
  const test = useMutation({ mutationFn: () => api.testZeroGPU(), onSettled: () => qc.invalidateQueries({ queryKey: ["settings"] }) });
  const testCharacter = useMutation({ mutationFn: () => api.testZeroGPU("character"), onSettled: () => qc.invalidateQueries({ queryKey: ["settings"] }) });

  const engine = status?.engine;
  const used = status?.gpu.used_today_s ?? 0;
  const quota = status?.gpu.free_quota_s ?? 300;

  return (
    <AnimatePresence>
      {open && (
        <>
          <motion.div className="fixed inset-0 z-40 bg-black/50 backdrop-blur-[2px]" initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={onClose} />
          <motion.aside
            role="dialog"
            aria-label="Moteur de calcul"
            className="fixed inset-y-0 right-0 z-50 flex w-full max-w-md flex-col bg-surface shadow-2xl ring-1 ring-line"
            initial={{ x: "100%" }}
            animate={{ x: 0 }}
            exit={{ x: "100%" }}
            transition={{ type: "spring", stiffness: 380, damping: 38 }}
          >
            <header className="flex items-center justify-between border-b border-line px-5 py-4">
              <h2 className="flex items-center gap-2 font-medium">
                <Cpu className="size-4.5 text-accent" /> Moteur de calcul
              </h2>
              <Button variant="ghost" size="sm" onClick={onClose} aria-label="Fermer" className="w-8 px-0">
                <X className="size-4" />
              </Button>
            </header>

            <div className="flex-1 space-y-5 overflow-y-auto p-5">
              {/* Ce PC */}
              <section className="rounded-xl bg-raised p-4 ring-1 ring-line">
                <div className="mb-1 flex items-center justify-between">
                  <span className="text-[12px] font-medium tracking-wide text-faint uppercase">Ce PC · par défaut</span>
                  <span className="rounded-md bg-accent-soft px-1.5 py-0.5 text-[11px] font-medium text-accent">Actif</span>
                </div>
                <div className="font-medium">
                  {engine?.label ?? "CPU"}
                  {engine?.gpus.length ? <span className="text-muted"> · {engine.gpus.join(" + ")}</span> : null}
                </div>
                <p className="mt-1 text-[13px] text-muted">Tous les rendus tournent ici, sauf si tu coches « Utiliser le GPU » pour un rendu précis.</p>
                {engine?.error && (
                  <Notice tone="warn" className="mt-3">
                    {engine.error}
                  </Notice>
                )}
              </section>

              {/* ZeroGPU */}
              <section className="space-y-4">
                <div className="flex items-center justify-between">
                  <h3 className="flex items-center gap-2 font-medium">
                    <Zap className="size-4 text-warn" /> GPU distant · ZeroGPU
                  </h3>
                  {z && (
                    <span
                      className={cx(
                        "rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                        z.tested ? "bg-accent-soft text-accent" : z.configured ? "bg-warn-soft text-warn" : "bg-overlay text-muted",
                      )}
                      title={z.configured && !z.tested ? "Réglages enregistrés : clique sur « Tester la connexion » pour vérifier" : undefined}
                    >
                      {z.tested ? "Connecté" : z.configured ? "Enregistré · à tester" : "Non branché"}
                    </span>
                  )}
                </div>
                <p className="text-[13px] text-muted">
                  Ton propre Space privé sur Hugging Face : GPU de 48 Go, quota de {Math.round(quota / 60)} min/jour en gratuit (40 en PRO). Indispensable
                  pour le niveau 4. Le brancher n'active rien : la case reste à cocher à chaque rendu.
                </p>

                {z?.configured && (
                  <div>
                    <div className="mb-1.5 flex justify-between text-[12px] text-muted">
                      <span>Quota utilisé aujourd'hui (estimation locale)</span>
                      <span className="font-mono tabular">
                        {duration(used)} / {duration(quota)}
                      </span>
                    </div>
                    <ProgressBar value={used / quota} tone={used > quota * 0.8 ? "warn" : "accent"} />
                  </div>
                )}

                {!z?.configured && (
                  <ol className="space-y-3 rounded-xl bg-raised p-4 text-[13px] ring-1 ring-line">
                    <li>
                      <span className="font-medium">1.</span> Crée un jeton <span className="font-mono">write</span> sur{" "}
                      <a href="https://huggingface.co/settings/tokens" target="_blank" rel="noreferrer" className="inline-flex items-center gap-1 text-accent underline-offset-2 hover:underline">
                        huggingface.co/settings/tokens <ExternalLink className="size-3" />
                      </a>
                    </li>
                    <li>
                      <span className="font-medium">2.</span> Dans un terminal, depuis le dossier du projet (remplace <span className="font-mono">ton-pseudo</span>) :
                      <div className="mt-1.5 flex items-start gap-2 rounded-lg bg-bg p-2 ring-1 ring-line">
                        <code className="flex-1 font-mono text-[11px] break-all text-fg">{DEPLOY_CMD}</code>
                        <button
                          onClick={() => {
                            void navigator.clipboard.writeText(DEPLOY_CMD);
                            setCopied(true);
                            setTimeout(() => setCopied(false), 1500);
                          }}
                          aria-label="Copier la commande"
                          className="text-muted hover:text-fg"
                        >
                          {copied ? <Check className="size-3.5 text-accent" /> : <Copy className="size-3.5" />}
                        </button>
                      </div>
                      <span className="mt-1 block text-muted">Il crée ton Space privé, génère la clé et l'enregistre ici tout seul.</span>
                    </li>
                    <li>
                      <span className="font-medium">3.</span> Attends la construction du Space (5 à 15 min), puis « Tester » ci-dessous.
                    </li>
                  </ol>
                )}

                {z?.configured && (
                  <dl className="space-y-2 rounded-xl bg-raised p-4 text-[13px] ring-1 ring-line">
                    <Saved label="Space" value={z.space} />
                    <Saved label="Jeton Hugging Face" value={z.token_set ? `enregistré · ${z.token_hint ?? "hf_…"}` : null} />
                    <Saved label="Clé APP_KEY" value={z.key_set ? "enregistrée" : null} />
                    <p className="pt-1 text-[12px] text-faint">Le jeton et la clé restent sur ce PC et ne sont jamais réaffichés.</p>
                  </dl>
                )}

                <details className="group rounded-xl ring-1 ring-line">
                  <summary className="cursor-pointer list-none px-4 py-3 text-[13px] font-medium">
                    {z?.configured ? "Modifier les réglages" : "Ou saisir les réglages à la main"}
                  </summary>
                  <div className="space-y-3 border-t border-line p-4">
                    {z?.configured && <p className="text-[12px] text-muted">Laisse un champ vide pour garder la valeur enregistrée.</p>}
                    <Field label="Space" value={space} onChange={setSpace} placeholder="ton-pseudo/faceswap-gpu" />
                    <Field label="Jeton Hugging Face" value={token} onChange={setToken} placeholder={z?.token_set ? `${z.token_hint ?? "hf_…"} (inchangé)` : "hf_…"} secret />
                    <Field label="Space du niveau 4 (optionnel)" value={characterSpace} onChange={setCharacterSpace} placeholder="ton-pseudo/faceswap-character" />
                    <Field label="Clé APP_KEY" value={key} onChange={setKey} placeholder={z?.key_set ? "•••••••• (inchangée)" : "affichée par le script"} secret />
                    <div className="flex flex-wrap gap-2">
                      <Button variant="secondary" size="sm" loading={save.isPending} disabled={!space.trim()} onClick={() => save.mutate()}>
                        Enregistrer
                      </Button>
                      {z?.configured && (
                        <Button variant="ghost" size="sm" icon={<Trash2 className="size-3.5" />} onClick={() => window.confirm("Débrancher ZeroGPU ?") && clear.mutate()}>
                          Débrancher
                        </Button>
                      )}
                    </div>
                  </div>
                </details>

                {z?.configured && (
                  <div className="space-y-3">
                    <SpaceLive kind="faces" />
                    <Button variant="primary" size="md" loading={test.isPending} onClick={() => test.mutate()} icon={<PlugZap className="size-4" />}>
                      Tester la connexion
                    </Button>
                    {test.data && (
                      <Notice tone="info">
                        <span className="text-accent">✓ Space joignable</span> en {test.data.latency_ms} ms · version {test.data.version} · niveaux {test.data.levels.join(", ")}
                        {!test.data.zerogpu && " · (tourne hors ZeroGPU)"}
                      </Notice>
                    )}
                    {test.isPending && (
                      <p className="flex items-center gap-2 text-[12px] text-muted">
                        <Loader2 className="size-3.5 animate-spin" /> Si le Space dormait, le réveil prend 1 à 2 min.
                      </p>
                    )}
                  </div>
                )}
                {(save.error || test.error || clear.error) && <Notice tone="danger">{(save.error ?? test.error ?? clear.error)!.message}</Notice>}
              </section>

              {/* Niveau 4 : 2e Space, même jeton et même clé */}
              <section className="space-y-4 border-t border-line pt-5">
                <div className="flex items-center justify-between">
                  <h3 className="flex items-center gap-2 font-medium">
                    <PersonStanding className="size-4 text-warn" /> Niveau 4 · personne entière
                  </h3>
                  {z && (
                    <span
                      className={cx(
                        "rounded-md px-1.5 py-0.5 text-[11px] font-medium",
                        z.character_tested ? "bg-accent-soft text-accent" : z.character_configured ? "bg-warn-soft text-warn" : "bg-overlay text-muted",
                      )}
                    >
                      {z.character_tested ? "Connecté" : z.character_configured ? "Enregistré · à tester" : "Non branché"}
                    </span>
                  )}
                </div>
                <p className="text-[13px] text-muted">
                  Un 2e Space privé avec Wan2.2-Animate-14B : remplace toute la personne, habits et gestuelle compris. Environ 2 min de GPU pour 5 s
                  en 360p (ton quota de {Math.round(quota / 60)} min/jour est partagé avec l'autre Space). Un compte gratuit peut héberger ces 2 Spaces.
                </p>
                {z?.character_configured ? (
                  <dl className="space-y-2 rounded-xl bg-raised p-4 text-[13px] ring-1 ring-line">
                    <Saved label="Space" value={z.character_space} />
                    <p className="pt-1 text-[12px] text-faint">Même jeton et même clé APP_KEY que ci-dessus.</p>
                  </dl>
                ) : (
                  <ol className="space-y-3 rounded-xl bg-raised p-4 text-[13px] ring-1 ring-line">
                    <li>
                      <span className="font-medium">1.</span> Dans un terminal, depuis le dossier du projet (remplace <span className="font-mono">ton-pseudo</span>, même jeton{" "}
                      <span className="font-mono">write</span>) :
                      <Command cmd={DEPLOY_CHARACTER_CMD} />
                    </li>
                    <li>
                      <span className="font-medium">2.</span> Le premier démarrage est long (installation + 57 Go de modèle) : compte 20 à 40 min, puis « Tester ».
                    </li>
                  </ol>
                )}
                {z?.character_configured && (
                  <div className="space-y-3">
                    <SpaceLive kind="character" />
                    <Button variant="secondary" size="md" loading={testCharacter.isPending} onClick={() => testCharacter.mutate()} icon={<PlugZap className="size-4" />}>
                      Tester le Space du niveau 4
                    </Button>
                    {testCharacter.data && (
                      <Notice tone="info">
                        <span className="text-accent">✓ Space joignable</span> en {testCharacter.data.latency_ms} ms · version {testCharacter.data.version}
                        {!testCharacter.data.zerogpu && " · (tourne hors ZeroGPU)"}
                      </Notice>
                    )}
                    {testCharacter.isPending && (
                      <p className="flex items-center gap-2 text-[12px] text-muted">
                        <Loader2 className="size-3.5 animate-spin" /> S'il dormait, le Space recharge son modèle : jusqu'à plusieurs minutes.
                      </p>
                    )}
                    {testCharacter.error && <Notice tone="danger">{testCharacter.error.message}</Notice>}
                  </div>
                )}
              </section>
            </div>
          </motion.aside>
        </>
      )}
    </AnimatePresence>
  );
}

function Command({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false);
  return (
    <div className="mt-1.5 flex items-start gap-2 rounded-lg bg-bg p-2 ring-1 ring-line">
      <code className="flex-1 font-mono text-[11px] break-all text-fg">{cmd}</code>
      <button
        onClick={() => {
          void navigator.clipboard.writeText(cmd);
          setCopied(true);
          setTimeout(() => setCopied(false), 1500);
        }}
        aria-label="Copier la commande"
        className="text-muted hover:text-fg"
      >
        {copied ? <Check className="size-3.5 text-accent" /> : <Copy className="size-3.5" />}
      </button>
    </div>
  );
}

function Saved({ label, value }: { label: string; value: string | null }) {
  return (
    <div className="flex items-center justify-between gap-3">
      <dt className="text-muted">{label}</dt>
      <dd className={cx("flex items-center gap-1.5 truncate font-mono text-[12px]", value ? "text-fg" : "text-danger")}>
        {value ? <Check className="size-3.5 shrink-0 text-accent" /> : <X className="size-3.5 shrink-0" />}
        {value ?? "manquant"}
      </dd>
    </div>
  );
}

function Field({ label, value, onChange, placeholder, secret }: { label: string; value: string; onChange: (v: string) => void; placeholder: string; secret?: boolean }) {
  return (
    <label className="block">
      <span className="mb-1 block text-[12px] text-muted">{label}</span>
      <input
        type={secret ? "password" : "text"}
        value={value}
        onChange={(e) => onChange(e.target.value)}
        placeholder={placeholder}
        autoComplete="off"
        spellCheck={false}
        className="h-9 w-full rounded-lg bg-bg px-3 font-mono text-[12px] ring-1 ring-line outline-none placeholder:font-sans placeholder:text-faint focus:ring-accent"
      />
    </label>
  );
}
