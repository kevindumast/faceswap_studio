// Client de l'API FastAPI (même origine : proxy Vite en dev, FastAPI sert le build en prod).

export type VideoInfo = {
  duration: number;
  width: number;
  height: number;
  fps: number;
  fps_str: string;
  has_audio: boolean;
  codec: string;
};

export type Filmstrip = {
  count: number;
  cols: number;
  rows: number;
  tile_w: number;
  tile_h: number;
  interval: number;
};

export type Video = {
  id: string;
  kind: "upload" | "url";
  title: string | null;
  url: string | null;
  status: "downloading" | "preparing" | "ready" | "error";
  progress: number;
  error: string | null;
  info: VideoInfo | null;
  filmstrip: Filmstrip | null;
  proxy_url: string | null;
  filmstrip_url: string | null;
  poster_url: string | null;
  created_at: number;
};

export type UrlInfo = {
  id: string;
  title: string;
  duration: number;
  thumbnail: string | null;
  uploader: string | null;
  webpage_url: string;
  too_long: boolean;
  max_duration_s: number;
};

/** Photo d'une personne de la bibliothèque (toujours avec un visage détecté). */
/** crop_url : visage recadré (vignette) ; full_url : photo entière 1024 px (celle qu'utilise le niveau 4). */
export type Photo = { id: string; name: string; crop_url: string; photo_url: string; full_url: string };
/** Cadrage d'une photo pour le niveau 4 : de la tête aux pieds, mi-corps, portrait, ou visage introuvable. */
export type Framing = "full" | "half" | "portrait" | "none";
export type PersonFraming = { reference: string | null; photos: Record<string, { face_ratio: number | null; framing: Framing }> };
/** Personne de la bibliothèque : permanente, réutilisable dans toutes les vidéos. */
export type Person = { id: string; name: string; count: number; cover_url: string | null; photos: Photo[]; created_at?: number; updated_at?: number };
export type RejectedPhoto = { id: string; name: string; photo_url: string };
/** Résultat de l'import d'une photo : reconnue dans une personne existante, nouvelle personne, ou pas de visage. */
export type ImportResult = { ok: boolean; name: string; person_id?: string; person_name?: string; created?: boolean; pending_id?: string };
/** Photo importée dans la bibliothèque sans visage auto-détecté : en attente d'un rattachement manuel. */
export type PendingPhoto = { id: string; name: string; added_at: number; photo_url: string; full_url: string };
/** Session d'une vidéo : les personnes de la bibliothèque choisies pour ce rendu. */
export type FaceSet = { id: string; persons: Person[]; rejected: RejectedPhoto[]; ok_count: number; legacy?: boolean; imported?: ImportResult[] };

export type Box = [number, number, number, number];
export type DetectedFace = { box: Box; score: number; crop: string };
export type FramesFaces = { t: number; width: number; height: number; frame: string; faces: DetectedFace[] };
/** Une personne vue dans le passage (plusieurs images analysées), représentée par sa meilleure apparition. */
export type ScanFace = { t: number; box: Box; score: number; seen: number; height_px: number; maybe_same: number | null; crop: string };
export type PassageScan = { start: number; end: number; samples: number; faces: ScanFace[]; frames: Record<string, string>; width: number; height: number };

export type Target = { t: number; box: Box };
/** Un visage du clip → une personne source. person = null : visage laissé intact (choix explicite). */
export type Mapping = Target & { person: string | null };

export type Level = "face" | "face_tone" | "head" | "character";

export type LevelStatus = {
  available: boolean;
  gpu_only: boolean;
  ready: boolean;
  missing_groups: string[];
  install_mb: number;
  installing: { running: boolean; progress: number } | null;
  install_error: string | null;
  sec_per_frame: number;
};

export type JobParams = {
  start: number;
  end: number;
  output: "segment" | "full";
  stabilize: boolean;
  ai_label: boolean;
  mappings: (Target & { person: string })[];
  target?: Target | null;
  level: Level;
  use_gpu: boolean;
  resolution?: Resolution;
  limit_fps?: boolean;
};

/** Niveau 4 : résolution à laquelle la personne est générée sur le Space (puis recollée sur la vidéo). */
export type Resolution = "360p" | "480p";

export type Job = {
  id: string;
  video_id: string;
  video_title: string | null;
  face_set_id: string;
  params: JobParams;
  status: "queued" | "running" | "cancelling" | "cancelled" | "done" | "error" | "pausing" | "paused";
  /** false pour ZeroGPU : le Space calcule tout l'extrait d'un coup. */
  pausable: boolean;
  /** Rendu distant : wake (secondes écoulées / durée typique), queue (rang / taille), gpu, puis étapes du Space. */
  stage: "cut" | "wake" | "queue" | "gpu" | "swap" | "pose" | "mask" | "generate" | "assemble" | null;
  done: number;
  total: number;
  elapsed: number | null;
  eta: number | null;
  sec_per_frame: number | null;
  queue_ahead: number;
  error: string | null;
  warnings: string[];
  preview_url: string;
  result_url: string | null;
  partial_url: string | null; // en pause : ce qui est déjà rendu
  before_url: string | null;
  created_at: number;
};

export type Engine = { accelerator: "cpu" | "dml" | "cuda"; label: string; gpus: string[]; error: string | null };

export type Status = {
  device: string;
  engine: Engine;
  ffmpeg: boolean;
  models: boolean;
  worker: boolean;
  segment: { min_s: number; max_s: number };
  photos_max: number;
  fps_cap: number;
  upload_max_mb: number;
  video_ext: string[];
  sec_per_frame: number;
  levels: Record<Level, LevelStatus>;
  gpu: {
    configured: boolean;
    space: string | null;
    used_today_s: number;
    free_quota_s: number;
    sec_per_frame: Partial<Record<Level, number>>;
    /** Niveau 4 : Space dédié (Wan2.2-Animate) et de quoi estimer son temps de GPU. */
    character: { configured: boolean; space: string | null; max_s: number; steps: number; gpu_s_per_second: Record<Resolution, number> };
  };
  example_url: string;
  youtube_max_duration_s: number;
};

export type ZeroGPUSettings = {
  space: string | null;
  token_set: boolean;
  key_set: boolean;
  /** « hf_…AB12 » : 4 derniers caractères seulement. */
  token_hint: string | null;
  configured: boolean;
  /** Dernier test de connexion réussi avec ces réglages. */
  tested: boolean;
  /** Space du niveau 4 (même jeton, même clé). */
  character_space: string | null;
  character_configured: boolean;
  character_tested: boolean;
};
export type SpaceKind = "faces" | "character";
/** État d'un Space lu chez Hugging Face (sans quota) ; waking_since = début du réveil (secondes epoch). */
export type SpaceState = {
  space: string | null;
  phase: "ready" | "starting" | "asleep" | "error" | "unknown" | "unconfigured";
  stage: string | null;
  hardware: string | null;
  error: string | null;
  waking_since: number | null;
  expected_s: number;
};
export type ZeroGPUTest = { ok: boolean; latency_ms: number; version: string; levels: Level[]; zerogpu: boolean };

export class ApiError extends Error {
  status: number;
  constructor(status: number, message: string) {
    super(message);
    this.status = status;
  }
}

async function parseError(res: Response): Promise<ApiError> {
  let message = `Erreur ${res.status}`;
  try {
    const body = await res.json();
    if (typeof body.detail === "string") message = body.detail;
    else if (Array.isArray(body.detail)) message = body.detail.map((d: { msg: string }) => d.msg).join(" · ");
  } catch {
    /* corps non JSON */
  }
  return new ApiError(res.status, message);
}

async function request<T>(method: string, url: string, body?: unknown): Promise<T> {
  const res = await fetch(url, {
    method,
    headers: body instanceof FormData || body === undefined ? undefined : { "Content-Type": "application/json" },
    body: body instanceof FormData ? body : body === undefined ? undefined : JSON.stringify(body),
  });
  if (!res.ok) throw await parseError(res);
  // Page HTML au lieu de données : en pratique, l'API tourne avec une ancienne version qui ne connaît pas la route.
  if (!(res.headers.get("content-type") ?? "").includes("application/json")) {
    throw new ApiError(res.status, "L'API ne connaît pas cette fonction : elle tourne sans doute avec une ancienne version. Relance l'API et le worker (DEMARRAGE.md).");
  }
  return res.json() as Promise<T>;
}

/** Upload avec progression (fetch ne remonte pas la progression d'envoi). */
function upload<T>(url: string, form: FormData, onProgress?: (frac: number) => void): Promise<T> {
  return new Promise((resolve, reject) => {
    const xhr = new XMLHttpRequest();
    xhr.open("POST", url);
    xhr.upload.onprogress = (e) => e.lengthComputable && onProgress?.(e.loaded / e.total);
    xhr.onload = () => {
      let body: { detail?: unknown } = {};
      try {
        body = JSON.parse(xhr.responseText);
      } catch {
        /* ignore */
      }
      if (xhr.status >= 200 && xhr.status < 300) resolve(body as T);
      else reject(new ApiError(xhr.status, typeof body.detail === "string" ? body.detail : `Erreur ${xhr.status}`));
    };
    xhr.onerror = () => reject(new ApiError(0, "Serveur injoignable."));
    xhr.send(form);
  });
}

export const api = {
  status: () => request<Status>("GET", "/api/status"),
  installModels: (group: string) => request<{ ready: boolean }>("POST", `/api/models/${group}/download`),

  videos: () => request<Video[]>("GET", "/api/videos"),
  video: (id: string) => request<Video>("GET", `/api/videos/${id}`),
  urlInfo: (url: string) => request<UrlInfo>("POST", "/api/videos/url/info", { url }),
  fromUrl: (url: string) => request<Video>("POST", "/api/videos/url", { url }),
  uploadVideo: (file: File, onProgress?: (f: number) => void) => {
    const form = new FormData();
    form.append("file", file);
    return upload<Video>("/api/videos/upload", form, onProgress);
  },
  deleteVideo: (id: string) => request<{ ok: boolean }>("DELETE", `/api/videos/${id}`),
  facesAt: (id: string, t: number) => request<FramesFaces>("GET", `/api/videos/${id}/faces?t=${t.toFixed(3)}`),

  scan: (id: string, start: number, end: number) =>
    request<PassageScan>("GET", `/api/videos/${id}/scan?start=${start.toFixed(2)}&end=${end.toFixed(2)}`),

  // Session de la vidéo (personnes choisies pour ce rendu)
  createFaceSet: (files: File[] = []) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("consent", "true");
    return upload<FaceSet>("/api/faces", form);
  },
  addPhotos: (setId: string, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("consent", "true");
    return upload<FaceSet>(`/api/faces/${setId}/photos`, form);
  },
  faceSet: (setId: string) => request<FaceSet>("GET", `/api/faces/${setId}`),
  addPersonToSet: (setId: string, personId: string) => request<FaceSet>("POST", `/api/faces/${setId}/people`, { person_id: personId }),
  removePersonFromSet: (setId: string, personId: string) => request<FaceSet>("DELETE", `/api/faces/${setId}/people/${personId}`),
  deleteRejected: (setId: string, rid: string) => request<FaceSet>("DELETE", `/api/faces/${setId}/rejected/${rid}`),

  // Bibliothèque de personnes
  people: (q = "") => request<Person[]>("GET", `/api/people${q ? `?q=${encodeURIComponent(q)}` : ""}`),
  person: (pid: string) => request<Person>("GET", `/api/people/${pid}`),
  renamePerson: (pid: string, name: string) => request<Person>("PATCH", `/api/people/${pid}`, { name }),
  deletePerson: (pid: string) => request<{ ok: boolean }>("DELETE", `/api/people/${pid}`),
  movePhoto: (pid: string, photoId: string, person: string | "new") =>
    request<{ moved_to: string; source_exists: boolean }>("PATCH", `/api/people/${pid}/photos/${photoId}`, { person }),
  framing: (pid: string) => request<PersonFraming>("GET", `/api/people/${pid}/framing`),
  deletePhoto: (pid: string, photoId: string) =>
    request<{ ok: boolean; person_exists: boolean }>("DELETE", `/api/people/${pid}/photos/${photoId}`),
  importToLibrary: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("consent", "true");
    return upload<{ imported: ImportResult[]; people: Person[] }>("/api/people/photos", form);
  },
  pendingPhotos: () => request<PendingPhoto[]>("GET", "/api/people/photos/pending"),
  deletePending: (rid: string) => request<{ ok: boolean }>("DELETE", `/api/people/photos/pending/${rid}`),
  assignPending: (rid: string, target: string, name?: string) =>
    request<Person>("POST", `/api/people/photos/pending/${rid}/assign`, { target, name }),

  // Moteur : Space ZeroGPU (le jeton et la clé ne reviennent jamais au navigateur)
  settings: () => request<{ zerogpu: ZeroGPUSettings }>("GET", "/api/settings"),
  saveZeroGPU: (body: { space: string; token?: string; key?: string; character_space?: string }) =>
    request<{ zerogpu: ZeroGPUSettings }>("PUT", "/api/settings/zerogpu", body),
  clearZeroGPU: () => request<{ zerogpu: ZeroGPUSettings }>("DELETE", "/api/settings/zerogpu"),
  spaceState: (kind: SpaceKind) => request<SpaceState>("GET", `/api/settings/zerogpu/state?kind=${kind}`),
  wakeSpace: (kind: SpaceKind) => request<SpaceState>("POST", `/api/settings/zerogpu/wake?kind=${kind}`),
  testZeroGPU: (kind: "faces" | "character" = "faces") =>
    request<ZeroGPUTest>("POST", `/api/settings/zerogpu/test${kind === "character" ? "?kind=character" : ""}`),

  jobs: () => request<Job[]>("GET", "/api/jobs"),
  job: (id: string) => request<Job>("GET", `/api/jobs/${id}`),
  createJob: (body: JobParams & { video_id: string; face_set_id: string; consent: boolean }) =>
    request<Job>("POST", "/api/jobs", body),
  cancelJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/cancel`),
  pauseJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/pause`),
  resumeJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/resume`),
  deleteJob: (id: string) => request<{ ok: boolean }>("DELETE", `/api/jobs/${id}`),
};
