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

export type Photo = { id: string; name: string; ok: boolean; person: string | null; crop_url: string | null; photo_url: string };
export type Person = { id: string; name: string; count: number; cover_url: string | null };
export type FaceSet = { id: string; photos: Photo[]; persons: Person[]; ok_count: number };

export type Box = [number, number, number, number];
export type DetectedFace = { box: Box; score: number; crop: string };
export type FramesFaces = { t: number; width: number; height: number; frame: string; faces: DetectedFace[] };

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
  resolution?: "360p" | "480p";
};

export type Job = {
  id: string;
  video_id: string;
  video_title: string | null;
  face_set_id: string;
  params: JobParams;
  status: "queued" | "running" | "cancelling" | "cancelled" | "done" | "error";
  stage: "cut" | "swap" | "assemble" | null;
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
  upload_max_mb: number;
  video_ext: string[];
  sec_per_frame: number;
  levels: Record<Level, LevelStatus>;
  gpu: { configured: boolean };
  example_url: string;
  youtube_max_duration_s: number;
};

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

  createFaceSet: (files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    form.append("consent", "true");
    return upload<FaceSet>("/api/faces", form);
  },
  addPhotos: (setId: string, files: File[]) => {
    const form = new FormData();
    files.forEach((f) => form.append("files", f));
    return upload<FaceSet>(`/api/faces/${setId}/photos`, form);
  },
  faceSet: (setId: string) => request<FaceSet>("GET", `/api/faces/${setId}`),
  deletePhoto: (setId: string, photoId: string) =>
    request<FaceSet>("DELETE", `/api/faces/${setId}/photos/${photoId}`),
  movePhoto: (setId: string, photoId: string, person: string | "new") =>
    request<FaceSet>("PATCH", `/api/faces/${setId}/photos/${photoId}`, { person }),

  jobs: () => request<Job[]>("GET", "/api/jobs"),
  job: (id: string) => request<Job>("GET", `/api/jobs/${id}`),
  createJob: (body: JobParams & { video_id: string; face_set_id: string; consent: boolean }) =>
    request<Job>("POST", "/api/jobs", body),
  cancelJob: (id: string) => request<Job>("POST", `/api/jobs/${id}/cancel`),
  deleteJob: (id: string) => request<{ ok: boolean }>("DELETE", `/api/jobs/${id}`),
};
