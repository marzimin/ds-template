/**
 * Thin wrapper around fetch, plus the types the rest of the app uses.
 *
 * Every type below is pulled from `schema.d.ts`, which is generated from the
 * backend's OpenAPI document. Nothing here is hand-written, so renaming a field
 * in a Python Pydantic class breaks the build at the exact line that needs
 * updating rather than silently producing `undefined` at runtime.
 *
 * Regenerate after changing the API with: `make types`
 */
import type { components } from './schema';

type Schemas = components['schemas'];

export type HealthResponse = Schemas['HealthResponse'];
export type FeatureSpec = Schemas['FeatureSpecResponse'];
export type PredictSchema = Schemas['PredictSchemaResponse'];
export type PredictResponse = Schemas['PredictResponse'];
export type ModelReloadResponse = Schemas['ModelReloadResponse'];
export type RunSummary = Schemas['RunSummary'];
export type RunDetail = Schemas['RunDetail'];
export type ArtifactEntry = Schemas['ArtifactEntry'];

/**
 * A failed HTTP call, carrying the status so callers can distinguish causes.
 *
 * The status matters: 503 means "no model trained yet", which is an expected
 * state on a fresh checkout and should be shown as guidance rather than as an
 * error. See `isNoModelError`.
 */
export class ApiError extends Error {
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

/** True when the backend reports that no model has been trained yet. */
export function isNoModelError(error: unknown): boolean {
  return error instanceof ApiError && error.status === 503;
}

/**
 * Requests are relative ('/api/...'), so the browser sends them to whatever
 * origin serves the page. In development Vite proxies them to the backend; in
 * production both are served from one host. Neither case needs a base URL
 * compiled into the bundle.
 */
async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const response = await fetch(path, {
    ...init,
    headers: { 'Content-Type': 'application/json', ...(init?.headers ?? {}) },
  });

  if (!response.ok) {
    throw new ApiError(response.status, await readErrorDetail(response));
  }

  return (await response.json()) as T;
}

/**
 * Pull the human-readable message out of a failed response.
 *
 * FastAPI reports errors as `{"detail": ...}`, where detail is a string for the
 * errors we raise and an array of field errors for its own validation
 * failures. Both are handled so the user never sees "[object Object]".
 */
async function readErrorDetail(response: Response): Promise<string> {
  try {
    const body: unknown = await response.json();
    const detail = (body as { detail?: unknown })?.detail;

    if (typeof detail === 'string') return detail;
    if (Array.isArray(detail)) {
      return detail
        .map((item) => {
          const entry = item as { loc?: unknown[]; msg?: string };
          const field = Array.isArray(entry.loc) ? entry.loc.join('.') : '';
          return field ? `${field}: ${entry.msg ?? 'invalid'}` : (entry.msg ?? 'invalid');
        })
        .join('; ');
    }
    if (detail) return JSON.stringify(detail);
  } catch {
    // Body was not JSON; fall through to the status text.
  }
  return `${response.status} ${response.statusText}`;
}

export const api = {
  health: () => request<HealthResponse>('/api/health'),

  predictSchema: () => request<PredictSchema>('/api/predict/schema'),

  predict: (features: Record<string, unknown>) =>
    request<PredictResponse>('/api/predict', {
      method: 'POST',
      body: JSON.stringify({ features }),
    }),

  reloadModel: () => request<ModelReloadResponse>('/api/predict/reload', { method: 'POST' }),

  runs: (limit = 50) => request<RunSummary[]>(`/api/runs?limit=${limit}`),

  run: (runId: string) => request<RunDetail>(`/api/runs/${encodeURIComponent(runId)}`),

  artifacts: (runId: string, path = '') => {
    const query = path ? `?path=${encodeURIComponent(path)}` : '';
    return request<ArtifactEntry[]>(`/api/runs/${encodeURIComponent(runId)}/artifacts${query}`);
  },
};

/** URL for an artifact file, used directly as an <img src>. */
export function artifactFileUrl(runId: string, path: string): string {
  return `/api/runs/${encodeURIComponent(runId)}/artifacts/file?path=${encodeURIComponent(path)}`;
}
