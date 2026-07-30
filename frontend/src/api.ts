import type { ApiErrorBody, DetectionResult, HealthStatus } from './types'

/**
 * Base path (or URL) of the API.
 *
 * Empty by default, which means same-origin at the root — correct when FastAPI
 * serves this bundle itself.
 *
 * On the Vercel deployment this is "/api": both halves live behind one domain,
 * where a rewrite routes /api/* to the Python service. It must match the
 * backend's API_PREFIX, or every call 404s while the page itself loads fine.
 */
const RAW_BASE = (import.meta.env.VITE_API_BASE ?? '').trim()

// Normalise away a trailing slash so `${API_BASE}/detect` never doubles up.
export const API_BASE = RAW_BASE.replace(/\/+$/, '')

/** Absolute URL for an API path. */
export function apiUrl(path: string): string {
  return `${API_BASE}${path.startsWith('/') ? path : `/${path}`}`
}

/**
 * Resolve the server-supplied annotated image for display.
 *
 * Three shapes arrive here, and all three must be usable as an <img> src:
 *  - "data:image/jpeg;base64,..." when the API inlines images (serverless)
 *  - "/api/media/abc.jpg" — a rooted path that already carries the API prefix
 *  - an absolute URL, if the API is ever hosted on another origin
 */
export function mediaUrl(path: string): string {
  if (!path) return ''
  if (/^(data:|https?:\/\/)/i.test(path)) return path
  // A rooted path is directly usable when the API shares this origin. Only a
  // cross-origin API base needs its host prefixed on.
  if (/^https?:\/\//i.test(API_BASE)) return `${API_BASE}${path}`
  return path
}

/**
 * Thin API client.
 *
 * The backend returns a uniform error body ({error, detail, allowed?}) for every
 * non-2xx, so failures are surfaced as a single exception type carrying the
 * machine-readable code alongside prose fit for display.
 */
export class ApiError extends Error {
  readonly code: string
  readonly status: number
  readonly allowed?: string[]

  constructor(status: number, body: Partial<ApiErrorBody>) {
    super(body.detail || `Request failed (HTTP ${status}).`)
    this.name = 'ApiError'
    this.status = status
    this.code = body.error || 'http_error'
    this.allowed = body.allowed
  }
}

async function toError(response: Response): Promise<ApiError> {
  let body: Partial<ApiErrorBody> = {}
  try {
    body = (await response.json()) as Partial<ApiErrorBody>
  } catch {
    // Non-JSON body; the status alone will have to do.
  }
  return new ApiError(response.status, body)
}

export async function fetchHealth(): Promise<HealthStatus> {
  const response = await fetch(apiUrl('/health'))
  if (!response.ok) throw await toError(response)
  return (await response.json()) as HealthStatus
}

export async function detect(
  file: Blob,
  filename = 'upload.jpg',
  endpoint: '/detect' | '/detect/frame' = '/detect',
): Promise<DetectionResult> {
  const form = new FormData()
  form.append('image', file, filename)

  const response = await fetch(apiUrl(endpoint), { method: 'POST', body: form })
  if (!response.ok) throw await toError(response)
  return (await response.json()) as DetectionResult
}
