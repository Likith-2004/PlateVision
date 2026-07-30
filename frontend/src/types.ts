/**
 * Mirrors app/schemas.py. If the API contract changes, change it here too --
 * the source of truth is the Pydantic model, visible at /docs.
 */

export interface BoundingBox {
  x1: number
  y1: number
  x2: number
  y2: number
}

export type PlateFormat = 'standard' | 'bh_series'

export interface Plate {
  number: string
  confidence: number
  detection_confidence: number
  /** False means OCR produced text matching no known Indian plate format. */
  valid: boolean
  format: PlateFormat | null
  state: string | null
  corrections: number
  variant: string | null
  passes: number
  votes: number
  box: BoundingBox
}

export interface DetectionResult {
  plates: Plate[]
  image_url: string
  /** Dimensions of the frame the boxes are relative to (may be downscaled). */
  source_width: number
  source_height: number
  elapsed_ms: number
}

export interface HealthStatus {
  status: 'ok' | 'loading' | 'error'
  ready: boolean
  model_loaded: boolean
  ocr_loaded: boolean
  ocr_engine: string
  version: string
  detail: string | null
}

export interface ApiErrorBody {
  error: string
  detail: string
  allowed?: string[]
}
