/// <reference types="vite/client" />

interface ImportMetaEnv {
  /**
   * Base URL of the PlateVision API, e.g. https://platevision.example.com
   * Leave unset for a same-origin deploy where FastAPI serves this bundle.
   */
  readonly VITE_API_BASE?: string
}

interface ImportMeta {
  readonly env: ImportMetaEnv
}
