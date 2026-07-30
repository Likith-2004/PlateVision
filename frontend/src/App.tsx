import { useCallback, useEffect, useRef, useState } from 'react'
import { ApiError, apiUrl, detect, fetchHealth } from './api'
import { prepareUpload } from './image'
import type { DetectionResult, HealthStatus } from './types'
import About from './components/About'
import CameraPanel from './components/CameraPanel'
import Results from './components/Results'
import UploadPanel from './components/UploadPanel'
import { CameraIcon, UploadIcon } from './components/Icons'

type Mode = 'upload' | 'camera'

export default function App() {
  const [health, setHealth] = useState<HealthStatus | null>(null)
  const [unreachable, setUnreachable] = useState(false)
  const [mode, setMode] = useState<Mode>('upload')
  const [file, setFile] = useState<File | null>(null)
  const [previewUrl, setPreviewUrl] = useState<string | null>(null)
  const [result, setResult] = useState<DetectionResult | null>(null)
  const [error, setError] = useState<ApiError | Error | null>(null)
  const [busy, setBusy] = useState(false)

  const resultsRef = useRef<HTMLDivElement>(null)
  // Tracked so the object URL can be revoked only once nothing renders it.
  const previewRef = useRef<string | null>(null)

  /* ------------------------------------------------------------- health ---- */

  useEffect(() => {
    let cancelled = false
    let timer: number | undefined

    const poll = async () => {
      try {
        const status = await fetchHealth()
        if (cancelled) return
        setHealth(status)
        setUnreachable(false)
        // Models load in a background task, so keep polling until ready.
        if (!status.ready) timer = window.setTimeout(poll, 1500)
      } catch {
        if (cancelled) return
        setUnreachable(true)
        timer = window.setTimeout(poll, 3000)
      }
    }

    void poll()
    return () => {
      cancelled = true
      if (timer) window.clearTimeout(timer)
    }
  }, [])

  /* ------------------------------------------------------------ preview ---- */

  const setPreview = useCallback((next: File | null) => {
    if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    const url = next ? URL.createObjectURL(next) : null
    previewRef.current = url
    setPreviewUrl(url)
  }, [])

  useEffect(
    () => () => {
      if (previewRef.current) URL.revokeObjectURL(previewRef.current)
    },
    [],
  )

  /* ------------------------------------------------------------- actions --- */

  const run = useCallback(
    async (target: File) => {
      setBusy(true)
      setError(null)
      setPreview(target)
      try {
        // Resize before upload: hosted functions cap the request body well
        // below a typical phone photo, and the detector discards the extra
        // pixels anyway. See image.ts.
        const { blob, filename } = await prepareUpload(target)
        const detection = await detect(blob, filename)
        setResult(detection)
        // Give the DOM a frame to paint before scrolling to it.
        requestAnimationFrame(() =>
          resultsRef.current?.scrollIntoView({ behavior: 'smooth', block: 'start' }),
        )
      } catch (err) {
        setResult(null)
        setError(err instanceof Error ? err : new Error('Detection failed.'))
      } finally {
        setBusy(false)
      }
    },
    [setPreview],
  )

  const pick = useCallback((next: File | null) => {
    setFile(next)
    setError(null)
    if (!next) {
      setResult(null)
      setPreview(null)
    }
  }, [setPreview])

  /* -------------------------------------------------------------- render --- */

  const ready = health?.ready === true
  const beacon = unreachable || health?.status === 'error' ? 'bad' : ready ? 'ok' : 'warn'
  const statusText = unreachable
    ? 'Server unreachable'
    : health?.status === 'error'
      ? 'Model failed to load'
      : ready
        ? 'Ready'
        : 'Loading models'

  return (
    <>
      <div className="aurora" aria-hidden="true">
        <span />
        <span />
        <span />
      </div>
      <div className="grid-veil" aria-hidden="true" />

      <div className="shell">
        <header className="hero">
          <span className="eyebrow">Automatic number plate recognition</span>
          <h1>Read any Indian plate</h1>
          <p className="lede">
            Detection with a fine-tuned YOLO model, recognition with PaddleOCR, and
            a resolver that checks the result against{' '}
            <strong>real Indian plate formats</strong> before it claims to have
            read anything.
          </p>

          <div className="hero-actions">
            <div className="status">
              <span className={`beacon ${beacon}`} />
              <span>
                {statusText}
                {ready && health && (
                  <>
                    {' · '}
                    <b>{health.ocr_engine}</b>
                    {' · v'}
                    {health.version}
                  </>
                )}
              </span>
            </div>
            <a className="btn ghost" href="#how">
              How it works
            </a>
            <a className="btn ghost" href={apiUrl('/docs')}>
              API docs
            </a>
          </div>

          <div className="stat-strip">
            <div>
              <b>~99%</b>
              <small>OCR confidence</small>
            </div>
            <div>
              <b>~358ms</b>
              <small>per plate</small>
            </div>
            <div>
              <b>41</b>
              <small>state codes</small>
            </div>
            <div>
              <b>2</b>
              <small>plate formats</small>
            </div>
          </div>
        </header>

        <section aria-label="Plate detection" className="studio">
          <div className="tabs" role="tablist">
            <button
              className="tab"
              role="tab"
              aria-selected={mode === 'upload'}
              onClick={() => setMode('upload')}
            >
              <UploadIcon /> Upload
            </button>
            <button
              className="tab"
              role="tab"
              aria-selected={mode === 'camera'}
              onClick={() => setMode('camera')}
            >
              <CameraIcon /> Camera
            </button>
          </div>

          <div className="studio-body">
            {mode === 'upload' ? (
              <UploadPanel
                file={file}
                busy={busy}
                disabled={!ready}
                onPick={pick}
                onRun={() => file && void run(file)}
              />
            ) : (
              <CameraPanel
                busy={busy}
                disabled={!ready}
                onCapture={(frame) => {
                  setFile(frame)
                  void run(frame)
                }}
              />
            )}

            {!ready && !unreachable && health?.status !== 'error' && (
              <div className="alert warn" role="status">
                <span className="spin" />
                <span>
                  The detector and OCR models are still loading. This takes a few
                  seconds on first start.
                </span>
              </div>
            )}

            {health?.status === 'error' && (
              <div className="alert bad" role="alert">
                <span>
                  The server could not load its models.
                  {health.detail ? <> <code>{health.detail}</code></> : null}
                </span>
              </div>
            )}

            {unreachable && (
              <div className="alert bad" role="alert">
                <span>
                  Cannot reach the API. Is the server running on this host?
                </span>
              </div>
            )}

            {error && (
              <div className="alert bad" role="alert">
                <span>
                  {error.message}
                  {error instanceof ApiError && error.allowed && (
                    <>
                      {' '}
                      Allowed types: <code>{error.allowed.join(', ')}</code>
                    </>
                  )}
                </span>
              </div>
            )}
          </div>
        </section>

        <div ref={resultsRef}>
          {result && (
            <section aria-label="Detection results">
              <div className="section-head">
                <span className="eyebrow">Result</span>
                <h2>
                  {result.plates.length === 0
                    ? 'No plate found'
                    : result.plates.some((p) => p.valid)
                      ? 'Plate recognised'
                      : 'Read, but not verified'}
                </h2>
              </div>
              <Results result={result} previewUrl={previewUrl} />
            </section>
          )}
        </div>

        <About />

        <footer>
          <span>
            PlateVision {health?.version ? `v${health.version}` : ''} &middot;
            detection and recognition run entirely on this server.
          </span>
          <span>
            <a href={apiUrl('/docs')}>API</a> &middot;{' '}
            <a href={apiUrl('/health')}>health</a>
          </span>
        </footer>
      </div>
    </>
  )
}
