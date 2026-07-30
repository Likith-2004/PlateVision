import { useRef, useState } from 'react'
import { UploadIcon } from './Icons'

interface Props {
  file: File | null
  busy: boolean
  disabled: boolean
  onPick: (file: File | null) => void
  onRun: () => void
}

const ACCEPT = '.png,.jpg,.jpeg,.webp,.bmp'

export default function UploadPanel({ file, busy, disabled, onPick, onRun }: Props) {
  const input = useRef<HTMLInputElement>(null)
  const [over, setOver] = useState(false)
  const [localError, setLocalError] = useState<string | null>(null)

  function accept(candidate: File | undefined) {
    if (!candidate) return
    // Cheap client-side guard; the server validates authoritatively.
    if (!candidate.type.startsWith('image/')) {
      setLocalError(`"${candidate.name}" is not an image file.`)
      onPick(null)
      return
    }
    setLocalError(null)
    onPick(candidate)
  }

  return (
    <>
      <div
        className={`drop${over ? ' over' : ''}`}
        role="button"
        tabIndex={0}
        aria-label="Choose an image, or drop one here"
        onClick={() => input.current?.click()}
        onKeyDown={(e) => {
          if (e.key === 'Enter' || e.key === ' ') {
            e.preventDefault()
            input.current?.click()
          }
        }}
        onDragEnter={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragOver={(e) => {
          e.preventDefault()
          setOver(true)
        }}
        onDragLeave={() => setOver(false)}
        onDrop={(e) => {
          e.preventDefault()
          setOver(false)
          accept(e.dataTransfer.files[0])
        }}
      >
        <div className="drop-icon">
          <UploadIcon />
        </div>
        <h3>Drop a vehicle photo</h3>
        <p>
          or click to browse &mdash; JPG, PNG, WEBP, BMP &middot; large photos are
          resized automatically
        </p>
      </div>

      <input
        ref={input}
        type="file"
        className="sr"
        accept={ACCEPT}
        onChange={(e) => accept(e.target.files?.[0])}
      />

      {file && (
        <div className="file-chip">
          <span>
            {file.name} &middot; {(file.size / 1024).toFixed(0)} KB
          </span>
          <button type="button" aria-label="Remove selected file" onClick={() => onPick(null)}>
            &times;
          </button>
        </div>
      )}

      <div className="actions">
        <button
          type="button"
          className="btn primary"
          disabled={!file || busy || disabled}
          onClick={onRun}
        >
          {busy ? (
            <>
              <span className="spin" style={{ display: 'inline-block', verticalAlign: '-2px', marginRight: 8 }} />
              Analysing&hellip;
            </>
          ) : (
            'Detect plates'
          )}
        </button>
      </div>

      {localError && (
        <div className="alert bad" role="alert">
          <span>{localError}</span>
        </div>
      )}
    </>
  )
}
