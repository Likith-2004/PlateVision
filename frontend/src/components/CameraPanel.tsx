import { useCallback, useEffect, useRef, useState } from 'react'
import { CameraIcon } from './Icons'

interface Props {
  busy: boolean
  disabled: boolean
  onCapture: (frame: File) => void
}

/**
 * Browser-side camera capture.
 *
 * The camera belongs in the client: the server has no camera once deployed
 * anywhere but a developer's own machine. Frames are drawn to a canvas and
 * POSTed like any other upload.
 */
export default function CameraPanel({ busy, disabled, onCapture }: Props) {
  const video = useRef<HTMLVideoElement>(null)
  const stream = useRef<MediaStream | null>(null)
  const [live, setLive] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const stop = useCallback(() => {
    stream.current?.getTracks().forEach((t) => t.stop())
    stream.current = null
    if (video.current) video.current.srcObject = null
    setLive(false)
  }, [])

  // Release the camera when the panel unmounts, so the indicator light does
  // not stay on after switching tabs.
  useEffect(() => stop, [stop])

  async function start() {
    setError(null)
    if (!navigator.mediaDevices?.getUserMedia) {
      setError('This browser does not expose a camera API.')
      return
    }
    try {
      const media = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 1280 } },
        audio: false,
      })
      stream.current = media
      if (video.current) {
        video.current.srcObject = media
        await video.current.play()
      }
      setLive(true)
    } catch (err) {
      const name = err instanceof DOMException ? err.name : ''
      setError(
        name === 'NotAllowedError'
          ? 'Camera permission was denied. Allow access, then try again.'
          : name === 'NotFoundError'
            ? 'No camera was found on this device.'
            : 'Could not open the camera.',
      )
    }
  }

  async function shoot() {
    const el = video.current
    if (!el || !el.videoWidth) {
      setError('The camera has not produced a frame yet.')
      return
    }
    const canvas = document.createElement('canvas')
    canvas.width = el.videoWidth
    canvas.height = el.videoHeight
    canvas.getContext('2d')?.drawImage(el, 0, 0)

    const blob = await new Promise<Blob | null>((resolve) =>
      canvas.toBlob(resolve, 'image/jpeg', 0.92),
    )
    if (!blob) {
      setError('Could not capture a frame.')
      return
    }
    onCapture(new File([blob], 'camera-frame.jpg', { type: 'image/jpeg' }))
  }

  return (
    <>
      <div className={`stage${busy ? ' scanning' : ''}`}>
        <video ref={video} playsInline muted />
        {!live && (
          <div className="stage-empty">
            <CameraIcon />
            <span>Camera is off</span>
          </div>
        )}
      </div>

      <div className="actions">
        {!live ? (
          <button type="button" className="btn primary" onClick={start} disabled={disabled}>
            Start camera
          </button>
        ) : (
          <>
            <button
              type="button"
              className="btn primary"
              onClick={shoot}
              disabled={busy || disabled}
            >
              {busy ? 'Analysing…' : 'Capture & detect'}
            </button>
            <button type="button" className="btn ghost" onClick={stop}>
              Stop
            </button>
          </>
        )}
      </div>

      {error && (
        <div className="alert bad" role="alert">
          <span>{error}</span>
        </div>
      )}
    </>
  )
}
