import { useState } from 'react'
import { mediaUrl } from '../api'
import type { DetectionResult } from '../types'
import PlateCard from './PlateCard'

interface Props {
  result: DetectionResult
  /** Object URL of the image the user supplied, for the overlay view. */
  previewUrl: string | null
}

type View = 'overlay' | 'server'

export default function Results({ result, previewUrl }: Props) {
  const [view, setView] = useState<View>('overlay')
  const [active, setActive] = useState<number | null>(null)

  const { plates, source_width: sw, source_height: sh } = result
  const verified = plates.filter((p) => p.valid).length

  // Box coordinates are pixels in the inference frame, which may have been
  // downscaled. Normalising against source dimensions makes them independent
  // of whatever size the browser renders the image at.
  const canOverlay = previewUrl !== null && sw > 0 && sh > 0
  const showOverlay = view === 'overlay' && canOverlay
  // Server paths are relative to the API, which may be a different host.
  const serverImage = mediaUrl(result.image_url)
  const imageSrc = showOverlay ? previewUrl : serverImage

  return (
    <div className="result-grid">
      <div>
        {canOverlay && serverImage && (
          <div className="view-toggle" role="group" aria-label="Image view">
            <button
              type="button"
              aria-pressed={view === 'overlay'}
              onClick={() => setView('overlay')}
            >
              Interactive
            </button>
            <button
              type="button"
              aria-pressed={view === 'server'}
              onClick={() => setView('server')}
            >
              Server render
            </button>
          </div>
        )}

        <div className="canvas-wrap">
          {imageSrc && (
            <img
              src={imageSrc}
              alt={
                showOverlay
                  ? 'Submitted image with detected plates outlined'
                  : 'Server-annotated detection result'
              }
            />
          )}

          {showOverlay &&
            plates.map((plate, i) => {
              const { x1, y1, x2, y2 } = plate.box
              return (
                <div
                  key={`${plate.number}-${i}`}
                  className={[
                    'bbox',
                    plate.valid ? '' : 'unverified',
                    active === i ? 'active' : '',
                  ]
                    .filter(Boolean)
                    .join(' ')}
                  style={{
                    left: `${(x1 / sw) * 100}%`,
                    top: `${(y1 / sh) * 100}%`,
                    width: `${((x2 - x1) / sw) * 100}%`,
                    height: `${((y2 - y1) / sh) * 100}%`,
                    animationDelay: `${i * 90}ms`,
                  }}
                  onMouseEnter={() => setActive(i)}
                  onMouseLeave={() => setActive(null)}
                >
                  <span>{plate.number}</span>
                </div>
              )
            })}
        </div>

        <p className="meta-line">
          <span>
            {sw}&times;{sh}
          </span>
          <span>{result.elapsed_ms.toFixed(0)} ms</span>
          <span>
            {plates.length} detected · {verified} verified
          </span>
        </p>
      </div>

      <div>
        {plates.length === 0 ? (
          <p className="empty-note">
            No number plate was found in this image. Try a closer or better-lit
            shot where the plate is clearly visible.
          </p>
        ) : (
          plates.map((plate, i) => (
            <PlateCard
              key={`${plate.number}-${i}`}
              plate={plate}
              index={i}
              active={active === i}
              onHover={setActive}
            />
          ))
        )}
      </div>
    </div>
  )
}
