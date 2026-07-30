/**
 * Downscale a picked image in the browser before uploading it.
 *
 * Why this exists: a Vercel Function caps the request body at 4.5 MB, and a
 * photo straight off a phone is routinely 4–12 MB. Uploading the original would
 * be rejected at the edge with an opaque error before the app ever sees it, so
 * the server's own friendly 413 would never get the chance to fire.
 *
 * It costs nothing in accuracy. The detector already downscales anything longer
 * than MAX_INFERENCE_EDGE (1280 px) before inference, so pixels beyond that
 * budget are discarded server-side regardless. Shrinking here just moves that
 * work off the wire, which also makes the upload noticeably faster on mobile.
 */

/** Longest edge kept. Comfortably above the detector's own 1280 px ceiling. */
const MAX_EDGE = 1600

/** Byte target, held well under Vercel's 4.5 MB hard cap. */
const TARGET_BYTES = 3_500_000

/** Quality ladder, tried in order until the result fits. */
const QUALITIES = [0.85, 0.7, 0.55]

export interface Prepared {
  blob: Blob
  filename: string
  /** True when the file was re-encoded rather than passed through untouched. */
  resized: boolean
  originalBytes: number
}

/**
 * Decode a file honouring EXIF orientation.
 *
 * Orientation matters: phone cameras commonly store a landscape sensor image
 * plus a "rotate 90°" tag. Drawing that to a canvas without applying the tag
 * feeds a sideways photo to the detector, which then finds no plate at all.
 */
async function decode(file: Blob): Promise<ImageBitmap | HTMLImageElement> {
  if (typeof createImageBitmap === 'function') {
    try {
      return await createImageBitmap(file, { imageOrientation: 'from-image' })
    } catch {
      // Safari has historically rejected the options bag, and some browsers
      // cannot decode BMP this way. Fall through to the <img> path.
    }
  }

  const url = URL.createObjectURL(file)
  try {
    return await new Promise<HTMLImageElement>((resolve, reject) => {
      const img = new Image()
      img.onload = () => resolve(img)
      img.onerror = () => reject(new Error('The browser could not decode that image.'))
      img.src = url
    })
  } finally {
    // Safe once decoding has settled; the bitmap is already in memory.
    URL.revokeObjectURL(url)
  }
}

function dimensions(source: ImageBitmap | HTMLImageElement): [number, number] {
  return source instanceof HTMLImageElement
    ? [source.naturalWidth, source.naturalHeight]
    : [source.width, source.height]
}

function toBlob(canvas: HTMLCanvasElement, quality: number): Promise<Blob> {
  return new Promise((resolve, reject) => {
    canvas.toBlob(
      (blob) =>
        blob ? resolve(blob) : reject(new Error('The browser could not encode that image.')),
      'image/jpeg',
      quality,
    )
  })
}

/** Swap any extension for .jpg, since the re-encoded bytes are always JPEG. */
function asJpegName(name: string): string {
  const base = name.replace(/\.[^./\\]+$/, '') || 'upload'
  return `${base}.jpg`
}

/**
 * Return upload-ready bytes for `file`.
 *
 * Small, correctly-sized images are passed through untouched so they are never
 * needlessly recompressed. Anything else is scaled to MAX_EDGE and encoded as
 * JPEG, stepping down the quality ladder until it fits TARGET_BYTES.
 */
export async function prepareUpload(file: File): Promise<Prepared> {
  const originalBytes = file.size
  const source = await decode(file)
  const [width, height] = dimensions(source)

  if (!width || !height) {
    throw new Error('That image appears to be empty or corrupt.')
  }

  const longest = Math.max(width, height)
  if (longest <= MAX_EDGE && originalBytes <= TARGET_BYTES) {
    if (source instanceof ImageBitmap) source.close()
    return { blob: file, filename: file.name, resized: false, originalBytes }
  }

  let scale = Math.min(1, MAX_EDGE / longest)
  let best: Blob | null = null

  // Try the quality ladder at this scale; if even the lowest quality is too
  // large, shrink further. Two rounds is ample: 1600 px at q0.55 is already
  // far below the target for any realistic photo.
  for (let round = 0; round < 3; round += 1) {
    const canvas = document.createElement('canvas')
    canvas.width = Math.max(1, Math.round(width * scale))
    canvas.height = Math.max(1, Math.round(height * scale))

    const ctx = canvas.getContext('2d')
    if (!ctx) throw new Error('This browser does not support image resizing.')
    ctx.imageSmoothingQuality = 'high'
    ctx.drawImage(source, 0, 0, canvas.width, canvas.height)

    for (const quality of QUALITIES) {
      const blob = await toBlob(canvas, quality)
      best = blob
      if (blob.size <= TARGET_BYTES) {
        if (source instanceof ImageBitmap) source.close()
        return {
          blob,
          filename: asJpegName(file.name),
          resized: true,
          originalBytes,
        }
      }
    }
    scale *= 0.75
  }

  if (source instanceof ImageBitmap) source.close()
  if (!best) throw new Error('Could not prepare that image for upload.')

  // Still over target after shrinking twice. Send it anyway and let the server
  // answer: a real 413 with a clear message beats silently refusing here.
  return { blob: best, filename: asJpegName(file.name), resized: true, originalBytes }
}
