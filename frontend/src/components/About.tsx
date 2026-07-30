import { apiUrl } from '../api'
import { BoltIcon, CodeIcon, GridIcon, ScanIcon, ShieldIcon } from './Icons'

/**
 * Explanatory sections. The content here is deliberately specific -- real
 * measured figures and the actual reason the format resolver exists -- rather
 * than generic marketing copy.
 */

const STEPS = [
  {
    title: 'Detect',
    body: 'A YOLO model fine-tuned on Indian vehicles locates plate regions and scores each one.',
    tag: 'ultralytics YOLO',
  },
  {
    title: 'Crop & enhance',
    body: 'Each region is cropped with padding, then contrast-equalised. Harder crops get progressively upscaled.',
    tag: 'OpenCV CLAHE',
  },
  {
    title: 'Read',
    body: 'PaddleOCR reads the crop. Clean plates resolve on the first, cheapest pass.',
    tag: 'PP-OCRv5 mobile',
  },
  {
    title: 'Resolve',
    body: 'Raw text is matched against real Indian plate formats, repairing O/0 and I/1 confusions.',
    tag: 'format validation',
  },
]

const CARDS = [
  {
    icon: <ShieldIcon />,
    title: 'It tells you when it is unsure',
    body: (
      <>
        <p>
          Every reading carries a <span className="kv">valid</span> flag. Verified
          means the text matched a genuine Indian plate format. Unverified means
          OCR produced something unrecognisable &mdash; shown as-is, never dressed
          up as a confirmed plate.
        </p>
        <p>
          Characters are never invented to force a match: repair is capped at 30%
          of the string.
        </p>
      </>
    ),
  },
  {
    icon: <ScanIcon />,
    title: 'Why confidence alone fails',
    body: (
      <>
        <p>
          On one test image OCR reads a dealer watermark <span className="kv">alam</span> at
          97.6% and the real plate at 43.4%. Picking the most confident string
          returns the watermark.
        </p>
        <p>
          Matching against plate structure instead is what gets the right answer.
        </p>
      </>
    ),
  },
  {
    icon: <GridIcon />,
    title: 'Format aware',
    body: (
      <>
        <p>
          Understands the 1988 series &mdash;{' '}
          <span className="kv">MH12AB1234</span> &mdash; and the newer Bharat
          series, <span className="kv">22BH6517A</span>.
        </p>
        <p>
          All 41 state and union-territory codes are checked, which is often what
          resolves an ambiguous first character.
        </p>
      </>
    ),
  },
  {
    icon: <BoltIcon />,
    title: 'Cheap work first',
    body: (
      <>
        <p>
          Recognition escalates through preprocessing variants only when needed.
          A clean plate costs one pass at native resolution; upscaling is
          reserved for crops that actually fail.
        </p>
        <p>
          Measured at <span className="kv">~358 ms</span> per plate for OCR on CPU.
        </p>
      </>
    ),
  },
  {
    icon: <CameraLike />,
    title: 'Upload or live camera',
    body: (
      <>
        <p>
          Drop a photo, or capture straight from your device camera. Capture
          happens in the browser and frames are posted to the API like any other
          image.
        </p>
        <p>Boxes below are drawn from the coordinates the API returns.</p>
      </>
    ),
  },
  {
    icon: <CodeIcon />,
    title: 'A real API, documented',
    body: (
      <>
        <p>
          Everything here runs on the same public endpoints you can call
          yourself. Schemas are generated from the server&rsquo;s own models, so
          the docs cannot drift from behaviour.
        </p>
        <p>
          Browse them at <a href={apiUrl('/docs')}>/docs</a>.
        </p>
      </>
    ),
  },
]

function CameraLike() {
  return (
    <svg
      width={20}
      height={20}
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth={1.8}
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      <rect x="3" y="6.5" width="18" height="12" rx="2.4" />
      <path d="M7.5 11h3M13.5 11h3M7.5 14.2h9" />
    </svg>
  )
}

const STACK = [
  'FastAPI',
  'Pydantic v2',
  'React',
  'TypeScript',
  'Vite',
  'Ultralytics YOLO',
  'PaddleOCR',
  'EasyOCR',
  'OpenCV',
  'pytest',
]

export default function About() {
  return (
    <>
      <section id="how">
        <div className="section-head">
          <span className="eyebrow">Pipeline</span>
          <h2>Four stages, from photo to plate</h2>
          <p>
            Each stage is independently testable, and the recognition logic knows
            nothing about the OCR engine behind it &mdash; which is how the engine
            can be swapped with one setting.
          </p>
        </div>
        <div className="steps">
          {STEPS.map((step) => (
            <div className="step" key={step.title}>
              <h4>{step.title}</h4>
              <p>{step.body}</p>
              <span className="tag">{step.tag}</span>
            </div>
          ))}
        </div>
      </section>

      <section id="about">
        <div className="section-head">
          <span className="eyebrow">What makes it different</span>
          <h2>Built to be honest about uncertainty</h2>
          <p>
            Most of the engineering here is not detection &mdash; that part works
            well. It is deciding whether a reading deserves to be trusted.
          </p>
        </div>
        <div className="cards">
          {CARDS.map((card) => (
            <article className="card" key={card.title}>
              <div className="ico">{card.icon}</div>
              <h3>{card.title}</h3>
              {card.body}
            </article>
          ))}
        </div>

        <div className="stack">
          {STACK.map((item) => (
            <span key={item}>{item}</span>
          ))}
        </div>
      </section>
    </>
  )
}
