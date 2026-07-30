import type { Plate } from '../types'
import { AlertIcon, CheckIcon } from './Icons'

interface Props {
  plate: Plate
  index: number
  active: boolean
  onHover: (index: number | null) => void
}

function Gauge({ label, value, warn }: { label: string; value: number; warn?: boolean }) {
  const pct = Math.max(0, Math.min(100, value))
  return (
    <div className="gauge-row">
      <label>{label}</label>
      <div
        className="gauge"
        role="meter"
        aria-label={label}
        aria-valuenow={Math.round(pct)}
        aria-valuemin={0}
        aria-valuemax={100}
      >
        <i className={warn ? 'warn' : undefined} style={{ width: `${pct}%` }} />
      </div>
      <b>{pct.toFixed(0)}%</b>
    </div>
  )
}

const FORMAT_LABEL: Record<string, string> = {
  standard: '1988 series',
  bh_series: 'Bharat series',
}

export default function PlateCard({ plate, index, active, onHover }: Props) {
  const verified = plate.valid

  return (
    <article
      className={`plate-card${active ? ' active' : ''}`}
      style={{ animationDelay: `${index * 70}ms` }}
      onMouseEnter={() => onHover(index)}
      onMouseLeave={() => onHover(null)}
    >
      <div className={`plate-visual${verified ? '' : ' is-unverified'}`}>
        {plate.number}
      </div>

      <p className={`verdict ${verified ? 'good' : 'iffy'}`}>
        {verified ? <CheckIcon /> : <AlertIcon />}
        {verified
          ? `Verified · ${FORMAT_LABEL[plate.format ?? ''] ?? plate.format}`
          : 'Unverified reading'}
      </p>

      <div className="gauges">
        <Gauge label="OCR" value={plate.confidence} warn={!verified} />
        <Gauge label="Detection" value={plate.detection_confidence} />
      </div>

      <div className="chips">
        {plate.state && <span className="chip hi">state {plate.state}</span>}
        {plate.variant && <span className="chip">{plate.variant}</span>}
        <span className="chip">
          {plate.passes} pass{plate.passes === 1 ? '' : 'es'}
        </span>
        {plate.votes > 1 && <span className="chip hi">{plate.votes} agreed</span>}
        {plate.corrections > 0 && (
          <span className="chip">
            {plate.corrections} char repaired
          </span>
        )}
      </div>

      {!verified && (
        <p className="caveat">
          This text matched no known Indian plate format, so it is shown exactly
          as OCR returned it. Characters are never invented to force a match.
        </p>
      )}
    </article>
  )
}
