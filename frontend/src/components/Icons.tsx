/** Inline SVG icons. Kept local so the page has no external asset requests. */

type Props = { className?: string }

const base = {
  width: 20,
  height: 20,
  viewBox: '0 0 24 24',
  fill: 'none',
  stroke: 'currentColor',
  strokeWidth: 1.8,
  strokeLinecap: 'round' as const,
  strokeLinejoin: 'round' as const,
}

export const UploadIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M12 16V4m0 0L7 9m5-5 5 5" />
    <path d="M4 17v1a3 3 0 0 0 3 3h10a3 3 0 0 0 3-3v-1" />
  </svg>
)

export const CameraIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M3 8.5A2.5 2.5 0 0 1 5.5 6h1.2a2 2 0 0 0 1.7-.95l.5-.8A2 2 0 0 1 10.6 3.3h2.8a2 2 0 0 1 1.7.95l.5.8A2 2 0 0 0 17.3 6h1.2A2.5 2.5 0 0 1 21 8.5v8A2.5 2.5 0 0 1 18.5 19h-13A2.5 2.5 0 0 1 3 16.5z" />
    <circle cx="12" cy="12.4" r="3.4" />
  </svg>
)

export const ScanIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M3 8V5.5A2.5 2.5 0 0 1 5.5 3H8M16 3h2.5A2.5 2.5 0 0 1 21 5.5V8M21 16v2.5A2.5 2.5 0 0 1 18.5 21H16M8 21H5.5A2.5 2.5 0 0 1 3 18.5V16" />
    <path d="M3 12h18" />
  </svg>
)

export const ShieldIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M12 3l7 3v5.5c0 4.3-2.9 8.1-7 9.5-4.1-1.4-7-5.2-7-9.5V6z" />
    <path d="M9.2 12.2l2 2 3.6-3.9" />
  </svg>
)

export const BoltIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M13 3L5 13.5h5L10.5 21 19 10.2h-5.2z" />
  </svg>
)

export const GridIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <rect x="3.5" y="3.5" width="7" height="7" rx="1.6" />
    <rect x="13.5" y="3.5" width="7" height="7" rx="1.6" />
    <rect x="3.5" y="13.5" width="7" height="7" rx="1.6" />
    <rect x="13.5" y="13.5" width="7" height="7" rx="1.6" />
  </svg>
)

export const CodeIcon = ({ className }: Props) => (
  <svg {...base} className={className} aria-hidden="true">
    <path d="M9 7l-5 5 5 5M15 7l5 5-5 5" />
  </svg>
)

export const CheckIcon = ({ className }: Props) => (
  <svg {...base} width={16} height={16} className={className} aria-hidden="true">
    <circle cx="12" cy="12" r="9" />
    <path d="M8.2 12.3l2.5 2.5 5-5.2" />
  </svg>
)

export const AlertIcon = ({ className }: Props) => (
  <svg {...base} width={16} height={16} className={className} aria-hidden="true">
    <path d="M12 4.5l8.5 15H3.5z" />
    <path d="M12 10v4M12 16.6v.4" />
  </svg>
)
