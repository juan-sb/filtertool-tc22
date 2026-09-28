// SI-prefix number formatting ("2.2k", "4.7n"), shared by SciInput and plot labels.

// Ordered largest→smallest so we pick the best prefix for display
export const SI = [
  { p: 'T', f: 1e12 },
  { p: 'G', f: 1e9  },
  { p: 'M', f: 1e6  },
  { p: 'k', f: 1e3  },
  { p: '',  f: 1    },
  { p: 'm', f: 1e-3 },
  { p: 'µ', f: 1e-6 },
  { p: 'n', f: 1e-9 },
  { p: 'p', f: 1e-12 },
  { p: 'f', f: 1e-15 },
]

/** 4 significant figures with the best SI prefix, trailing zeros stripped. */
export function formatSI(num) {
  if (!isFinite(num)) return '—'
  const abs = Math.abs(num)
  // Exact / near-zero must not pick a tiny SI prefix (0 would become "0f")
  if (abs < 1e-15) return '0'
  // find the best prefix: largest factor where abs >= factor (with small tolerance)
  const entry = SI.find(s => abs >= s.f * 0.9995) ?? SI[SI.length - 1]
  const scaled = num / entry.f
  const str = parseFloat(scaled.toPrecision(4)).toString()
  return `${str}${entry.p}`
}
