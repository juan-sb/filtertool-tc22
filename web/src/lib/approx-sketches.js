// Mini |H| sketches of every approximation, drawn from real engine designs:
// a normalized 5th-order low-pass (fp = 1 Hz, fa = 2 Hz, 3 dB ripple, 25 dB
// stop), so each curve shows its approximation's character (flatness,
// passband / stopband ripple, roll-off). Computed once per session.

import { buildParams, DEFAULT_FORM, LP } from './params.js'

export const SKETCH_W = 40, SKETCH_H = 20

const SPEC = { ...DEFAULT_FORM, filterType: LP, nMin: 5, nMax: 5, fp: 1, fa: 2, apDb: 3, aaDb: 25, gainDb: 0, denorm: 0 }
const F_MIN = 0.1, F_MAX = 10, POINTS = 90
const DB_TOP = 1, DB_BOTTOM = -40

let cache = null

function toPath(bode) {
  const lx0 = Math.log10(F_MIN), lx1 = Math.log10(F_MAX)
  const pts = bode.freq.map((f, i) => {
    const m = bode.magnitude[i]
    const db = m > 0 ? 20 * Math.log10(m) : DB_BOTTOM
    const x = ((Math.log10(f) - lx0) / (lx1 - lx0)) * SKETCH_W
    const t = (DB_TOP - Math.min(DB_TOP, Math.max(DB_BOTTOM, db))) / (DB_TOP - DB_BOTTOM)
    return `${x.toFixed(2)},${(1 + t * (SKETCH_H - 2)).toFixed(2)}`
  })
  return `M${pts.join('L')}`
}

/**
 * @param {import('./engine-api').EngineApi} api
 * @returns {Promise<(string|null)[]>} SVG path per approximation index (null if a design failed)
 */
export function loadSketches(api) {
  cache ??= Promise.all(
    Array.from({ length: 7 }, async (_, i) => {
      try {
        const r = await api.filterDesign(buildParams({ ...SPEC, approxType: i }, 2 * Math.PI))
        if (r.error) return null
        return toPath(await api.computeBode(r.num, r.den, F_MIN, F_MAX, POINTS))
      } catch {
        return null
      }
    }),
  ).catch(() => { cache = null; return Array(7).fill(null) })
  return cache
}
