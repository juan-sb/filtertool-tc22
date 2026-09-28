// Mini |H| sketches of every approximation, drawn from real engine designs:
// a normalized 4th-order low-pass (fp = 1, fa = 2 rad/s, Gp = 0.7, Ga = 0.3),
// plotted as linear amplitude over a linear ω axis (textbook style), so
// passband ripple, stopband ripple and transmission zeros are all visible.
// (With specs this loose Cauer is very selective: its first zero sits at
// ω ≈ 1.007, right after the last ripple peak, the second at ω ≈ 1.32.)
// Computed once per session.

import { buildParams, DEFAULT_FORM, LP } from './params.js'

export const SKETCH_W = 40, SKETCH_H = 20

const GP = 0.7, GA = 0.3
const SPEC = {
  ...DEFAULT_FORM, filterType: LP, nMin: 4, nMax: 4, fp: 1, fa: 2, gainDb: 0, denorm: 0,
  apDb: -20 * Math.log10(GP), aaDb: -20 * Math.log10(GA),
}
const W_MAX = 2.2, POINTS = 300, A_MAX = 1.05

let cache = null

/** |P(jω)| for descending-power real coefficients. */
function polyAbs(c, w) {
  let re = 0, im = 0
  for (const a of c) [re, im] = [a - im * w, re * w]   // Horner: P ← P·jω + a
  return Math.hypot(re, im)
}

function toPath({ num, den, zeros }) {
  // Dense grid plus the exact transmission-zero frequencies, so nulls reach 0.
  const ws = Array.from({ length: POINTS + 1 }, (_, i) => (i / POINTS) * W_MAX)
  for (const [re, im] of zeros ?? []) if (Math.abs(re) < 1e-9 && im > 0 && im < W_MAX) ws.push(im)
  ws.sort((a, b) => a - b)
  const pts = ws.map(w => {
    const m = Math.min(A_MAX, polyAbs(num, w) / polyAbs(den, w))
    const x = (w / W_MAX) * SKETCH_W
    const y = 1 + (1 - m / A_MAX) * (SKETCH_H - 2)
    return `${x.toFixed(2)},${y.toFixed(2)}`
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
        // toRad = 1: the sketch spec is already in rad/s
        const r = await api.filterDesign(buildParams({ ...SPEC, approxType: i }, 1))
        return r.error ? null : toPath(r)
      } catch {
        return null
      }
    }),
  ).catch(() => { cache = null; return Array(7).fill(null) })
  return cache
}

/** y of amplitude a in sketch coordinates (for guide lines). */
export const sketchY = a => 1 + (1 - a / A_MAX) * (SKETCH_H - 2)
export const SKETCH_GP = GP, SKETCH_GA = GA
