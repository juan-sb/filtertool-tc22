// Mini |H| sketches of every approximation, drawn from real engine designs of
// a normalized low-pass (fp = 1 rad/s), plotted as linear amplitude over a
// linear ω axis (textbook style). Each sketch also carries a Butterworth
// "ghost" of the same spec, so e.g. Legendre's steeper monotonic edge or
// Bessel's gentle roll-off read by contrast. Computed once per session.
//
// Spec: Gp = 0.6, Ga = 0.4 for all; N = 5 except Cauer. At a fixed order an
// elliptic design spends loose specs on selectivity: at N ≥ 3 its zeros land at
// ω ≈ 1.00–1.02, on top of the passband edge. N = 2 puts the zero at ω ≈ 1.17,
// and (even order) the stopband climbs back to Ga, a full-height bounce.

import { buildParams, DEFAULT_FORM, LP } from './params.js'

export const SKETCH_W = 40, SKETCH_H = 20

const BASE = { n: 5, gp: 0.6, ga: 0.4 }
const SPECS = [BASE, BASE, BASE, { ...BASE, n: 2 }, BASE, BASE, BASE]
const W_MAX = 2.2, POINTS = 300, A_MAX = 1.05

let cache = null

/** |P(jω)| for descending-power real coefficients. */
function polyAbs(c, w) {
  let re = 0, im = 0
  for (const a of c) [re, im] = [a - im * w, re * w]   // Horner: P ← P·jω + a
  return Math.hypot(re, im)
}

/** y of amplitude a in sketch coordinates. */
export const sketchY = a => 1 + (1 - Math.min(A_MAX, a) / A_MAX) * (SKETCH_H - 2)

function toPath({ num, den, zeros }) {
  // Dense grid plus the exact transmission-zero frequencies, so nulls reach 0.
  const ws = Array.from({ length: POINTS + 1 }, (_, i) => (i / POINTS) * W_MAX)
  for (const [re, im] of zeros ?? []) if (Math.abs(re) < 1e-9 && im > 0 && im < W_MAX) ws.push(im)
  ws.sort((a, b) => a - b)
  const pts = ws.map(w => `${((w / W_MAX) * SKETCH_W).toFixed(2)},${sketchY(polyAbs(num, w) / polyAbs(den, w)).toFixed(2)}`)
  return `M${pts.join('L')}`
}

async function design(api, approxType, { n, gp, ga }) {
  const form = {
    ...DEFAULT_FORM, filterType: LP, approxType, nMin: n, nMax: n, fp: 1, fa: 2, gainDb: 0, denorm: 0,
    apDb: -20 * Math.log10(gp), aaDb: -20 * Math.log10(ga),
  }
  // toRad = 1: the sketch spec is already in rad/s
  const r = await api.filterDesign(buildParams(form, 1))
  return r.error ? null : toPath(r)
}

/**
 * @param {import('./engine-api').EngineApi} api
 * @returns {Promise<({ path: string|null, ghost: string|null, gp: number, ga: number })[]>}
 */
export function loadSketches(api) {
  cache ??= Promise.all(
    SPECS.map(async (spec, i) => {
      try {
        const [path, ghost] = await Promise.all([design(api, i, spec), i === 0 ? null : design(api, 0, spec)])
        return { path, ghost, gp: spec.gp, ga: spec.ga }
      } catch {
        return { path: null, ghost: null, gp: spec.gp, ga: spec.ga }
      }
    }),
  ).catch(() => { cache = null; return SPECS.map(s => ({ path: null, ghost: null, gp: s.gp, ga: s.ga })) })
  return cache
}
