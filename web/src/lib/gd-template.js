// Group-delay template (E7). The engine picks the order so the normalized delay
// satisfies 1 − τn(ω_rg·τ0) ≤ γ/100, i.e. the delay must stay at or above
// τ_min = τ0·(1 − γ/100) from DC up to f_rg (it is τ0 at DC and falls with f).
//
// Units: frequencies in Hz, delays in seconds; form frequencies in the data
// unit with `uf` = data units per Hz.

const EPS = 1.002
const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

/** { tau0, tauMin, frgHz, gamma } from the design form (GD type). */
export function gdGeom(form, uf) {
  const tau0 = form.tau0, gamma = form.gamma
  return { tau0, gamma, tauMin: tau0 * (1 - gamma / 100), frgHz: form.frg / uf }
}

/**
 * Handles (x Hz, y s; span = y-range of 'v' or x-range of 'h', 0 / Infinity = open):
 * floor edge (γ), f_rg edge, their corner, and the τ0 line.
 */
export function gdHandles(g) {
  if (!g) return []
  return [
    { id: 'h:gamma', kind: 'h', group: 'pass', y: g.tauMin, span: [0, g.frgHz], yf: 'gamma' },
    { id: 'v:frg',   kind: 'v', group: 'pass', x: g.frgHz, span: [-Infinity, g.tauMin], xf: 'frg' },
    { id: 'c:frg',   kind: 'c', group: 'pass', x: g.frgHz, y: g.tauMin, xf: 'frg', yf: 'gamma' },
    { id: 't:tau0',  kind: 't', group: 'centre', y: g.tau0, span: [0, Infinity], yf: 'tau0' },
  ]
}

/** New form after dragging handle `h` to (xHz, yS), from the form at drag start. */
export function gdDragTo(form, h, xHz, yS, uf) {
  const f = { ...form }
  if (h.xf === 'frg' && Number.isFinite(xHz) && xHz > 0) f.frg = xHz * uf
  if (h.yf === 'gamma' && Number.isFinite(yS)) f.gamma = clamp(100 * (1 - yS / f.tau0), 0.01, 99)
  if (h.yf === 'tau0' && Number.isFinite(yS) && yS > 0) {
    // Keep the floor below τ0: at least 1 % tolerance (γ ≤ 99 % is the other bound).
    f.tau0 = Math.max(yS, 1e-15)
  }
  return f
}

/**
 * Worst relative delay drop up to f_rg vs the allowed γ.
 * @returns {null | { ok, worstPct, at (Hz), bad: (number|null)[] }}  bad = delay (s)
 *   where τ < τ_min (plus one neighbour each side), null elsewhere
 */
export function gdCompliance(g, bode, tolPct = 0.01) {
  if (!g || !bode?.freq?.length || !bode.groupDelay) return null
  const { freq, groupDelay: tau } = bode
  const n = freq.length
  let worst = -Infinity, at = null
  const hit = new Uint8Array(n)
  for (let i = 0; i < n; i++) {
    if (freq[i] > g.frgHz) break
    const drop = 100 * (1 - tau[i] / g.tau0)
    if (drop > worst) { worst = drop; at = freq[i] }
    if (drop > g.gamma + tolPct) hit[i] = 1
  }
  if (at == null) return null
  const bad = new Array(n).fill(null)
  for (let i = 0; i < n; i++) if (hit[i] || hit[i - 1] || hit[i + 1]) bad[i] = tau[i]
  return { ok: worst <= g.gamma + tolPct, worstPct: worst, at, bad }
}

/** Display unit for delays around τ0: { k (s → unit), unit }. */
export function delayUnit(tau0) {
  const e = Number.isFinite(tau0) && tau0 > 0 ? Math.floor(Math.log10(tau0) / 3) * 3 : -3
  const exp = clamp(e, -12, 0)
  return { k: 10 ** -exp, unit: { 0: 's', '-3': 'ms', '-6': 'µs', '-9': 'ns', '-12': 'ps' }[exp] }
}
