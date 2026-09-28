// Magnitude template from the live design form: geometry (Hz / dB), drag
// handles, drag → form updates with clamping, and compliance of a Bode curve.
//
// Units: geometry is in Hz and dB. Form frequencies are in the data unit;
// `uf` = data units per Hz (1 for Hz, 2π for rad/s).

import { LP, HP, BP, BR, GD, F0_BW, isBand, bandEdges } from './params.js'

/** Compliance tolerance (dB): designs meet the spec exactly at the edges. */
export const TOL_DB = 0.01
/** Minimum ratio between neighbouring edges while dragging. */
const EPS = 1.002
/** Minimum gap between ripple and attenuation (dB). */
const MIN_DB_GAP = 0.1

/**
 * @returns {null | {
 *   ft: number, f0bw: boolean, gainDb: number, passDb: number, stopDb: number,
 *   f0: number|null, pass: [number, number][], stop: [number, number][],
 *   passEdges: Edge[], stopEdges: Edge[] }}
 * Edge = { x: Hz, xf: form field, side: 0|1|null }
 * pass / stop: frequency intervals (Hz; 0 / Infinity open ends) where |H| must
 * stay ≥ passDb / ≤ stopDb.
 */
export function templateGeom(form, uf) {
  const ft = form.filterType
  if (ft === GD) return null
  const hz = v => v / uf
  const passDb = form.gainDb - form.apDb
  const stopDb = form.gainDb - form.aaDb
  const f0bw = isBand(ft) && form.defineWith === F0_BW

  if (!isBand(ft)) {
    const fp = hz(form.fp), fa = hz(form.fa)
    return {
      ft, f0bw, gainDb: form.gainDb, passDb, stopDb, f0: null,
      pass: ft === LP ? [[0, fp]] : [[fp, Infinity]],
      stop: ft === LP ? [[fa, Infinity]] : [[0, fa]],
      passEdges: [{ x: fp, xf: 'fp', side: null }],
      stopEdges: [{ x: fa, xf: 'fa', side: null }],
    }
  }

  const fp = f0bw ? bandEdges(form.f0, form.bwp).map(hz) : [hz(form.fp1), hz(form.fp2)]
  const fa = f0bw ? bandEdges(form.f0, form.bwa).map(hz) : [hz(form.fa1), hz(form.fa2)]
  const edges = (xs, f0bwField, fields) =>
    xs.map((x, side) => ({ x, xf: f0bw ? f0bwField : fields[side], side }))
  return {
    ft, f0bw, gainDb: form.gainDb, passDb, stopDb, f0: f0bw ? hz(form.f0) : null,
    pass: ft === BP ? [fp] : [[0, fp[0]], [fp[1], Infinity]],
    stop: ft === BP ? [[0, fa[0]], [fa[1], Infinity]] : [fa],
    passEdges: edges(fp, 'bwp', ['fp1', 'fp2']),
    stopEdges: edges(fa, 'bwa', ['fa1', 'fa2']),
  }
}

/**
 * Drag handles. kind: 'v' vertical edge (moves x), 'h' horizontal edge (moves y),
 * 'c' corner (both), 'f0' centre line (moves the whole band).
 * x / y in Hz / dB; span = y-range of a 'v' edge or x-range of an 'h' edge
 * (±Infinity / 0 = open to the plot border). xf / yf = form fields edited by the
 * x / y motion; they are also the hover-link keys.
 */
export function templateHandles(g) {
  if (!g) return []
  const out = []
  const add = h => out.push({ ...h, id: `${h.kind}:${h.xf ?? ''}:${h.yf ?? ''}:${h.side ?? ''}:${out.length}` })

  for (const [x0, x1] of g.pass) add({ kind: 'h', group: 'pass', y: g.passDb, span: [x0, x1], yf: 'apDb' })
  for (const [x0, x1] of g.stop) add({ kind: 'h', group: 'stop', y: g.stopDb, span: [x0, x1], yf: 'aaDb' })
  for (const e of g.passEdges) {
    add({ kind: 'v', group: 'pass', x: e.x, span: [-Infinity, g.passDb], xf: e.xf, side: e.side, edge: 'p' })
    add({ kind: 'c', group: 'pass', x: e.x, y: g.passDb, xf: e.xf, yf: 'apDb', side: e.side, edge: 'p' })
  }
  for (const e of g.stopEdges) {
    add({ kind: 'v', group: 'stop', x: e.x, span: [g.stopDb, Infinity], xf: e.xf, side: e.side, edge: 'a' })
    add({ kind: 'c', group: 'stop', x: e.x, y: g.stopDb, xf: e.xf, yf: 'aaDb', side: e.side, edge: 'a' })
  }
  if (g.f0 != null) add({ kind: 'f0', group: 'centre', x: g.f0, span: [-Infinity, Infinity], xf: 'f0' })
  return out
}

const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v))

/**
 * New form after dragging handle `h` to (xHz, yDb). Pass the form captured at
 * drag start: only the dragged fields change, clamped so the template stays valid.
 */
export function dragTo(form, h, xHz, yDb, uf) {
  const f = { ...form }
  if (h.xf && Number.isFinite(xHz) && xHz > 0) moveX(f, h, xHz * uf)
  if (h.yf && Number.isFinite(yDb)) moveY(f, h, yDb)
  return f
}

function moveY(f, h, y) {
  if (h.yf === 'apDb') f.apDb = clamp(f.gainDb - y, 0.01, f.aaDb - MIN_DB_GAP)
  else f.aaDb = clamp(f.gainDb - y, f.apDb + MIN_DB_GAP, 300)
}

function moveX(f, h, x) {
  const ft = f.filterType
  if (h.xf === 'f0') { f.f0 = x; return }

  if (!isBand(ft)) {
    if (h.xf === 'fp') f.fp = ft === LP ? Math.min(x, f.fa / EPS) : Math.max(x, f.fa * EPS)
    else               f.fa = ft === LP ? Math.max(x, f.fp * EPS) : Math.min(x, f.fp / EPS)
    return
  }

  if (f.defineWith === F0_BW) {
    // The dragged edge sits below (side 0) or above (side 1) f0; its band is
    // geometrically symmetric, so lo·hi = f0² gives the arithmetic width.
    const f0 = f.f0
    const xs = h.side === 0 ? Math.min(x, f0 / EPS) : Math.max(x, f0 * EPS)
    let bw = h.side === 0 ? (f0 * f0) / xs - xs : xs - (f0 * f0) / xs
    // BP: bwp < bwa ; BR: bwa < bwp
    const inner = (ft === BP) === (h.edge === 'p')
    const other = h.edge === 'p' ? f.bwa : f.bwp
    bw = inner ? Math.min(bw, other / EPS) : Math.max(bw, other * EPS)
    f[h.edge === 'p' ? 'bwp' : 'bwa'] = bw
    return
  }

  // Band edges: keep the frequency order of the template.
  const order = ft === BP ? ['fa1', 'fp1', 'fp2', 'fa2'] : ['fp1', 'fa1', 'fa2', 'fp2']
  const i = order.indexOf(h.xf)
  const lo = i > 0 ? f[order[i - 1]] * EPS : 0
  const hi = i < order.length - 1 ? f[order[i + 1]] / EPS : Infinity
  f[h.xf] = clamp(x, lo, hi)
}

/**
 * Check a Bode curve against the template.
 * @returns {null | { pass: Check|null, stop: Check|null, bad: (number|null)[] }}
 * Check = { ok, margin (dB, worst case, negative = violation), at (Hz) }.
 * bad = dB value where the curve breaks the template (plus one neighbour on each
 * side so short runs still draw), null elsewhere.
 */
export function compliance(g, bode) {
  if (!g || !bode?.freq?.length) return null
  const { freq, magnitude } = bode
  const n = freq.length
  const within = (bands, f) => bands.some(([a, b]) => f >= a && f <= b)
  const worst = { pass: { margin: Infinity, at: null }, stop: { margin: Infinity, at: null } }
  const hit = new Uint8Array(n)
  const db = new Float64Array(n)

  for (let i = 0; i < n; i++) {
    const f = freq[i], m = magnitude[i]
    db[i] = m > 0 ? 20 * Math.log10(m) : -Infinity
    if (within(g.pass, f)) {
      const d = db[i] - g.passDb
      if (d < worst.pass.margin) worst.pass = { margin: d, at: f }
      if (d < -TOL_DB) hit[i] = 1
    }
    if (within(g.stop, f)) {
      const d = g.stopDb - db[i]
      if (d < worst.stop.margin) worst.stop = { margin: d, at: f }
      if (d < -TOL_DB) hit[i] = 1
    }
  }

  const bad = new Array(n).fill(null)
  for (let i = 0; i < n; i++) {
    if (!hit[i] && !hit[i - 1] && !hit[i + 1]) continue
    if (Number.isFinite(db[i])) bad[i] = db[i]
  }
  const check = w => (w.at == null ? null : { ok: w.margin >= -TOL_DB, margin: w.margin, at: w.at })
  return { pass: check(worst.pass), stop: check(worst.stop), bad }
}
