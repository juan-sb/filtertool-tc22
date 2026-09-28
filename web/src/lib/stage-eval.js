// Fast JS evaluation of a stage's |H(jω)| straight from its roots, for live
// drag previews (the engine still rebuilds num / den on release). Mirrors the
// engine's build_stage(): num = k·Π(s − z), den = Π(s − p), k = norm · gain.

import { resolveNorm } from './stage-math.js'

/** |Π(jω − r)| over roots [re, im]. */
function prodAbs(roots, w) {
  let m = 1
  for (const [re, im] of roots) m *= Math.hypot(-re, w - im)
  return m
}

/** Engine normalization gain for (zeros, poles) with gain 1. */
export function normGain(zeros, poles, normtype, filterType) {
  const n = resolveNorm(normtype ?? 'Passband', filterType)
  if (n === 'ω→0') {
    // |Π p / Π z| over roots away from the origin
    let pp = 1, zp = 1
    for (const [re, im] of poles) { const a = Math.hypot(re, im); if (a >= 1e-5) pp *= a }
    for (const [re, im] of zeros) { const a = Math.hypot(re, im); if (a >= 1e-5) zp *= a }
    return pp / zp
  }
  if (n === 'ω→∞') return 1
  if (n === 'ω→ω0') {
    const w0 = Math.hypot(...poles[0])
    const v = prodAbs(zeros, w0) / prodAbs(poles, w0)
    return v > 1e-12 ? 1 / v : 1
  }
  return 1
}

/** |H| in dB at each frequency (Hz) for a stage { zeros, poles, normtype, gainDb }. */
export function stageDb(stage, filterType, freqHz, out = new Float64Array(freqHz.length)) {
  const k = normGain(stage.zeros, stage.poles, stage.normtype, filterType) * Math.pow(10, (stage.gainDb ?? 0) / 20)
  const kDb = 20 * Math.log10(k)
  for (let i = 0; i < freqHz.length; i++) {
    const w = 2 * Math.PI * freqHz[i]
    const m = prodAbs(stage.zeros, w) / prodAbs(stage.poles, w)
    out[i] = m > 0 ? kDb + 20 * Math.log10(m) : -Infinity
  }
  return out
}
