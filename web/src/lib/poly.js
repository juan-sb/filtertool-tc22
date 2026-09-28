// Evaluate transfer functions given as descending-power real coefficients.

/** |P(jω)| (Horner: P ← P·jω + a). */
export function polyAbs(c, w) {
  let re = 0, im = 0
  for (const a of c) [re, im] = [a - im * w, re * w]
  return Math.hypot(re, im)
}

/** Degree ignoring leading (near-)zero coefficients. */
function degree(c) {
  let i = 0
  while (i < c.length - 1 && Math.abs(c[i]) < 1e-300) i++
  return c.length - 1 - i
}

/**
 * |H(jω)| = |num(jω) / den(jω)|, including the limits ω = 0 and ω = ∞.
 */
export function tfAbs(num, den, w) {
  if (w === Infinity) {
    const dn = degree(num), dd = degree(den)
    if (dn < dd) return 0
    if (dn > dd) return Infinity
    return Math.abs(num[num.length - 1 - dn] / den[den.length - 1 - dd])
  }
  return polyAbs(num, w) / polyAbs(den, w)
}

export const toDb = m => (m > 0 ? 20 * Math.log10(m) : -Infinity)
