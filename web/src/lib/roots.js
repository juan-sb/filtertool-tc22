// Stable identity for poles/zeros. Roots are identified by id, never by value,
// so repeated roots (e.g. the N zeros at s = 0 of a band-pass) stay distinct.
//
// Root: { id, re, im, conj }   conj = id of the conjugate partner, or null if real.

const isRealVal = (re, im) => Math.abs(im) <= 1e-9 * Math.max(1, Math.hypot(re, im))

/** Tag [[re, im], …] with ids `${prefix}0…` and pair complex conjugates. */
export function tagRoots(list, prefix) {
  const roots = list.map(([re, im], i) => ({ id: `${prefix}${i}`, re, im, conj: null }))
  const upper = roots.filter(r => !isRealVal(r.re, r.im) && r.im > 0)
  const lower = roots.filter(r => !isRealVal(r.re, r.im) && r.im < 0)
  // Greedy nearest match of each upper root to an unpaired lower root at (re, −im).
  for (const u of upper) {
    let best = null, bestD = Infinity
    for (const l of lower) {
      if (l.conj) continue
      const d = Math.hypot(l.re - u.re, l.im + u.im)
      if (d < bestD) { bestD = d; best = l }
    }
    if (best) { u.conj = best.id; best.conj = u.id }
  }
  // Numerically complex roots with no partner are treated as real (snap im to 0).
  for (const r of roots) if (!r.conj && !isRealVal(r.re, r.im)) r.im = 0
  return roots
}

/** Adds `roots: { zeros, poles }` to a filterDesign() result. */
export function withRoots(result) {
  return { ...result, roots: { zeros: tagRoots(result.zeros, 'z'), poles: tagRoots(result.poles, 'p') } }
}

export const isComplexRoot = r => r.conj !== null
export const rootValue = r => [r.re, r.im]
