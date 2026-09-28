// Auto-stage: split the unassigned roots into ≤ 2nd-order sections, in the
// spirit of SciPy's zpk2sos(pairing='minimal') used by the desktop app:
// the highest-Q pole pairs are paired with the nearest complex zeros first,
// real zeros (incl. those at s = 0) are spread over the sections that still
// have room, leftover real poles become first-order sections, and the result
// is ordered from low to high Q.

import { poleSummary } from './stage-math.js'

const isRep = r => r.conj === null || r.im > 0
const mag = r => Math.hypot(r.re, r.im)
const relDist = (a, b) => Math.abs(mag(a) - mag(b)) / Math.max(mag(a), mag(b), 1e-12)

/**
 * @param {{ zeros: object[], poles: object[] }} remaining  root objects (lib/roots.js)
 * @returns {{ sections: { zeroIds: string[], poleIds: string[], q: number|null }[], leftoverZeros: number }}
 */
export function autoStages(remaining) {
  const byId = new Map([...remaining.zeros, ...remaining.poles].map(r => [r.id, r]))
  const pair = r => (r.conj !== null && byId.has(r.conj) ? [r, byId.get(r.conj)] : [r])

  // Pole sections: complex pairs, then single real poles
  const sections = remaining.poles.filter(isRep).map(r => {
    const poles = pair(r)
    const { q } = poleSummary(poles.map(p => [p.re, p.im]))
    return { poles, zeros: [], q: poles.length === 2 ? q : null, ref: r }
  })
  // Highest Q first while pairing
  const byQ = sections.slice().sort((a, b) => (b.q ?? -1) - (a.q ?? -1))

  const zeroReps = remaining.zeros.filter(isRep)
  const complexZ = zeroReps.filter(z => z.conj !== null && byId.has(z.conj))
  const realZ = zeroReps.filter(z => z.conj === null)

  for (const sec of byQ) {
    if (sec.poles.length !== 2 || !complexZ.length) continue
    let best = 0
    for (let i = 1; i < complexZ.length; i++) if (relDist(complexZ[i], sec.ref) < relDist(complexZ[best], sec.ref)) best = i
    sec.zeros.push(...pair(complexZ.splice(best, 1)[0]))
  }

  // Real zeros: fewest-zeros sections first, nearest zero to the section's poles.
  while (realZ.length) {
    const open = byQ.filter(s => s.zeros.length < s.poles.length)
    if (!open.length) break
    open.sort((a, b) => a.zeros.length - b.zeros.length)
    const sec = open[0]
    let best = 0
    for (let i = 1; i < realZ.length; i++) if (relDist(realZ[i], sec.ref) < relDist(realZ[best], sec.ref)) best = i
    sec.zeros.push(realZ.splice(best, 1)[0])
  }

  // Low Q first (first-order sections count as lowest)
  const ordered = sections.slice().sort((a, b) => (a.q ?? 0) - (b.q ?? 0))
  return {
    sections: ordered.map(s => ({ zeroIds: s.zeros.map(z => z.id), poleIds: s.poles.map(p => p.id), q: s.q })),
    leftoverZeros: complexZ.length * 2 + realZ.length,
  }
}
