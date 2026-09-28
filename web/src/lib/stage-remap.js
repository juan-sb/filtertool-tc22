// Carry stages across a redesign: map each stage's roots (ids from the old
// design) onto the nearest roots of the new design. Complex roots are matched
// by their upper-half representative and bring their conjugate along; real
// roots match real roots only.

/** Relative distance between two roots. */
function dist(a, b) {
  const d = Math.hypot(a.re - b.re, a.im - b.im)
  return d / Math.max(Math.hypot(a.re, a.im), Math.hypot(b.re, b.im), 1e-12)
}

const isRep = r => r.conj === null || r.im > 0

/**
 * Greedy global nearest matching old → new for one root list.
 * @returns {Map<string, object> | null} old id → new root, or null if the lists
 *   can't correspond (different counts of real / complex roots).
 */
function matchRoots(oldRoots, newRoots) {
  if (oldRoots.length !== newRoots.length) return null
  const oldReps = oldRoots.filter(isRep), newReps = newRoots.filter(isRep)
  const complex = rs => rs.filter(r => r.conj !== null).length
  if (oldReps.length !== newReps.length || complex(oldReps) !== complex(newReps)) return null

  const pairs = []
  for (const o of oldReps)
    for (const n of newReps)
      if ((o.conj === null) === (n.conj === null)) pairs.push([dist(o, n), o, n])
  pairs.sort((a, b) => a[0] - b[0])

  const byId = new Map(newRoots.map(r => [r.id, r]))
  const map = new Map(), taken = new Set()
  for (const [, o, n] of pairs) {
    if (map.has(o.id) || taken.has(n.id)) continue
    map.set(o.id, n)
    taken.add(n.id)
    if (o.conj !== null) map.set(o.conj, byId.get(n.conj))
  }
  return map.size === oldRoots.length ? map : null
}

/**
 * @param {object[]} stages  stages referencing oldRoots by id
 * @param {{ zeros: object[], poles: object[] }} oldRoots
 * @param {{ zeros: object[], poles: object[] }} newRoots
 * @returns {object[] | null} stages with zeroIds / poleIds / zeros / poles
 *   pointing at newRoots (num / den still to be rebuilt), or null if they can't be carried.
 */
export function remapStages(stages, oldRoots, newRoots) {
  const zm = matchRoots(oldRoots.zeros, newRoots.zeros)
  const pm = matchRoots(oldRoots.poles, newRoots.poles)
  if (!zm || !pm) return null
  const out = []
  for (const s of stages) {
    const z = (s.zeroIds ?? []).map(id => zm.get(id))
    const p = (s.poleIds ?? []).map(id => pm.get(id))
    if (z.some(r => !r) || p.some(r => !r)) return null
    out.push({
      ...s,
      zeroIds: z.map(r => r.id), poleIds: p.map(r => r.id),
      zeros: z.map(r => [r.re, r.im]), poles: p.map(r => [r.re, r.im]),
    })
  }
  return out
}
