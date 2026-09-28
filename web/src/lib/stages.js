// Stage model and actions.
//
// Stage: {
//   id, name,
//   zeroIds, poleIds,     // links to filterResult.roots (designed roots)
//   zeros, poles,         // current [re, im] values (rad/s); may be moved by the user
//   normtype,             // 'Passband' | 'ω→0' | 'ω→∞' | 'ω→ω0'
//   gainDb,               // user gain on top of the normalization
//   gain, num, den,       // from the engine: gain = normalization · 10^(gainDb/20)
//   orig: { zeros, poles, normtype, gainDb },   // as built (Reset)
// }
//
// Edits (updateStage) apply to the store at once and rebuild num/den through
// the engine worker, latest-wins per stage, so drags stay responsive.

import { get } from 'svelte/store'
import { getWorkerApi } from './worker-client.js'
import { stages, filterParams } from '../stores/app.js'
import { moveRoot, withQ, poleSummary } from './stage-math.js'

const snapshot = s => ({ zeros: s.zeros, poles: s.poles, normtype: s.normtype, gainDb: s.gainDb })

/** A new stage from selected roots. */
export function makeStage({ id, name, zeroIds, poleIds, zeros, poles, normtype = 'Passband', gainDb = 0, orig = null }) {
  const s = { id, name, zeroIds, poleIds, zeros, poles, normtype, gainDb }
  return { ...s, orig: orig ?? snapshot(s) }
}

const sameRoots = (a, b) =>
  a.length === b.length && a.every((r, i) => Math.abs(r[0] - b[i][0]) < 1e-9 * Math.max(1, Math.abs(b[i][0])) &&
                                          Math.abs(r[1] - b[i][1]) < 1e-9 * Math.max(1, Math.abs(b[i][1])))

/** Roots moved away from where the stage was built. */
export const rootsModified = s => !!s.orig && !(sameRoots(s.zeros, s.orig.zeros) && sameRoots(s.poles, s.orig.poles))

/** Anything (roots, normalization, gain) differs from the built stage. */
export const isModified = s =>
  rootsModified(s) || (s.orig && (s.normtype !== s.orig.normtype || Math.abs((s.gainDb ?? 0) - (s.orig.gainDb ?? 0)) > 1e-9))

/** Rebuild num / den / gain for a stage through the engine. */
export async function buildStage(api, s, filterType) {
  const k = Math.pow(10, (s.gainDb ?? 0) / 20)
  const r = await api.buildStageFromZPK(s.zeros, s.poles, k, s.normtype ?? 'Passband', filterType)
  if (r.error) throw new Error(r.error)
  return { ...s, gain: r.gain, num: r.num, den: r.den }
}

// ── Live edits ──────────────────────────────────────────────────────────────
const inflight = new Map()   // stage id → true while a rebuild runs
const queued   = new Map()   // stage id → true when another rebuild is needed

/**
 * Apply `patch` (object, or function stage → patch) to a stage now, then
 * rebuild its num / den in the background (latest-wins per stage).
 */
export function updateStage(id, patch) {
  let found = false
  stages.update(list => list.map(s => {
    if (s.id !== id) return s
    found = true
    return { ...s, ...(typeof patch === 'function' ? patch(s) : patch) }
  }))
  if (found) scheduleRebuild(id)
}

function scheduleRebuild(id) {
  if (inflight.get(id)) { queued.set(id, true); return }
  inflight.set(id, true)
  ;(async () => {
    try {
      do {
        queued.delete(id)
        const s = get(stages).find(st => st.id === id)
        if (!s) break
        const ft = get(filterParams)?.filter_type ?? 0
        try {
          const built = await buildStage(getWorkerApi(), s, ft)
          // Only publish if nothing edited the stage meanwhile (else the queued run will).
          if (!queued.get(id)) {
            stages.update(list => list.map(st => (st.id === id ? { ...st, gain: built.gain, num: built.num, den: built.den } : st)))
          }
        } catch (e) {
          console.warn('stage rebuild failed', e)
        }
      } while (queued.get(id))
    } finally {
      inflight.delete(id)
    }
  })()
}

/** Back to the stage as built. */
export function resetStage(id) {
  updateStage(id, s => (s.orig ? { ...s.orig } : {}))
}

export function resetAllStages() {
  for (const s of get(stages)) if (isModified(s)) resetStage(s.id)
}

export function removeStage(id) {
  stages.update(list => list.filter(s => s.id !== id))
}

// ── Root interaction on PZ maps ─────────────────────────────────────────────
// Staged roots on a PzMap use refs 's:<stageId>:<p|z>:<index>'.

export const rootRef = (stageId, kind, index) => `s:${stageId}:${kind}:${index}`

/** { stageId, kind: 'p' | 'z', index } or null. */
export function parseRootRef(ref) {
  const m = typeof ref === 'string' && ref.match(/^s:([^:]+):([pz]):(\d+)$/)
  return m ? { stageId: Number(m[1]), kind: m[2], index: Number(m[3]) } : null
}

/** Drag a staged root to (re, im) rad/s; poles stay in the LHP. */
export function dragStageRoot(ref, re, im, snapIm) {
  const r = parseRootRef(ref)
  if (!r) return
  updateStage(r.stageId, s => {
    const list = r.kind === 'p' ? s.poles : s.zeros
    if (r.index >= list.length) return {}
    const scale = Math.max(1e-12, ...s.poles.map(([a, b]) => Math.hypot(a, b)))
    const moved = moveRoot(list, r.index, [re, im], { snapIm, lhp: r.kind === 'p', minRe: 1e-6 * scale })
    return r.kind === 'p' ? { poles: moved } : { zeros: moved }
  })
}

/** Wheel over a pole: Q × 1.1^dir at fixed ω0 (2-pole stages). */
export function wheelStageQ(stageId, dir) {
  updateStage(stageId, s => {
    if (s.poles.length !== 2) return {}
    const { q } = poleSummary(s.poles)
    return Number.isFinite(q) ? { poles: withQ(s.poles, q * Math.pow(1.1, dir)) } : {}
  })
}
