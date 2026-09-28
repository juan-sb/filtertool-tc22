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

import { get, writable } from 'svelte/store'
import { getWorkerApi } from './worker-client.js'
import { stages, filterParams, filterResult, remainingPZ } from '../stores/app.js'
import { autoStages } from './auto-stage.js'
import { nextColorIndex } from './stage-colors.js'
import { moveRoot, withQ, poleSummary } from './stage-math.js'

const snapshot = s => ({ zeros: s.zeros, poles: s.poles, normtype: s.normtype, gainDb: s.gainDb })

/** A new stage from selected roots. */
export function makeStage({ id, name, zeroIds, poleIds, zeros, poles, normtype = 'Passband', gainDb = 0, orig = null, colorIndex = null }) {
  const s = { id, name, zeroIds, poleIds, zeros, poles, normtype, gainDb, colorIndex }
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

/**
 * Live preview: apply `patch` to the store without an engine rebuild (the
 * Stages tab draws it from lib/stage-eval.js); commitStage() rebuilds once.
 */
export function previewStage(id, patch) {
  stages.update(list => list.map(s => (s.id === id ? { ...s, ...(typeof patch === 'function' ? patch(s) : patch) } : s)))
}

/** Rebuild num / den after a preview. */
export function commitStage(id) {
  scheduleRebuild(id)
}

// ── Preview session (one stage at a time) ───────────────────────────────────
// { id, phase: 'drag' | 'commit', num0, t0 }. While set, the Stages tab
// freezes its Plotly traces and draws the stage live on a canvas; in 'commit'
// the engine is rebuilding and the tab clears the session once the new Bode is in.
export const stagePreview = writable(null)

export function beginStagePreview(id) {
  const cur = get(stagePreview)
  if (cur?.id === id && cur.phase === 'drag') return
  if (cur?.phase === 'drag') endStagePreview(true)
  stagePreview.set({ id, phase: 'drag', num0: get(stages).find(s => s.id === id)?.num, t0: 0 })
}

export function endStagePreview(changed) {
  clearTimeout(idleTimer)
  const cur = get(stagePreview)
  if (!cur || cur.phase !== 'drag') return
  if (!changed) { stagePreview.set(null); return }
  commitStage(cur.id)
  const next = { ...cur, phase: 'commit', t0: performance.now() }
  stagePreview.set(next)
  // Safety net if the rebuild never lands
  setTimeout(() => { if (get(stagePreview) === next) stagePreview.set(null) }, 2000)
}

/**
 * Card edits (typing, arrows, wheel, scrubbing, normalization): preview now,
 * commit to the engine after a short idle pause (flushStageEdit() commits at once).
 */
const IDLE_COMMIT_MS = 300
let idleTimer = null
export function editStageLive(id, patch) {
  beginStagePreview(id)
  previewStage(id, patch)
  clearTimeout(idleTimer)
  idleTimer = setTimeout(() => endStagePreview(true), IDLE_COMMIT_MS)
}
export function flushStageEdit() {
  if (get(stagePreview)?.phase === 'drag') endStagePreview(true)
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

/** Drag a staged root to (re, im) rad/s; poles stay in the LHP. preview: no engine rebuild. */
export function dragStageRoot(ref, re, im, snapIm, { preview = false } = {}) {
  const r = parseRootRef(ref)
  if (!r) return
  ;(preview ? previewStage : updateStage)(r.stageId, s => {
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

/** Build stages from all unassigned roots (lib/auto-stage.js) and append them. */
export async function autoStage() {
  const fr = get(filterResult)
  if (!fr?.roots) return 0
  const { sections } = autoStages(get(remainingPZ))
  if (!sections.length) return 0
  const byId = new Map([...fr.roots.zeros, ...fr.roots.poles].map(r => [r.id, [r.re, r.im]]))
  const ft = get(filterParams)?.filter_type ?? 0
  const base = get(stages).length
  const color0 = nextColorIndex(get(stages))
  const api = getWorkerApi()
  const built = await Promise.all(sections.map((sec, i) => buildStage(api, makeStage({
    id: Date.now() + i,
    name: `Stage ${base + i + 1}`,
    colorIndex: color0 + i,
    zeroIds: sec.zeroIds, poleIds: sec.poleIds,
    zeros: sec.zeroIds.map(id => byId.get(id)), poles: sec.poleIds.map(id => byId.get(id)),
  }), ft)))
  stages.update(list => [...list, ...built])
  return built.length
}

/** Move a stage to position `to` in the list (cascade order). */
export function moveStage(id, to) {
  stages.update(list => {
    const from = list.findIndex(s => s.id === id)
    if (from < 0 || to === from) return list
    const next = list.slice()
    const [s] = next.splice(from, 1)
    next.splice(Math.max(0, Math.min(next.length, to)), 0, s)
    return next
  })
}
