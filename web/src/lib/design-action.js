// The Design action, shared by the Design button, template-drag release and the
// live denorm slider. Latest-wins: a call made while a design is in flight is
// coalesced, and only the newest request runs next.
//
// Stages survive a redesign when they can: they're remapped onto the new
// design's roots and rebuilt with their own normalization. When they can't
// (other type / approximation / order, or REMAP_STAGES off) they're cleared
// with an "Undo" toast that restores the previous design, form and stages.

import { get } from 'svelte/store'
import { TWO_PI, freqRangeFromParams } from './approx.js'
import { buildParams, formFromParams, validateForm } from './params.js'
import { withRoots } from './roots.js'
import { remapStages } from './stage-remap.js'
import { getWorkerApi } from './worker-client.js'
import {
  designForm, dataUnit, bodePoints, filterParams, filterResult, bodeData, stages,
  engineStatus, designBusy, designError, toast,
} from '../stores/app.js'

/**
 * Q2: remap stages onto the new roots after a redesign. Set false to fall back
 * to clearing them (with Undo) if remapping ever costs too much while dragging.
 */
export const REMAP_STAGES = true

let running = false
/** @type {null | { params?: object }} newest request waiting for the in-flight one */
let pending = null

/**
 * Design and publish. With `params`, design exactly those (live denorm);
 * otherwise from the current $designForm.
 * @param {{ params?: object }} [request]
 * @returns {Promise<boolean>} true when the (last) run published a design
 */
export async function runDesign(request = {}) {
  if (running) { pending = request; return false }
  running = true
  let ok = false
  try {
    let next = request
    while (next) {
      pending = null
      ok = await designOnce(next)
      next = pending
    }
  } finally {
    running = false
  }
  return ok
}

const toRadNow = () => (get(dataUnit) === 'rad' ? 1 : TWO_PI)

async function designOnce({ params: given } = {}) {
  let params = given
  if (!params) {
    const form = get(designForm)
    const errs = Object.values(validateForm(form))
    if (errs.length) { designError.set(errs[0]); return false }
    params = buildParams(form, toRadNow())
  }

  designBusy.set(true)
  designError.set('')
  engineStatus.set('Computing…')
  try {
    const api = getWorkerApi()
    const raw = await api.filterDesign(params)
    if (raw.error) { designError.set(raw.error.split('\n').at(-2) ?? raw.error); return false }
    const result = withRoots(raw)
    const r      = freqRangeFromParams(params)
    const bode   = await api.computeBode(result.num, result.den, r.min, r.max, get(bodePoints))
    const prev   = { params: get(filterParams), result: get(filterResult), bode: get(bodeData), stages: get(stages) }
    const carried = await carryStages(api, prev, params, result)

    // Publish together so plots never pair a new design with an old Bode / stages.
    stages.set(carried)
    filterParams.set(params)
    filterResult.set(result)
    bodeData.set(bode)
    return true
  } catch (e) {
    designError.set(e?.message ?? String(e))
    return false
  } finally {
    designBusy.set(false)
    engineStatus.set('Ready')
  }
}

async function carryStages(api, prev, params, result) {
  if (!prev.stages.length) return []
  if (REMAP_STAGES && prev.result?.roots && prev.params
      && prev.params.filter_type === params.filter_type
      && prev.params.approx_type === params.approx_type) {
    const moved = remapStages(prev.stages, prev.result.roots, result.roots)
    if (moved) {
      const rebuilt = await Promise.all(moved.map(async s => {
        const r = await api.buildStageFromZPK(s.zeros, s.poles, 1, s.normtype ?? 'Passband', params.filter_type)
        return r.error ? null : { ...s, gain: r.gain, num: r.num, den: r.den }
      }))
      if (rebuilt.every(Boolean)) return rebuilt
    }
  }
  offerUndo(prev)
  return []
}

function offerUndo(prev) {
  const n = prev.stages.length
  toast.set({
    message: `${n} stage${n === 1 ? '' : 's'} cleared: the new design's poles and zeros don't match the old ones.`,
    actionLabel: 'Undo',
    timeoutMs: 10000,
    onAction: () => {
      designForm.update(f => formFromParams(prev.params, toRadNow(), f))
      filterParams.set(prev.params)
      filterResult.set(prev.result)
      bodeData.set(prev.bode)
      stages.set(prev.stages)
    },
  })
}
