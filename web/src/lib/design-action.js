// The Design action, shared by the Design button and template-drag release
// (and later the live denorm slider). Latest-wins: a call made while a design
// is in flight is coalesced into one re-run with the newest form.

import { get } from 'svelte/store'
import { TWO_PI, freqRangeFromParams } from './approx.js'
import { buildParams, validateForm } from './params.js'
import { withRoots } from './roots.js'
import { getWorkerApi } from './worker-client.js'
import {
  designForm, dataUnit, bodePoints, filterParams, filterResult, bodeData, stages,
  engineStatus, designBusy, designError,
} from '../stores/app.js'

let running = false
let rerun = false

/** Design from the current $designForm. Resolves true when a design was published. */
export async function runDesign() {
  if (running) { rerun = true; return false }
  running = true
  let ok = false
  try {
    do {
      rerun = false
      ok = await designOnce()
    } while (rerun)
  } finally {
    running = false
  }
  return ok
}

async function designOnce() {
  const form = get(designForm)
  const errs = Object.values(validateForm(form))
  if (errs.length) { designError.set(errs[0]); return false }

  const toRad = get(dataUnit) === 'rad' ? 1 : TWO_PI
  designBusy.set(true)
  designError.set('')
  engineStatus.set('Computing…')
  try {
    const params = buildParams(form, toRad)
    const api    = getWorkerApi()
    const result = await api.filterDesign(params)
    if (result.error) { designError.set(result.error.split('\n').at(-2) ?? result.error); return false }
    const r    = freqRangeFromParams(params)
    const bode = await api.computeBode(result.num, result.den, r.min, r.max, get(bodePoints))
    // Publish together so plots never pair a new design with an old Bode.
    stages.set([])
    filterParams.set(params)
    filterResult.set(withRoots(result))
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
