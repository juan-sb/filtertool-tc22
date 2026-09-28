// Zoom history + app-aware "home" for a Plotly graph div.
//
// Plotly's own buttons don't fit this app: zoom-out is a ×2 zoom about the
// centre (not "back"), and reset-axes / double-click return to the ranges seen
// when the plot was first created, while the app keeps setting new explicit
// ranges per design. This keeps a stack of user zoom / pan steps and resets to
// whatever the app currently wants (home()).

import Plotly from 'plotly.js-dist'

const RANGE_KEY = /^[xy]axis\.(range|autorange)/
const MAX_STEPS = 50

/** Custom "back" arrow for the mode bar (Plotly has no undo icon). */
export const BACK_ICON = {
  width: 1000, height: 1000,
  path: 'M420 170 L100 480 L420 790 L420 590 L900 590 L900 370 L420 370 Z',
}

/**
 * @param {HTMLElement} gd  Plotly graph div (after newPlot)
 * @param {() => { x: number[]|null, y: number[]|null }} home  axis ranges in
 *   Plotly units (log10 for log axes); null = autorange
 * @param {() => void} [onReset]  called before a reset (e.g. to drop frozen ranges)
 */
export function zoomHistory(gd, home, onReset) {
  const stack = []
  let internal = false
  const snapshot = () => {
    const fl = gd._fullLayout
    return fl?.xaxis && fl?.yaxis ? { x: fl.xaxis.range.slice(), y: fl.yaxis.range.slice() } : null
  }
  let last = snapshot()

  // plotly_relayout fires after a user zoom / pan (with the new ranges); push
  // the view from before it. Plotly.react (new designs) doesn't fire it, so
  // only re-baseline on react.
  const onRelayout = ev => {
    const touchesRange = Object.keys(ev || {}).some(k => RANGE_KEY.test(k))
    if (touchesRange && !internal && last) {
      stack.push(last)
      if (stack.length > MAX_STEPS) stack.shift()
    }
    last = snapshot()
  }
  const onReact = () => { if (!internal) last = snapshot() }
  gd.on('plotly_relayout', onRelayout)
  gd.on('plotly_react', onReact)

  // As a GUI edit (like a user zoom): with uirevision, Plotly re-applies GUI
  // ranges on the next redraw and would undo a plain API relayout.
  const relayout = Plotly._guiRelayout ?? Plotly.relayout
  async function apply(update) {
    internal = true
    try { await relayout(gd, update) } finally { internal = false; last = snapshot() }
  }

  return {
    /** Previous zoom / pan; resets when there's no history. */
    back() {
      const prev = stack.pop()
      if (!prev) return this.reset()
      return apply({ 'xaxis.range': prev.x, 'yaxis.range': prev.y })
    },
    /** The view the app currently intends. */
    reset() {
      stack.length = 0
      onReset?.()
      const h = home() ?? {}
      return apply({
        ...(h.x ? { 'xaxis.range': h.x.slice(), 'xaxis.autorange': false } : { 'xaxis.autorange': true }),
        ...(h.y ? { 'yaxis.range': h.y.slice(), 'yaxis.autorange': false } : { 'yaxis.autorange': true }),
      })
    },
    get depth() { return stack.length },
    detach() {
      gd.removeListener?.('plotly_relayout', onRelayout)
      gd.removeListener?.('plotly_react', onReact)
    },
  }
}

/** Mode-bar config: our Back + Home instead of Plotly's zoom-out / reset. */
export function zoomButtons(getHistory) {
  return {
    modeBarButtonsToAdd: [
      { name: 'zoomBack', title: 'Back to the previous zoom', icon: BACK_ICON, click: () => getHistory()?.back() },
      { name: 'resetView', title: 'Reset view (or double-click the plot)', icon: Plotly.Icons.home, click: () => getHistory()?.reset() },
    ],
    doubleClick: false,
  }
}
