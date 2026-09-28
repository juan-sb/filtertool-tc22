// Svelte action: drag horizontally on an element (a field label) to scrub a value.
//
//   <span use:scrub={{ get: () => v, set: x => v = x, log: true, min, max }}>fp</span>
//
// log: true  → ×10 per 200 px (frequencies); log: false → `step` per 4 px (dB, %).
// Shift = 10× finer. Esc while dragging restores the starting value.

const LOG_PX_PER_DECADE = 200
const LIN_PX_PER_STEP = 4

export function scrub(node, opts) {
  let o = opts
  let startX = 0, startV = 0, dragging = false, moved = false

  node.classList.add('scrubbable')

  function clamp(v) {
    return Math.min(o.max ?? Infinity, Math.max(o.min ?? -Infinity, v))
  }

  function valueAt(dx, fine) {
    const k = fine ? 0.1 : 1
    if (o.log) return clamp(startV * Math.pow(10, (dx * k) / LOG_PX_PER_DECADE))
    const steps = Math.round((dx * k) / LIN_PX_PER_STEP)
    return clamp(startV + steps * (o.step ?? 1))
  }

  function onDown(e) {
    if (e.button !== 0 || o.disabled) return
    const v = Number(o.get())
    if (!Number.isFinite(v) || (o.log && !(v > 0))) return
    e.preventDefault()
    startX = e.clientX
    startV = v
    dragging = true
    moved = false
    node.setPointerCapture(e.pointerId)
    node.classList.add('scrubbing')
    window.addEventListener('keydown', onKey)
  }

  function onMove(e) {
    if (!dragging) return
    const dx = e.clientX - startX
    if (!moved && Math.abs(dx) < 2) return
    moved = true
    o.set(valueAt(dx, e.shiftKey))
  }

  function end(e) {
    if (!dragging) return
    dragging = false
    node.classList.remove('scrubbing')
    window.removeEventListener('keydown', onKey)
    if (e?.pointerId != null && node.hasPointerCapture(e.pointerId)) node.releasePointerCapture(e.pointerId)
    if (moved) o.done?.()
  }

  function onKey(e) {
    if (e.key !== 'Escape') return
    o.set(startV)
    moved = false
    end()
  }

  node.addEventListener('pointerdown', onDown)
  node.addEventListener('pointermove', onMove)
  node.addEventListener('pointerup', end)
  node.addEventListener('pointercancel', end)

  return {
    update(next) { o = next },
    destroy() {
      window.removeEventListener('keydown', onKey)
      node.removeEventListener('pointerdown', onDown)
      node.removeEventListener('pointermove', onMove)
      node.removeEventListener('pointerup', end)
      node.removeEventListener('pointercancel', end)
    },
  }
}
