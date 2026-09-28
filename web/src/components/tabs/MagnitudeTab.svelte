<script>
  import { onMount, onDestroy } from 'svelte'
  import {
    bodeData, filterParams, comparisons, theme, compareDash, colorMode, colorShuffle, activeTab,
    plotUnit, dataUnit, designForm, hoveredFields, designBusy,
  } from '../../stores/app.js'
  import { APPROX_NAMES, plotColor, compareLine, freqAxis, freqRangeFromParams, TWO_PI } from '../../lib/approx.js'
  import { GD, buildParams, paramsClose, validateForm } from '../../lib/params.js'
  import { templateGeom, templateHandles, dragTo, compliance } from '../../lib/template.js'
  import { runDesign } from '../../lib/design-action.js'
  import { formatSI } from '../../lib/si.js'
  import BodePlot from '../BodePlot.svelte'

  export let showTemplate = true

  function toDb(v) {
    if (!(v > 0)) return null
    const db = 20 * Math.log10(v)
    return Number.isFinite(db) ? db : null
  }

  $: axis   = freqAxis($plotUnit)
  $: uf     = $dataUnit === 'rad' ? TWO_PI : 1
  $: toRad  = TWO_PI / uf
  $: tabId  = showTemplate ? 'template' : 'magnitude'

  // ── Live template (from the form, not the last design) ───────────────────
  $: formValid = Object.keys(validateForm($designForm)).length === 0
  let lastGeom = null
  $: if (showTemplate && formValid) lastGeom = templateGeom($designForm, uf)
  // An invalid form keeps showing the last valid template.
  $: geom    = showTemplate ? (formValid ? templateGeom($designForm, uf) : lastGeom) : null
  $: handles = templateHandles(geom)
  $: stale   = showTemplate && !!$filterParams && formValid &&
               !paramsClose(buildParams($designForm, toRad), $filterParams)
  $: comp    = showTemplate ? compliance(geom, $bodeData) : null

  // ── Drag state ───────────────────────────────────────────────────────────
  /** @type {null | { h: any, gx: number, gy: number, startForm: any, moved: boolean }} */
  let drag = null
  let dragging = false
  let hoverId = null
  let label = null          // { x, y, text } floating value label
  let plot                  // BodePlot instance

  // ── Axis ranges, frozen while dragging so the plot doesn't move under the cursor
  let xRange = null, yRange = null
  $: if (!dragging) {
    const fr = $bodeData?.freq?.length
      ? { min: $bodeData.freq[0], max: $bodeData.freq[$bodeData.freq.length - 1] }
      : freqRangeFromParams(formValid ? buildParams($designForm, toRad) : $filterParams)
    xRange = showTemplate ? [fr.min * axis.scale, fr.max * axis.scale] : null
    const f = formValid ? $designForm : null
    yRange = showTemplate && f && f.filterType !== GD ? [f.gainDb - 2 * f.aaDb, f.gainDb] : null
  }

  // ── Traces (untouched while dragging: only shapes move) ──────────────────
  const DANGER = { dark: '#f85149', light: '#cf222e' }
  let traces = []
  $: if (!dragging) traces = [
    ...($bodeData ? [{
      x: $bodeData.freq.map(f => f * axis.scale),
      y: $bodeData.magnitude.map(toDb),
      mode: 'lines',
      name: APPROX_NAMES[$filterParams?.approx_type ?? 0],
      line: { color: plotColor($filterParams?.approx_type ?? 0, $theme, $colorMode, $colorShuffle), width: 2 },
    }] : []),
    ...$comparisons.map(c => ({
      x: c.bodeData.freq.map(f => f * axis.scale),
      y: c.bodeData.magnitude.map(toDb),
      mode: 'lines',
      name: APPROX_NAMES[c.approxType],
      line: compareLine(c.approxType, $theme, { dash: $compareDash, mode: $colorMode, shuffle: $colorShuffle }),
    })),
    ...($bodeData && comp?.bad?.some(v => v !== null) ? [{
      x: $bodeData.freq.map(f => f * axis.scale),
      y: comp.bad,
      mode: 'lines',
      name: 'Outside template',
      line: { color: DANGER[$theme] ?? DANGER.dark, width: 3.5 },
      hoverinfo: 'skip',
    }] : []),
  ]

  // ── Template shapes ──────────────────────────────────────────────────────
  const HANDLE = {
    dark:  { pass: '#3fb950', stop: '#d29922', centre: '#8b949e', bg: '#0d1117' },
    light: { pass: '#1a7f37', stop: '#9a6700', centre: '#57606a', bg: '#f6f8fa' },
  }
  const X_OPEN = [1e-30, 1e30], Y_OPEN = [-1e4, 1e4]

  function buildShapes(g, hs, ax, th, hovered, activeId) {
    if (!g) return []
    const C = HANDLE[th] ?? HANDLE.dark
    const X = v => (v <= 0 ? X_OPEN[0] : v === Infinity ? X_OPEN[1] : v * ax.scale)
    const Y = v => (v === -Infinity ? Y_OPEN[0] : v === Infinity ? Y_OPEN[1] : v)
    const fill = th === 'light' ? 'rgba(255, 204, 203, 0.45)' : 'rgba(248, 81, 73, 0.16)'
    const rect = (x0, x1, y0, y1) => ({
      type: 'rect', xref: 'x', yref: 'y', layer: 'below',
      x0: X(x0), x1: X(x1), y0: Y(y0), y1: Y(y1), fillcolor: fill, line: { width: 0 },
    })
    const out = [
      ...g.pass.map(([a, b]) => rect(a, b, -Infinity, g.passDb)),
      ...g.stop.map(([a, b]) => rect(a, b, g.stopDb, Infinity)),
    ]
    const lit = h => h.id === activeId || hovered.includes(h.xf) || hovered.includes(h.yf)
    for (const h of hs) {
      const color = C[h.group], on = lit(h)
      if (h.kind === 'v' || h.kind === 'f0') {
        out.push({
          type: 'line', xref: 'x', yref: 'y', x0: X(h.x), x1: X(h.x), y0: Y(h.span[0]), y1: Y(h.span[1]),
          line: { color, width: on ? 3 : 1.5, ...(h.kind === 'f0' ? { dash: 'dash' } : {}) },
          opacity: on ? 1 : 0.8,
        })
      } else if (h.kind === 'h') {
        out.push({
          type: 'line', xref: 'x', yref: 'y', x0: X(h.span[0]), x1: X(h.span[1]), y0: h.y, y1: h.y,
          line: { color, width: on ? 3 : 1.5 }, opacity: on ? 1 : 0.8,
        })
      }
    }
    // Corner grips on top of the edges
    for (const h of hs) {
      if (h.kind !== 'c') continue
      const r = lit(h) ? 6 : 4
      out.push({
        type: 'circle', xref: 'x', yref: 'y', xsizemode: 'pixel', ysizemode: 'pixel',
        xanchor: X(h.x), yanchor: h.y, x0: -r, x1: r, y0: -r, y1: r,
        fillcolor: C[h.group], line: { color: C.bg, width: 1.5 },
      })
    }
    return out
  }

  $: shapes = showTemplate ? buildShapes(geom, handles, axis, $theme, $hoveredFields, drag?.h.id ?? hoverId) : []

  // ── Pixel geometry / hit testing ─────────────────────────────────────────
  const HIT_EDGE = 6, HIT_CORNER = 9

  function axesOf(gd) {
    const fl = gd?._fullLayout
    return fl?.xaxis && fl?.yaxis ? { xa: fl.xaxis, ya: fl.yaxis } : null
  }
  // Plot-unit x / dB y ↔ pixels relative to the graph div. On log axes d2l is log10.
  const xPx = (xa, x) => xa._offset + xa.l2p(xa.d2l(x))
  const yPx = (ya, y) => ya._offset + ya.l2p(ya.d2l(y))

  function hitTest(gd, px, py) {
    const ax = axesOf(gd)
    if (!ax) return null
    const { xa, ya } = ax
    const L = xa._offset, R = L + xa._length, T = ya._offset, B = T + ya._length
    if (px < L - HIT_EDGE || px > R + HIT_EDGE || py < T - HIT_EDGE || py > B + HIT_EDGE) return null
    const clampX = v => Math.min(R, Math.max(L, v)), clampY = v => Math.min(B, Math.max(T, v))
    const hxOf = x => clampX(x <= 0 ? -Infinity : x === Infinity ? Infinity : xPx(xa, x * axis.scale))
    const hyOf = y => clampY(y === -Infinity ? Infinity : y === Infinity ? -Infinity : yPx(ya, y))
    let best = null, bestScore = Infinity
    for (const h of handles) {
      let score = Infinity
      if (h.kind === 'c') {
        const d = Math.hypot(px - hxOf(h.x), py - hyOf(h.y))
        if (d <= HIT_CORNER) score = d - 100
      } else if (h.kind === 'v' || h.kind === 'f0') {
        const hx = hxOf(h.x)
        const [y0, y1] = [hyOf(h.span[1]), hyOf(h.span[0])]   // pixel y grows downward
        if (Math.abs(px - hx) <= HIT_EDGE && py >= y0 - HIT_EDGE && py <= y1 + HIT_EDGE)
          score = Math.abs(px - hx) + (h.kind === 'f0' ? 50 : 0)
      } else if (h.kind === 'h') {
        const hy = hyOf(h.y)
        const [x0, x1] = [hxOf(h.span[0]), hxOf(h.span[1])]
        if (Math.abs(py - hy) <= HIT_EDGE && px >= x0 - HIT_EDGE && px <= x1 + HIT_EDGE) score = Math.abs(py - hy)
      }
      if (score < bestScore) { bestScore = score; best = h }
    }
    return best
  }

  const CURSOR = { v: 'ew', f0: 'ew', h: 'ns', c: 'move' }
  function setCursor(gd, h) {
    if (h) gd.dataset.tplCursor = CURSOR[h.kind]
    else delete gd.dataset.tplCursor
  }

  const fieldsOf = h => [h.xf, h.yf].filter(Boolean)
  let ownHover = false
  function setHover(h) {
    hoverId = h?.id ?? null
    if (h) { hoveredFields.set(fieldsOf(h)); ownHover = true }
    else if (ownHover) { hoveredFields.set([]); ownHover = false }
  }

  // ── Floating label ───────────────────────────────────────────────────────
  const SUB = { 1: '₁', 2: '₂' }
  function fieldLabel(field, sym) {
    if (field === 'apDb') return 'Ap'
    if (field === 'aaDb') return 'Aa'
    if (field === 'f0') return `${sym}₀`
    if (field === 'bwp') return 'BWp'
    if (field === 'bwa') return 'BWa'
    const m = field.match(/^f([pa])(\d)?$/)
    return m ? `${sym}${m[1]}${SUB[m[2]] ?? ''}` : field
  }
  function labelText(h, f) {
    const sym = $dataUnit === 'rad' ? 'ω' : 'f'
    const unit = $dataUnit === 'rad' ? 'rad/s' : 'Hz'
    const parts = []
    if (h.xf) parts.push(`${fieldLabel(h.xf, sym)} = ${formatSI(f[h.xf])} ${unit}`)
    if (h.yf) parts.push(`${fieldLabel(h.yf, sym)} = ${f[h.yf].toFixed(2)} dB`)
    return parts.join(' · ')
  }

  // ── Pointer handling ─────────────────────────────────────────────────────
  let gd = null
  let swallowMouse = false
  let moveFrame = null, lastMove = null

  function rel(e) {
    const r = gd.getBoundingClientRect()
    return [e.clientX - r.left, e.clientY - r.top]
  }

  function onHoverMove(e) {
    if (drag || !showTemplate || e.buttons) return
    const [px, py] = rel(e)
    const h = hitTest(gd, px, py)
    if ((h?.id ?? null) !== hoverId) setHover(h)
    setCursor(gd, h)
  }

  function onLeave() {
    if (drag) return
    setHover(null)
    setCursor(gd, null)
  }

  function onDown(e) {
    if (!showTemplate || e.button !== 0 || drag) return
    const [px, py] = rel(e)
    const h = hitTest(gd, px, py)
    if (!h) return
    // Ours: keep Plotly from starting a zoom / pan.
    e.stopPropagation()
    e.preventDefault()
    swallowMouse = true
    const { xa, ya } = axesOf(gd)
    const hx = h.x != null ? xPx(xa, h.x * axis.scale) : px
    const hy = h.y != null ? yPx(ya, h.y) : py
    drag = { h, gx: px - hx, gy: py - hy, startForm: { ...$designForm }, moved: false }
    dragging = true
    setHover(h)
    setCursor(gd, h)
    window.addEventListener('pointermove', onDragMove)
    window.addEventListener('pointerup', onDragEnd)
    window.addEventListener('pointercancel', onDragCancel)
    window.addEventListener('keydown', onKey)
  }

  // Plotly listens to mousedown / touchstart; swallow the ones that follow our pointerdown.
  function onMouseDown(e) {
    if (!swallowMouse) return
    swallowMouse = false
    e.stopPropagation()
    e.preventDefault()
  }

  function onDragMove(e) {
    if (!drag) return
    lastMove = rel(e)
    if (moveFrame != null) return
    moveFrame = requestAnimationFrame(applyMove)
  }

  function applyMove() {
    moveFrame = null
    if (!drag || !lastMove) return
    const ax = axesOf(gd)
    if (!ax) return
    const [px, py] = lastMove
    const tx = px - drag.gx, ty = py - drag.gy
    const xPlot = ax.xa.l2d(ax.xa.p2l(tx - ax.xa._offset))
    const yDb   = ax.ya.l2d(ax.ya.p2l(ty - ax.ya._offset))
    const next = dragTo(drag.startForm, drag.h, xPlot / axis.scale, yDb, uf)
    drag.moved = true
    designForm.set(next)
    label = { x: px + 14, y: py - 30, text: labelText(drag.h, next) }
  }

  function endDrag() {
    window.removeEventListener('pointermove', onDragMove)
    window.removeEventListener('pointerup', onDragEnd)
    window.removeEventListener('pointercancel', onDragCancel)
    window.removeEventListener('keydown', onKey)
    if (moveFrame != null) { cancelAnimationFrame(moveFrame); moveFrame = null }
    const d = drag
    drag = null
    dragging = false
    label = null
    lastMove = null
    return d
  }

  function onDragEnd() {
    if (moveFrame != null) applyMove()
    const d = endDrag()
    // Q3: releasing a template drag re-designs.
    if (d?.moved) runDesign()
  }

  function onDragCancel() {
    const d = endDrag()
    if (d) designForm.set(d.startForm)
  }

  function onKey(e) {
    if (e.key === 'Escape') { e.preventDefault(); onDragCancel() }
  }

  onMount(() => {
    gd = plot.plotElement()
    gd.addEventListener('pointerdown', onDown, true)
    gd.addEventListener('mousedown', onMouseDown, true)
    gd.addEventListener('touchstart', onMouseDown, true)
    gd.addEventListener('pointermove', onHoverMove)
    gd.addEventListener('pointerleave', onLeave)
  })

  onDestroy(() => {
    endDrag()
    if (ownHover) hoveredFields.set([])
    gd?.removeEventListener('pointerdown', onDown, true)
    gd?.removeEventListener('mousedown', onMouseDown, true)
    gd?.removeEventListener('touchstart', onMouseDown, true)
    gd?.removeEventListener('pointermove', onHoverMove)
    gd?.removeEventListener('pointerleave', onLeave)
  })

  // ── Compliance chips ─────────────────────────────────────────────────────
  function fmtMargin(c) {
    const m = c.margin
    const s = `${m >= 0 ? '+' : '−'}${Math.abs(m).toFixed(2)} dB`
    return c.ok ? s : `${s} @ ${formatSI(c.at * uf)} ${$dataUnit === 'rad' ? 'rad/s' : 'Hz'}`
  }

  $: yLabel = $plotUnit === 'rad' ? '$|H(\\omega)|$ [dB]' : '$|H(f)|$ [dB]'
  $: uirevision = showTemplate ? `tpl-${$designForm.filterType}-${$plotUnit}` : undefined
</script>

<BodePlot
  bind:this={plot}
  {traces}
  {shapes}
  {yRange}
  {xRange}
  {uirevision}
  {yLabel}
  xLabel={axis.xLabel}
  logX={true}
  filename={showTemplate ? 'filtool_template' : 'filtool_magnitude'}
  active={$activeTab === tabId}
>
  {#if showTemplate && geom}
    <div class="tpl-overlay">
      {#if $bodeData && comp?.pass}
        <span class="chip" class:bad={!comp.pass.ok} title="Worst-case margin above the passband limit">
          Pass {comp.pass.ok ? '✓' : '✗'} <b>{fmtMargin(comp.pass)}</b>
        </span>
      {/if}
      {#if $bodeData && comp?.stop}
        <span class="chip" class:bad={!comp.stop.ok} title="Worst-case margin below the stopband limit">
          Stop {comp.stop.ok ? '✓' : '✗'} <b>{fmtMargin(comp.stop)}</b>
        </span>
      {/if}
      {#if !$filterParams}
        <span class="chip muted">Template preview · press Design</span>
      {:else if stale && !dragging}
        <button class="chip action" disabled={$designBusy} on:click={runDesign}>
          {$designBusy ? 'Designing…' : 'Out of date · Redesign'}
        </button>
      {/if}
    </div>
    {#if label}
      <div class="drag-label" style="left: {label.x}px; top: {label.y}px">{label.text}</div>
    {/if}
  {/if}
</BodePlot>

<style>
  .tpl-overlay {
    position: absolute;
    top: 42px;
    left: 72px;
    display: flex;
    flex-wrap: wrap;
    gap: 0.3rem;
    pointer-events: none;
    z-index: 5;
    max-width: calc(100% - 240px);
  }
  .chip {
    font-size: 0.74rem;
    line-height: 1.2;
    padding: 0.18rem 0.5rem;
    border-radius: 999px;
    border: 1px solid color-mix(in srgb, var(--success) 45%, var(--border));
    background: color-mix(in srgb, var(--success) 12%, var(--surface));
    color: var(--text);
    white-space: nowrap;
  }
  .chip b { font-weight: 600; font-family: ui-monospace, 'SF Mono', Consolas, monospace; }
  .chip.bad {
    border-color: color-mix(in srgb, var(--danger) 55%, var(--border));
    background: color-mix(in srgb, var(--danger) 14%, var(--surface));
  }
  .chip.muted {
    border-color: var(--border);
    background: var(--surface);
    color: var(--text-dim);
  }
  .chip.action {
    pointer-events: auto;
    cursor: pointer;
    font: inherit;
    font-size: 0.74rem;
    border-color: var(--accent);
    background: color-mix(in srgb, var(--accent) 16%, var(--surface));
  }
  .chip.action:hover:not(:disabled) { background: color-mix(in srgb, var(--accent) 26%, var(--surface)); }
  .chip.action:disabled { opacity: 0.6; cursor: default; }

  .drag-label {
    position: absolute;
    pointer-events: none;
    z-index: 6;
    font-size: 0.76rem;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    padding: 0.2rem 0.45rem;
    border-radius: 4px;
    background: var(--surface);
    border: 1px solid var(--border);
    color: var(--text);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25);
    white-space: nowrap;
  }

  /* Handle cursors override Plotly's drag-layer cursor */
  :global(.js-plotly-plot[data-tpl-cursor='ew'] .nsewdrag),
  :global(.js-plotly-plot[data-tpl-cursor='ew'] .drag) { cursor: ew-resize !important; }
  :global(.js-plotly-plot[data-tpl-cursor='ns'] .nsewdrag),
  :global(.js-plotly-plot[data-tpl-cursor='ns'] .drag) { cursor: ns-resize !important; }
  :global(.js-plotly-plot[data-tpl-cursor='move'] .nsewdrag),
  :global(.js-plotly-plot[data-tpl-cursor='move'] .drag) { cursor: move !important; }
</style>
