<script>
  import { onMount, onDestroy } from 'svelte'
  import {
    stages, filterParams, filterResult, bodeData, bodePoints, theme, activeTab, plotUnit, dataUnit,
    remainingPZ, hoveredStageId,
  } from '../../stores/app.js'
  import { getWorkerApi } from '../../lib/worker-client.js'
  import { freqAxis, freqRangeFromParams, sPlaneAxis, TWO_PI } from '../../lib/approx.js'
  import { stageColor } from '../../lib/stage-colors.js'
  import { normOmega, resolveNorm, scaleRoots, poleSummary } from '../../lib/stage-math.js'
  import { tfAbs, toDb } from '../../lib/poly.js'
  import {
    updateStage, resetAllStages, isModified, rootsModified, rootRef, parseRootRef, dragStageRoot, wheelStageQ,
  } from '../../lib/stages.js'
  import { formatSI } from '../../lib/si.js'
  import BodePlot from '../BodePlot.svelte'
  import PzMap from '../PzMap.svelte'
  import StageCard from '../StageCard.svelte'

  $: ft     = $filterParams?.filter_type ?? 0
  $: axis   = freqAxis($plotUnit)
  $: sAxis  = sPlaneAxis($plotUnit)
  $: active = $activeTab === 'stages'

  // ── Stage Bode curves: recompute only stages whose num / den changed ──────
  $: freqRange = freqRangeFromParams($filterParams)
  $: points    = Math.min(Number($bodePoints) || 2000, 5000)
  let bodes = new Map()          // stage id → { num, den, bode }
  const pendingBode = new Map()  // stage id → token
  let rangeKey = ''

  $: {
    const key = `${freqRange.min}|${freqRange.max}|${points}`
    if (key !== rangeKey) { rangeKey = key; bodes = new Map() }
    syncBodes($stages)
  }

  function syncBodes(list) {
    const ids = new Set(list.map(s => s.id))
    for (const id of [...bodes.keys()]) if (!ids.has(id)) bodes.delete(id)
    for (const s of list) {
      const c = bodes.get(s.id)
      if (!s.num || (c && c.num === s.num && c.den === s.den)) continue
      const token = {}
      pendingBode.set(s.id, token)
      const { num, den } = s, range = freqRange, pts = points
      getWorkerApi().computeBode(num, den, range.min, range.max, pts).then(bode => {
        if (pendingBode.get(s.id) !== token) return
        pendingBode.delete(s.id)
        bodes.set(s.id, { num, den, bode })
        bodes = bodes
      }).catch(() => pendingBode.delete(s.id))
    }
  }

  const dbArr = b => b.magnitude.map(m => { const d = toDb(m); return Number.isFinite(d) ? d : null })

  // ── Traces ────────────────────────────────────────────────────────────────
  $: hovered = $hoveredStageId
  $: traces = buildTraces($stages, bodes, $bodeData, axis, $theme, hovered)

  function buildTraces(list, bmap, designed, ax, th, hov) {
    const out = []
    const light = th === 'light'
    if (designed) {
      out.push({
        x: designed.freq.map(f => f * ax.scale), y: dbArr(designed), mode: 'lines', name: 'Designed filter',
        line: { color: light ? '#8c959f' : '#6e7681', width: 4 }, opacity: 0.35, hoverinfo: 'skip',
      })
    }
    list.forEach((s, i) => {
      const b = bmap.get(s.id)?.bode
      if (!b) return
      const on = hov === s.id, dim = hov != null && !on
      out.push({
        x: b.freq.map(f => f * ax.scale), y: dbArr(b), mode: 'lines', name: s.name,
        line: { color: stageColor(i, th), width: on ? 3 : 1.6 }, opacity: dim ? 0.3 : 1,
      })
    })
    // Cascade = point-wise sum of stage dB, once every stage has a curve
    const bs = list.map(s => bmap.get(s.id)?.bode)
    if (bs.length && bs.every(Boolean) && bs.every(b => b.freq.length === bs[0].freq.length)) {
      const y = bs[0].freq.map((_, k) => {
        let sum = 0
        for (const b of bs) { const d = toDb(b.magnitude[k]); if (!Number.isFinite(d)) return null; sum += d }
        return sum
      })
      out.push({
        x: bs[0].freq.map(f => f * ax.scale), y, mode: 'lines', name: 'Cascade',
        line: { color: light ? '#24292f' : '#e6edf3', width: 2, dash: 'dot' },
      })
    }
    // 0 dB reference marker per stage: where its normalization sets unity (plus the gain offset)
    list.forEach((s, i) => {
      const b = bmap.get(s.id)?.bode
      if (!b || !s.num) return
      const w = normOmega(s, ft)
      if (w == null) return
      const fHz = w === 0 ? b.freq[0] : w === Infinity ? b.freq[b.freq.length - 1] : w / TWO_PI
      const y = toDb(tfAbs(s.num, s.den, w === 0 ? 0 : w === Infinity ? Infinity : w))
      if (!Number.isFinite(y)) return
      out.push({
        x: [fHz * ax.scale], y: [y], mode: 'markers', showlegend: false, hoverinfo: 'text',
        text: [`${s.name}: normalization point (${(s.gainDb ?? 0).toFixed(2)} dB)`],
        marker: { symbol: 'diamond', size: hov === s.id ? 11 : 8, color: stageColor(i, th), line: { width: 1, color: light ? '#ffffff' : '#0d1117' } },
        opacity: hov != null && hov !== s.id ? 0.3 : 1,
      })
    })
    return out
  }

  // ── Cascade vs target at the passband reference ───────────────────────────
  $: refW = (() => {
    const n = resolveNorm('Passband', ft)
    if (n === 'ω→0') return 0
    if (n === 'ω→∞') return Infinity
    const wp = $filterParams?.wp
    return Array.isArray(wp) ? Math.sqrt(wp[0] * wp[1]) : null
  })()
  $: refText = refW === 0 ? 'DC' : refW === Infinity ? 'HF' : refW != null ? `${formatSI((refW / TWO_PI) * ($dataUnit === 'rad' ? TWO_PI : 1))} ${$dataUnit === 'rad' ? 'rad/s' : 'Hz'}` : ''
  $: readout = (() => {
    if (!$stages.length || refW == null || !$filterResult) return null
    let cas = 1
    for (const s of $stages) { if (!s.num) return null; cas *= tfAbs(s.num, s.den, refW) }
    const tgt = tfAbs($filterResult.num, $filterResult.den, refW)
    const c = toDb(cas), t = toDb(tgt)
    return { c, t, d: Number.isFinite(c) && Number.isFinite(t) ? t - c : null }
  })()
  $: unassigned = ($remainingPZ.poles?.length ?? 0) + ($remainingPZ.zeros?.length ?? 0)
  $: anyEdited = $stages.some(isModified)

  const fmtDb = v => (Number.isFinite(v) ? `${v >= 0 ? '+' : '−'}${Math.abs(v).toFixed(2)} dB` : '—')

  // E4: put the cascade-vs-target difference into the last stage's gain offset.
  function absorbGain() {
    const last = $stages[$stages.length - 1]
    if (!last || readout?.d == null) return
    updateStage(last.id, s => ({ gainDb: (s.gainDb ?? 0) + readout.d }))
  }

  // ── PZ mini-map ───────────────────────────────────────────────────────────
  const asRoots = (list, refOf) => list.map(([re, im], i) => ({ re, im, ref: refOf?.(i) }))
  $: mapGroups = (() => {
    const out = []
    const grey = $theme === 'light' ? '#8c959f' : '#6e7681'
    out.push({ roots: ($remainingPZ.poles ?? []).map(r => ({ re: r.re, im: r.im })), symbol: 'x', color: grey, size: 7, opacity: 0.6, name: 'Unassigned poles' })
    out.push({ roots: ($remainingPZ.zeros ?? []).map(r => ({ re: r.re, im: r.im })), symbol: 'circle-open', color: grey, size: 7, opacity: 0.6, name: 'Unassigned zeros' })
    $stages.forEach((s, i) => {
      const col = stageColor(i, $theme)
      const on = hovered === s.id, dim = hovered != null && !on
      if (rootsModified(s)) {
        out.push({ roots: asRoots(s.orig.poles), symbol: 'x', color: col, size: 7, opacity: 0.3, name: `${s.name} (designed)` })
        out.push({ roots: asRoots(s.orig.zeros), symbol: 'circle-open', color: col, size: 7, opacity: 0.3, name: `${s.name} (designed)` })
      }
      out.push({ roots: asRoots(s.poles, i => rootRef(s.id, 'p', i)), symbol: 'x', color: col, size: on ? 13 : 10, opacity: dim ? 0.3 : 1, name: `${s.name} poles` })
      out.push({ roots: asRoots(s.zeros, i => rootRef(s.id, 'z', i)), symbol: 'circle-open', color: col, size: on ? 13 : 10, opacity: dim ? 0.3 : 1, name: `${s.name} zeros` })
    })
    return out
  })()

  function onMapHover(e) {
    const ref = e.detail.ref
    hoveredStageId.set(ref?.startsWith('s:') ? Number(ref.split(':')[1]) : null)
  }
  const isStageRoot = ref => !!parseRootRef(ref)
  const isStagePole = ref => parseRootRef(ref)?.kind === 'p'
  const onMapDrag  = e => dragStageRoot(e.detail.ref, e.detail.re, e.detail.im, e.detail.snapIm)
  const onMapWheel = e => { const r = parseRootRef(e.detail.ref); if (r) wheelStageQ(r.stageId, e.detail.dir) }

  // ── Bode hover: nearest stage curve ───────────────────────────────────────
  let plot, gd
  const HIT = 6

  function stageAt(px, py) {
    const fl = gd?._fullLayout
    if (!fl?.xaxis) return null
    const xa = fl.xaxis, ya = fl.yaxis
    if (px < xa._offset || px > xa._offset + xa._length || py < ya._offset || py > ya._offset + ya._length) return null
    let best = null, bestD = HIT
    for (const s of $stages) {
      const b = bodes.get(s.id)?.bode
      if (!b) continue
      const f = b.freq
      const fAt = p => xa.l2d(xa.p2l(p - xa._offset)) / axis.scale
      let lo = 0, hi = f.length
      const target = fAt(px - 10)
      while (lo < hi) { const m = (lo + hi) >> 1; if (f[m] < target) lo = m + 1; else hi = m }
      for (let i = Math.max(0, lo - 1); i < f.length - 1; i++) {
        const x0 = xa._offset + xa.l2p(xa.d2l(f[i] * axis.scale))
        if (x0 > px + 10) break
        const x1 = xa._offset + xa.l2p(xa.d2l(f[i + 1] * axis.scale))
        const d0 = toDb(b.magnitude[i]), d1 = toDb(b.magnitude[i + 1])
        if (!Number.isFinite(d0) || !Number.isFinite(d1)) continue
        const y0 = ya._offset + ya.l2p(d0), y1 = ya._offset + ya.l2p(d1)
        const dx = x1 - x0, dy = y1 - y0, L2 = dx * dx + dy * dy
        const t = L2 ? Math.max(0, Math.min(1, ((px - x0) * dx + (py - y0) * dy) / L2)) : 0
        const d = Math.hypot(px - (x0 + t * dx), py - (y0 + t * dy))
        if (d < bestD) { bestD = d; best = s }
      }
    }
    return best
  }

  function rel(e) { const r = gd.getBoundingClientRect(); return [e.clientX - r.left, e.clientY - r.top] }

  // ── Drag a stage curve: vertical = gain offset, horizontal = frequency scale
  // (all its poles and zeros × ratio, so shape and Q are kept). Shift locks to
  // the dominant axis; Esc restores.
  let cdrag = null          // { id, zeros, poles, gainDb, x0, y0, f0, db0 }
  let frozenY = null
  let swallow = false
  let dragLabel = null

  function onBodeDown(e) {
    if (e.button !== 0 || cdrag) return
    const [px, py] = rel(e)
    const s = stageAt(px, py)
    if (!s) return
    e.stopPropagation(); e.preventDefault(); swallow = true
    const { xaxis: xa, yaxis: ya } = gd._fullLayout
    frozenY = ya.range.slice()
    cdrag = {
      id: s.id, zeros: s.zeros, poles: s.poles, gainDb: s.gainDb ?? 0, x0: px, y0: py,
      f0: xa.l2d(xa.p2l(px - xa._offset)), db0: ya.p2l(py - ya._offset),
    }
    hoveredStageId.set(s.id)
    window.addEventListener('pointermove', onCurveMove)
    window.addEventListener('pointerup', onCurveUp)
    window.addEventListener('keydown', onCurveKey)
  }
  function onBodeMouseDown(e) {
    if (!swallow) return
    swallow = false
    e.stopPropagation(); e.preventDefault()
  }
  function onCurveMove(e) {
    if (!cdrag) return
    const [px, py] = rel(e)
    const { xaxis: xa, yaxis: ya } = gd._fullLayout
    let dx = px - cdrag.x0, dy = py - cdrag.y0
    if (e.shiftKey) { if (Math.abs(dx) > Math.abs(dy)) dy = 0; else dx = 0 }
    const ratio = xa.l2d(xa.p2l(cdrag.x0 + dx - xa._offset)) / cdrag.f0
    const dDb = ya.p2l(cdrag.y0 + dy - ya._offset) - cdrag.db0
    updateStage(cdrag.id, {
      zeros: scaleRoots(cdrag.zeros, ratio), poles: scaleRoots(cdrag.poles, ratio), gainDb: cdrag.gainDb + dDb,
    })
    const w0 = poleSummary(scaleRoots(cdrag.poles, ratio)).w0
    const uf = $dataUnit === 'rad' ? TWO_PI : 1
    dragLabel = {
      x: px + 14, y: py - 30,
      text: `${w0 ? `${$dataUnit === 'rad' ? 'ω' : 'f'}₀ ${formatSI((w0 / TWO_PI) * uf)} ${$dataUnit === 'rad' ? 'rad/s' : 'Hz'} · ` : ''}gain ${fmtDb(cdrag.gainDb + dDb)}`,
    }
  }
  function endCurveDrag() {
    window.removeEventListener('pointermove', onCurveMove)
    window.removeEventListener('pointerup', onCurveUp)
    window.removeEventListener('keydown', onCurveKey)
    cdrag = null
    frozenY = null
    dragLabel = null
  }
  function onCurveUp() { endCurveDrag() }
  function onCurveKey(e) {
    if (e.key !== 'Escape' || !cdrag) return
    e.preventDefault()
    updateStage(cdrag.id, { zeros: cdrag.zeros, poles: cdrag.poles, gainDb: cdrag.gainDb })
    endCurveDrag()
  }
  // Wheel over a stage curve: its Q (2-pole stages)
  function onBodeWheel(e) {
    const s = stageAt(...rel(e))
    if (!s || s.poles.length !== 2) return
    e.preventDefault(); e.stopPropagation()
    wheelStageQ(s.id, e.deltaY < 0 ? 1 : -1)
  }

  let ownHover = false
  function onBodeMove(e) {
    if (cdrag) return
    if (gd) gd.dataset.stageCursor = stageAt(...rel(e)) ? 'grab' : ''
    if (e.buttons) return
    const s = stageAt(...rel(e))
    if (s) { hoveredStageId.set(s.id); ownHover = true }
    else if (ownHover) { hoveredStageId.set(null); ownHover = false }
  }
  function onBodeLeave() { if (!cdrag && ownHover) { hoveredStageId.set(null); ownHover = false } }

  onMount(() => {
    gd = plot?.plotElement()
    gd?.addEventListener('pointermove', onBodeMove)
    gd?.addEventListener('pointerleave', onBodeLeave)
    gd?.addEventListener('pointerdown', onBodeDown, true)
    gd?.addEventListener('mousedown', onBodeMouseDown, true)
    gd?.addEventListener('touchstart', onBodeMouseDown, true)
    gd?.addEventListener('wheel', onBodeWheel, { capture: true, passive: false })
  })
  onDestroy(() => {
    endCurveDrag()
    gd?.removeEventListener('pointerdown', onBodeDown, true)
    gd?.removeEventListener('mousedown', onBodeMouseDown, true)
    gd?.removeEventListener('touchstart', onBodeMouseDown, true)
    gd?.removeEventListener('wheel', onBodeWheel, { capture: true })
    gd?.removeEventListener('pointermove', onBodeMove)
    gd?.removeEventListener('pointerleave', onBodeLeave)
    if (ownHover) hoveredStageId.set(null)
  })

  $: yLabel = $plotUnit === 'rad' ? '$|H(\\omega)|$ [dB]' : '$|H(f)|$ [dB]'
  $: xRange = [freqRange.min * axis.scale, freqRange.max * axis.scale]
</script>

<div class="stages-tab">
  <div class="bode">
    <BodePlot bind:this={plot} {traces} {yLabel} {xRange} yRange={frozenY} xLabel={axis.xLabel} uirevision={`stages-${$plotUnit}`}
      filename="filtool_stages" {active}>
      {#if dragLabel}
        <div class="drag-label" style="left: {dragLabel.x}px; top: {dragLabel.y}px">{dragLabel.text}</div>
      {/if}
      {#if !$stages.length}
        <div class="empty">
          {#if $filterResult}
            No stages yet: select poles / zeros in the Pole-Zero tab and press Add Stage.
          {:else}
            Design a filter first.
          {/if}
        </div>
      {/if}
    </BodePlot>
  </div>

  <aside class="side">
    <div class="map">
      <PzMap groups={mapGroups} scale={sAxis.scale} compact {active} resetKey={$filterResult}
        filename="filtool_stages_pz" canDrag={isStageRoot} canWheel={isStagePole}
        on:hover={onMapHover} on:drag={onMapDrag} on:wheel={onMapWheel} />
    </div>

    <div class="bar">
      {#if readout}
        <div class="readout" title="Cascade of all stages vs the designed filter, at the passband reference">
          Cascade @ {refText}: <b>{fmtDb(readout.c)}</b> · target <b>{fmtDb(readout.t)}</b>
          {#if readout.d != null}· Δ <b class:off={Math.abs(readout.d) > 0.01}>{fmtDb(readout.d)}</b>{/if}
        </div>
      {/if}
      <div class="actions">
        <button disabled={!readout || readout.d == null || Math.abs(readout.d) < 1e-3} on:click={absorbGain}
          title="Add Δ to the last stage's gain offset">Absorb Δ</button>
        <button disabled={!anyEdited} on:click={resetAllStages} title="Reset every stage to how it was built">Reset all</button>
      </div>
      {#if unassigned}
        <div class="note">{unassigned} root{unassigned === 1 ? '' : 's'} not in a stage yet</div>
      {/if}
    </div>

    <div class="cards">
      {#each $stages as s, i (s.id)}
        <StageCard stage={s} color={stageColor(i, $theme)} filterType={ft} />
      {/each}
    </div>
  </aside>
</div>

<style>
  .stages-tab { display: flex; height: 100%; min-height: 0; overflow: hidden; }
  .bode { flex: 1; min-width: 0; position: relative; }
  .drag-label {
    position: absolute; pointer-events: none; z-index: 6;
    font-size: 0.76rem; font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    padding: 0.2rem 0.45rem; border-radius: 4px;
    background: var(--surface); border: 1px solid var(--border); color: var(--text);
    box-shadow: 0 2px 8px rgba(0, 0, 0, 0.25); white-space: nowrap;
  }
  :global(.js-plotly-plot[data-stage-cursor='grab'] .nsewdrag) { cursor: grab !important; }
  .empty {
    position: absolute; inset: 0; display: flex; align-items: center; justify-content: center;
    padding: 2rem; color: var(--text-dim); font-size: 0.85rem; pointer-events: none; text-align: center;
  }

  .side {
    width: 330px; flex-shrink: 0;
    display: flex; flex-direction: column; min-height: 0;
    background: var(--surface); border-left: 1px solid var(--surface-2);
  }
  .map { height: 250px; flex-shrink: 0; border-bottom: 1px solid var(--surface-2); }

  .bar {
    display: flex; flex-direction: column; gap: 0.35rem;
    padding: 0.5rem 0.55rem; border-bottom: 1px solid var(--surface-2);
  }
  .readout { font-size: 0.76rem; color: var(--text-muted); line-height: 1.4; }
  .readout b { font-family: ui-monospace, 'SF Mono', Consolas, monospace; font-weight: 600; color: var(--text); }
  .readout b.off { color: var(--warning); }
  .actions { display: flex; gap: 0.35rem; }
  .actions button {
    flex: 1;
    background: var(--surface-2); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text-muted); cursor: pointer; font-size: 0.78rem; padding: 0.25rem 0.4rem;
  }
  .actions button:hover:not(:disabled) { background: var(--hover); color: var(--text); }
  .actions button:disabled { opacity: 0.45; cursor: default; }
  .note { font-size: 0.72rem; color: var(--text-dim); }

  .cards {
    flex: 1; min-height: 0; overflow-y: auto;
    display: flex; flex-direction: column; gap: 0.45rem; padding: 0.55rem;
  }
</style>
