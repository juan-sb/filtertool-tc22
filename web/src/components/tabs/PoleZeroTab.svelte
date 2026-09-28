<script>
  import { onMount, onDestroy } from 'svelte'
  import Plotly from 'plotly.js-dist'
  import { filterResult, filterParams, stages, remainingPZ, comparisons, theme, colorMode, colorShuffle, showLegend, activeTab, plotUnit, dataUnit, plotCursor } from '../../stores/app.js'
  import { getWorkerApi } from '../../lib/worker-client.js'
  import { APPROX_NAMES, plotColor, sPlaneAxis } from '../../lib/approx.js'
  import { isComplexRoot, rootValue } from '../../lib/roots.js'

  let container
  let plotMounted = false
  let destroyed = false
  let resizeObserver
  let wasActive = false

  // Selection / hover state, by root id (lib/roots.js): repeated roots such as
  // the band-pass zeros at s = 0 stay individually selectable.
  let selectedIds = new Set()
  let hoveredId = null   // id of the list row currently under the mouse

  // Reset selection whenever a new filter is designed
  $: $filterResult, selectedIds = new Set(), hoveredId = null

  // Engine poles/zeros are rad/s: the plot follows plotUnit, the list dataUnit.
  $: axis     = sPlaneAxis($plotUnit)
  $: listAxis = sPlaneAxis($dataUnit)

  // Format a complex number for display, scaled into the target unit
  function fmtComplex({ re: r, im: i }, k = 1) {
    const rr = (r * k).toFixed(4)
    if (Math.abs(i) < 1e-9) return rr
    const sign = i >= 0 ? '+' : '−'
    return `${rr} ${sign} j${Math.abs(i * k).toFixed(4)}`
  }

  // A complex root toggles together with its conjugate; real roots one at a time.
  function toggleRoot(root) {
    const s = new Set(selectedIds)
    const on = !s.has(root.id)
    for (const id of [root.id, root.conj]) {
      if (id === null) continue
      if (on) s.add(id); else s.delete(id)
    }
    selectedIds = s
  }

  $: selectedZeros = ($remainingPZ.zeros ?? []).filter(z => selectedIds.has(z.id))
  $: selectedPoles = ($remainingPZ.poles ?? []).filter(p => selectedIds.has(p.id))

  // Mirrors the engine's build_stage() limits.
  $: selectionError = (() => {
    if (selectedIds.size === 0) return ''
    if (selectedPoles.length === 0) return 'Select at least one pole.'
    if (selectedPoles.length > 2) return `A stage takes one or two poles (${selectedPoles.length} selected).`
    if (selectedZeros.length > selectedPoles.length)
      return `More zeros (${selectedZeros.length}) than poles (${selectedPoles.length}): the stage would be improper.`
    return ''
  })()
  $: selectionValid = selectedPoles.length > 0 && !selectionError

  const NORM_OPTIONS = ['Passband', 'ω→0', 'ω→∞', 'ω→ω0']
  let normtype = 'Passband'
  let adding = false
  let addError = ''

  async function addStage() {
    adding = true
    addError = ''
    try {
      const api = getWorkerApi()
      const zeros = selectedZeros.map(rootValue), poles = selectedPoles.map(rootValue)
      const result = await api.buildStageFromZPK(zeros, poles, 1, normtype, $filterParams?.filter_type ?? 0)
      if (result.error) { addError = result.error; return }
      const id = Date.now()
      const name = `Stage ${($stages.length ?? 0) + 1}`
      stages.update(s => [...s, {
        id, name, normtype,
        zeroIds: selectedZeros.map(r => r.id), poleIds: selectedPoles.map(r => r.id),
        zeros, poles,
        gain: result.gain, num: result.num, den: result.den,
      }])
      selectedIds = new Set()
    } catch (e) {
      addError = e.message
    } finally {
      adding = false
    }
  }

  // ── Plotly ─────────────────────────────────────────────────────────────────
  $: C = $theme === 'light'
    ? { hi: '#1f2328', used: '#8c959f', unit: '#d0d7de', grid: '#d8dee4', bg: '#f6f8fa', axis: '#afb8c1', zero: '#afb8c1' }
    : { hi: '#e6edf3', used: '#484f58', unit: '#30363d', grid: '#21262d', bg: '#0d1117', axis: '#484f58', zero: '#52565c' }

  $: mainColor = plotColor($filterParams?.approx_type ?? 0, $theme, $colorMode, $colorShuffle)

  function buildTraces(fr, remaining, selIds, hovId, mainCol, compList, k) {
    if (!fr?.roots) return []
    const avail = new Set([...(remaining.zeros ?? []), ...(remaining.poles ?? [])].map(r => r.id))
    const θ = Array.from({ length: 361 }, (_, i) => i * Math.PI / 180)
    // Unit circle (|s| = 1 rad/s) only — Re/Im axes come from Plotly zerolines
    // (avoids double-thick axes).
    const out = [
      { x: θ.map(t => Math.cos(t) * k), y: θ.map(t => Math.sin(t) * k),
        mode: 'lines', line: { color: C.unit, width: 1, dash: 'dot' },
        hoverinfo: 'skip', showlegend: false },
    ]

    // Hover set: hovered root + its conjugate (so both of a complex pair light up)
    const hoverSet = new Set()
    if (hovId) {
      hoverSet.add(hovId)
      const hovRoot = [...fr.roots.zeros, ...fr.roots.poles].find(r => r.id === hovId)
      if (hovRoot && isComplexRoot(hovRoot)) hoverSet.add(hovRoot.conj)
    }

    // Partition each group into: normal / hoverOnly / selOnly / selHover
    function partition(pts) {
      const normal = [], hoverOnly = [], selOnly = [], selHover = []
      for (const pt of pts) {
        const h = hoverSet.has(pt.id), s = selIds.has(pt.id)
        if (s && h)       selHover.push(pt)
        else if (s)       selOnly.push(pt)
        else if (h)       hoverOnly.push(pt)
        else              normal.push(pt)
      }
      return { normal, hoverOnly, selOnly, selHover }
    }

    const usedPoles = fr.roots.poles.filter(p => !avail.has(p.id))
    const usedZeros = fr.roots.zeros.filter(z => !avail.has(z.id))
    const { normal: nP, hoverOnly: hoP, selOnly: soP, selHover: shP } = partition(remaining.poles ?? [])
    const { normal: nZ, hoverOnly: hoZ, selOnly: soZ, selHover: shZ } = partition(remaining.zeros ?? [])

    // Comparison filters drawn first (behind main filter)
    for (const comp of (compList ?? [])) {
      const cc = plotColor(comp.approxType, $theme, $colorMode, $colorShuffle)
      const cn = APPROX_NAMES[comp.approxType]
      const asRoots = list => list.map(([re, im]) => ({ re, im }))
      if (comp.filterResult.poles.length) out.push(mkX(asRoots(comp.filterResult.poles), cc, 7, `${cn} poles`, k))
      if (comp.filterResult.zeros.length) out.push(mkO(asRoots(comp.filterResult.zeros), cc, 7, `${cn} zeros`, k))
    }

    // Main filter
    if (usedPoles.length) out.push(mkX(usedPoles, C.used,  8,  'Used poles', k))
    if (usedZeros.length) out.push(mkO(usedZeros, C.used,  8,  'Used zeros', k))
    if (nP.length)        out.push(mkX(nP,        mainCol, 10, 'Poles', k))
    if (nZ.length)        out.push(mkO(nZ,        mainCol, 10, 'Zeros', k))
    if (hoP.length)       out.push(mkX(hoP,       C.hi,    12, 'Poles (hover)', k))
    if (hoZ.length)       out.push(mkO(hoZ,       C.hi,    12, 'Zeros (hover)', k))
    if (soP.length)       out.push(mkX(soP,       C.hi,    14, 'Selected poles', k))
    if (soZ.length)       out.push(mkO(soZ,       C.hi,    14, 'Selected zeros', k))
    if (shP.length)       out.push(mkX(shP,       C.hi,    16, 'Selected poles (hover)', k))
    if (shZ.length)       out.push(mkO(shZ,       C.hi,    16, 'Selected zeros (hover)', k))
    return out
  }

  function mkX(pts, color, size, name, k) {
    return {
      x: pts.map(p => p.re * k), y: pts.map(p => p.im * k),
      mode: 'markers', name,
      marker: { symbol: 'x', size, color, line: { width: 2, color } },
      hovertemplate: pts.map(p => `${fmtComplex(p, k)}<extra>${name}</extra>`),
    }
  }

  function mkO(pts, color, size, name, k) {
    return {
      x: pts.map(p => p.re * k), y: pts.map(p => p.im * k),
      mode: 'markers', name,
      marker: { symbol: 'circle-open', size, color, line: { width: 2 } },
      hovertemplate: pts.map(p => `${fmtComplex(p, k)}<extra>${name}</extra>`),
    }
  }

  const mkLayout = () => {
    const text = $theme === 'light' ? '#1f2328' : '#e6edf3'
    const baseFont = { color: text, size: 12, family: 'system-ui, sans-serif' }
    const tickFont = { color: text, size: 11, family: 'system-ui, sans-serif' }
    return {
    paper_bgcolor: C.bg, plot_bgcolor: C.bg,
    font: baseFont,
    showlegend: $showLegend,
    margin: { t: 36, b: 56, l: 64, r: 24 },
    legend: {
      bgcolor:     $theme === 'light' ? '#ffffff' : '#161b22',
      bordercolor: $theme === 'light' ? '#d0d7de' : '#30363d',
      borderwidth: 1,
      font:        { size: 11, family: 'system-ui, sans-serif' },
      x: 1, xanchor: 'right',
      y: 0.98, yanchor: 'top',
      tracegroupgap: 4,
    },
    hovermode: $plotCursor ? 'closest' : false,
    xaxis: {
      title: { text: axis.xLabel, standoff: 8, font: baseFont },
      gridcolor: C.grid,
      linecolor: C.axis,
      tickcolor: C.axis,
      tickfont: tickFont,
      zeroline: true,
      zerolinecolor: C.zero,
      zerolinewidth: 1.5,
      scaleanchor: 'y', scaleratio: 1,
    },
    yaxis: {
      title: { text: axis.yLabel, standoff: 8, font: baseFont },
      gridcolor: C.grid,
      linecolor: C.axis,
      tickcolor: C.axis,
      tickfont: tickFont,
      zeroline: true,
      zerolinecolor: C.zero,
      zerolinewidth: 1.5,
    },
    modebar: {
      color:       $theme === 'light' ? '#57606a' : '#7d8590',
      activecolor: $theme === 'light' ? '#0969da' : '#58a6ff',
      bgcolor:     $theme === 'light' ? 'rgba(255,255,255,0.85)' : 'rgba(22,27,34,0.85)',
    },
  }
  }

  const cfg = { responsive: true, displaylogo: false,
    toImageButtonOptions: { format: 'svg', filename: 'filtool_pz' } }

  function refreshTitles() {
    if (!plotMounted || destroyed || !container) return
    Plotly.react(
      container,
      buildTraces($filterResult, $remainingPZ, selectedIds, hoveredId, mainColor, $comparisons, axis.scale),
      mkLayout(),
      cfg,
    )
    Plotly.Plots.resize(container)
  }

  $: if (plotMounted && $activeTab === 'poleZero' && !wasActive) {
    wasActive = true
    requestAnimationFrame(() => requestAnimationFrame(refreshTitles))
  } else if ($activeTab !== 'poleZero') {
    wasActive = false
  }

  function mountPlot() {
    if (!container || destroyed) return
    Plotly.newPlot(container, buildTraces($filterResult, $remainingPZ, selectedIds, hoveredId, mainColor, $comparisons, axis.scale), mkLayout(), cfg)
    plotMounted = true
    wasActive = $activeTab === 'poleZero'
    resizeObserver = new ResizeObserver(() => {
      if (plotMounted && !destroyed && container) Plotly.Plots.resize(container)
    })
    resizeObserver.observe(container)
    // Layout may not be final on first paint (esp. when tab was visibility-hidden).
    requestAnimationFrame(() => {
      if (plotMounted && !destroyed && container) {
        Plotly.Plots.resize(container)
        if ($activeTab === 'poleZero') refreshTitles()
      }
    })
  }

  function updatePlot() {
    if (!plotMounted || destroyed || !container) return
    Plotly.react(container, buildTraces($filterResult, $remainingPZ, selectedIds, hoveredId, mainColor, $comparisons, axis.scale), mkLayout(), cfg)
  }

  $: updatePlot(), [$filterResult, $remainingPZ, selectedIds, hoveredId, mainColor, $comparisons, $theme, $colorMode, $colorShuffle, $showLegend, axis, $plotCursor]

  onMount(mountPlot)
  onDestroy(() => {
    destroyed = true
    plotMounted = false
    resizeObserver?.disconnect()
    if (container) Plotly.purge(container)
  })
</script>

<div class="pz-tab">
  <!-- Plot -->
  <div class="plot-wrap" bind:this={container}></div>

  <!-- Selection panel -->
  <div class="panel">
    {#if !$filterResult}
      <p class="hint">Design a filter to see its poles and zeros.</p>
    {:else}
      <div class="sec">Poles [{listAxis.unit}]</div>

      {#if ($remainingPZ.poles ?? []).length === 0}
        <p class="hint-sm">All poles assigned.</p>
      {:else}
        {#each ($remainingPZ.poles ?? []) as p (p.id)}
          <label class="pz-row" class:sel={selectedIds.has(p.id)} class:hov={hoveredId === p.id}
            on:mouseenter={() => hoveredId = p.id} on:mouseleave={() => hoveredId = null}>
            <input type="checkbox" checked={selectedIds.has(p.id)} on:change={() => toggleRoot(p)} />
            <span class="val" style="color: {mainColor}">{fmtComplex(p, listAxis.scale)}</span>
          </label>
        {/each}
      {/if}

      {#if ($filterResult.zeros ?? []).length > 0}
        <div class="sec mt">Zeros [{listAxis.unit}]</div>
        {#if ($remainingPZ.zeros ?? []).length === 0}
          <p class="hint-sm">All zeros assigned.</p>
        {:else}
          {#each ($remainingPZ.zeros ?? []) as z (z.id)}
            <label class="pz-row" class:sel={selectedIds.has(z.id)} class:hov={hoveredId === z.id}
              on:mouseenter={() => hoveredId = z.id} on:mouseleave={() => hoveredId = null}>
              <input type="checkbox" checked={selectedIds.has(z.id)} on:change={() => toggleRoot(z)} />
              <span class="val" style="color: {mainColor}">{fmtComplex(z, listAxis.scale)}</span>
            </label>
          {/each}
        {/if}
      {/if}

      <div class="div"></div>

      {#if selectionError}
        <p class="warn">{selectionError}</p>
      {/if}
      {#if addError}<p class="err">{addError}</p>{/if}

      <div class="norm-row">
        <span class="norm-lbl">Norm.</span>
        <select class="norm-sel" bind:value={normtype}>
          {#each NORM_OPTIONS as n}<option value={n}>{n}</option>{/each}
        </select>
      </div>

      <button class="add-btn" disabled={!selectionValid || adding} on:click={addStage}>
        {adding ? 'Adding…' : 'Add Stage'}
      </button>

      {#if $stages.length > 0}
        <div class="div"></div>
        <div class="sec">Stages ({$stages.length})</div>
        {#each $stages as stage (stage.id)}
          <div class="stage-row">
            <span class="sname">{stage.name}</span>
            <span class="sdet">{stage.poles.length}P/{stage.zeros.length}Z</span>
            <button class="rm" on:click={() => stages.update(s => s.filter(st => st.id !== stage.id))}>×</button>
          </div>
        {/each}
      {/if}
    {/if}
  </div>
</div>

<style>
  .pz-tab {
    display: flex;
    height: 100%;
    min-height: 0;
    min-width: 0;
    overflow: hidden;
  }

  .plot-wrap { flex: 1; min-width: 0; min-height: 0; width: 100%; height: 100%; }

  .panel {
    width: min(260px, 42vw);
    flex-shrink: 0;
    background: var(--surface);
    border-left: 1px solid var(--surface-2);
    overflow-x: auto;
    overflow-y: auto;
    padding: 0.5rem 0.45rem;
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    min-width: 0;
  }

  .sec {
    font-size: 0.68rem;
    color: var(--text-dim);
    text-transform: uppercase;
    letter-spacing: 0.05em;
  }
  .sec.mt { margin-top: 0.45rem; }

  .pz-row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.2rem 0.25rem;
    border-radius: 3px;
    cursor: pointer;
    user-select: none;
    min-width: 0;
  }
  .pz-row:hover { background: var(--surface-2); }
  .pz-row.hov { background: var(--success-bg); }
  .pz-row.sel { background: var(--selected); }
  .pz-row.sel.hov { background: var(--selected); }

  .pz-row input[type=checkbox] {
    accent-color: var(--accent);
    width: 13px; height: 13px;
    flex-shrink: 0; cursor: pointer;
  }

  .val {
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    font-size: 0.74rem;
    white-space: nowrap;
    overflow: visible;
  }

  .div { height: 1px; background: var(--surface-2); margin: 0.3rem 0; }

  .hint {
    font-size: 0.8rem;
    color: var(--text-dim);
    text-align: left;
    padding: 0.75rem 0.35rem;
    overflow-wrap: anywhere;
  }
  .hint-sm { font-size: 0.7rem; color: var(--disabled); margin: 0; overflow-wrap: anywhere; }

  .warn {
    font-size: 0.72rem; color: var(--warning);
    background: var(--warning-bg); border-radius: 3px;
    padding: 0.28rem 0.4rem; margin: 0;
    overflow-wrap: anywhere;
  }
  .err {
    font-size: 0.72rem; color: var(--danger);
    background: var(--danger-bg); border-radius: 3px;
    padding: 0.28rem 0.4rem; margin: 0;
    overflow-wrap: anywhere;
  }

  .norm-row {
    display: flex;
    align-items: center;
    gap: 0.35rem;
    margin-top: 0.1rem;
    min-width: 0;
  }
  .norm-lbl { font-size: 0.68rem; color: var(--text-dim); white-space: nowrap; }
  .norm-sel {
    flex: 1;
    min-width: 0;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 3px;
    color: var(--text);
    font-size: 0.75rem;
    padding: 0.22rem 0.3rem;
    outline: none;
  }
  .norm-sel:focus { border-color: var(--accent); }

  .add-btn {
    background: var(--accent-strong); border: none; border-radius: 4px;
    color: #fff; cursor: pointer;
    font-size: 0.8rem; font-weight: 600;
    padding: 0.38rem; width: 100%;
  }
  .add-btn:hover:not(:disabled) { background: var(--accent-hover); }
  .add-btn:disabled { background: var(--surface-2); color: var(--disabled); cursor: default; }

  .stage-row {
    display: flex; align-items: center; gap: 0.3rem;
    padding: 0.2rem 0.3rem; border-radius: 3px;
    background: var(--surface-2);
    min-width: 0;
  }
  .sname { font-size: 0.75rem; flex: 1; min-width: 0; overflow-wrap: anywhere; }
  .sdet { font-size: 0.68rem; color: var(--text-dim); flex-shrink: 0; }
  .rm {
    background: none; border: none; color: var(--text-dim);
    cursor: pointer; font-size: 0.85rem; line-height: 1;
    padding: 0 0.1rem; border-radius: 3px;
  }
  .rm:hover { color: var(--danger); background: var(--danger-bg); }

  @media (max-width: 720px) {
    .pz-tab { flex-direction: column; }
    .panel {
      width: 100%;
      max-height: 38vh;
      border-left: none;
      border-top: 1px solid var(--surface-2);
    }
  }
</style>
