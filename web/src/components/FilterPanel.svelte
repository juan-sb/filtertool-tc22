<script>
  import { TWO_PI } from '../lib/approx.js'
  import { runDesign, liveDenorm } from '../lib/design-action.js'
  import {
    LP, HP, BP, BR, GD, F0_BW, FREQS, MAX_ORDER, GD_APPROX,
    isBand as isBandType, buildParams, formFromParams, rescaleForm, validateForm, switchFilterType, paramsClose,
  } from '../lib/params.js'
  import {
    designForm, filterParams, filterResult, uiEnabled, pendingFormHydration, dataUnit,
    designBusy, designError, liveMode,
  } from '../stores/app.js'
  import Segmented  from './form/Segmented.svelte'
  import OrderRange from './form/OrderRange.svelte'
  import NumField   from './form/NumField.svelte'
  import ApproxTiles from './form/ApproxTiles.svelte'

  // ── Constants ─────────────────────────────────────────────────────────────
  // Response-shape glyphs, 24×12 viewBox.
  const TYPE_OPTIONS = [
    { value: LP, label: 'LP', title: 'Low-pass',    glyph: 'M1 3 H11 L17 10 H23' },
    { value: HP, label: 'HP', title: 'High-pass',   glyph: 'M1 10 H7 L13 3 H23' },
    { value: BP, label: 'BP', title: 'Band-pass',   glyph: 'M1 10 H5 L9 3 H15 L19 10 H23' },
    { value: BR, label: 'BR', title: 'Band-reject', glyph: 'M1 3 H6 L10 10 H14 L18 3 H23' },
    { value: GD, label: 'GD', title: 'Group delay', glyph: 'M1 6 H23 M3 2.5 V9.5 M21 2.5 V9.5' },
  ]
  const DEFINE_OPTIONS = [
    { value: F0_BW, label: 'Centre + BW' },
    { value: FREQS, label: 'Band edges' },
  ]

  // ── Units ─────────────────────────────────────────────────────────────────
  // Form frequencies ($designForm) are held in the current data unit (Hz or
  // rad/s); params sent to the engine are always rad/s.
  /** Hz value × uf = value in the current data unit. */
  $: uf     = $dataUnit === 'rad' ? TWO_PI : 1
  $: toRad  = TWO_PI / uf
  $: uLabel = $dataUnit === 'rad' ? 'rad/s' : 'Hz'
  /** Symbol prefix: f for Hz, ω for rad/s. */
  $: fsym   = $dataUnit === 'rad' ? 'ω' : 'f'
  $: fMin   = 1e-3 * uf
  $: fMax   = 1e12 * uf
  $: bwMin  = 1e-6 * uf

  // Rescale the entered values so the physical frequencies survive a unit flip.
  let lastUnit = $dataUnit
  $: if ($dataUnit !== lastUnit) {
    const k = $dataUnit === 'rad' ? TWO_PI : 1 / TWO_PI
    lastUnit = $dataUnit
    designForm.update(f => rescaleForm(f, k))
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  $: ft         = $designForm.filterType
  $: isBand     = isBandType(ft)
  $: isGD       = ft === GD
  $: formErrors = validateForm($designForm)
  $: hasErrors  = Object.keys(formErrors).length > 0
  /** Form differs from the last successful design. */
  $: stale = !!$filterParams && !hasErrors && !paramsClose(buildParams($designForm, toRad), $filterParams)

  function onTypeChange(e) {
    designForm.update(f => {
      const next = switchFilterType(f, e.detail)
      // Group delay only supports Bessel / Gauss.
      if (e.detail === GD && !GD_APPROX.has(next.approxType)) next.approxType = 5
      return next
    })
  }

  function setApprox(i) {
    designForm.update(f => ({ ...f, approxType: i }))
  }

  function onOrderChange(e) {
    designForm.update(f => ({ ...f, nMin: e.detail.lo, nMax: e.detail.hi }))
  }

  // Apply params from Save/Load without re-running Design.
  $: if ($pendingFormHydration) {
    designForm.update(f => formFromParams($pendingFormHydration, toRad, f))
    pendingFormHydration.set(null)
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  // Editing the form clears the last engine error (depends on $designForm only).
  const clearError = () => designError.set('')
  $: clearError($designForm)

  function design() {
    if (!hasErrors) runDesign()
  }

  // ── Live denorm (T3): see liveDenorm in lib/design-action.js ──────────────
  function onDenormInput() {
    if (liveDenorm.start()) liveDenorm.update($designForm.denorm)
  }

  function onDenormRelease() {
    liveDenorm.end()
  }
</script>

<div class="fp">

  <!-- ── Specs ─────────────────────────────────────────────────────────── -->
  <div class="group">Specs</div>

  <Segmented options={TYPE_OPTIONS} value={ft} ariaLabel="Filter type" on:change={onTypeChange} />

  <ApproxTiles
    value={$designForm.approxType}
    allowed={isGD ? GD_APPROX : null}
    disabledTitle="not available for group delay"
    on:change={e => setApprox(e.detail)}
  />
  {#if formErrors.approxType}<p class="hint">{formErrors.approxType}</p>{/if}

  <div class="order-row">
    <span class="lbl">Order</span>
    <span class="order-val">N {$designForm.nMin}–{$designForm.nMax}</span>
  </div>
  <OrderRange
    lo={$designForm.nMin} hi={$designForm.nMax} min={1} max={MAX_ORDER}
    designed={$filterResult?.N ?? null} {stale}
    on:change={onOrderChange}
  />
  {#if formErrors.nMin || formErrors.nMax}<p class="hint">{formErrors.nMin || formErrors.nMax}</p>{/if}

  <!-- ── Template ──────────────────────────────────────────────────────── -->
  <div class="group">Template</div>

  {#if isGD}
    <NumField label="τ₀" bind:value={$designForm.tau0} unit="s" min={1e-12} max={1} error={formErrors.tau0} edge="tau0" group="centre" />
    <NumField label="{fsym} ref" bind:value={$designForm.frg} unit={uLabel} min={fMin} max={fMax} error={formErrors.frg} edge="frg" group="pass" />
    <NumField label="γ" bind:value={$designForm.gamma} unit="%" min={0.01} max={99} log={false} step={0.5} error={formErrors.gamma} edge="gamma" group="pass" />
  {:else}
    {#if !isBand}
      <div class="pair">
        <NumField layout="stack" label="{fsym}p (pass)" bind:value={$designForm.fp} edge="fp" group="pass" unit={uLabel} min={fMin} max={fMax} error={formErrors.fp} />
        <NumField layout="stack" label="{fsym}a (stop)" bind:value={$designForm.fa} edge="fa" group="stop" unit={uLabel} min={fMin} max={fMax} error={formErrors.fa} />
      </div>
    {:else}
      <Segmented options={DEFINE_OPTIONS} bind:value={$designForm.defineWith} ariaLabel="Define band by" />
      {#if $designForm.defineWith === F0_BW}
        <NumField label="{fsym}₀" bind:value={$designForm.f0} edge="f0" group="centre" unit={uLabel} min={fMin} max={fMax} error={formErrors.f0} />
        <NumField label="BWp (pass)" bind:value={$designForm.bwp} edge="bwp" group="pass" unit={uLabel} min={bwMin} max={fMax} error={formErrors.bwp} />
        <NumField label="BWa (stop)" bind:value={$designForm.bwa} edge="bwa" group="stop" unit={uLabel} min={bwMin} max={fMax} error={formErrors.bwa} />
      {:else}
        <!-- Low edge first, in frequency order for the selected type -->
        {#if ft === BP}
          <div class="pair">
            <NumField layout="stack" label="{fsym}a₁ (stop)" bind:value={$designForm.fa1} edge="fa1" group="stop" unit={uLabel} min={fMin} max={fMax} error={formErrors.fa1} />
            <NumField layout="stack" label="{fsym}p₁ (pass)" bind:value={$designForm.fp1} edge="fp1" group="pass" unit={uLabel} min={fMin} max={fMax} error={formErrors.fp1} />
          </div>
          <div class="pair">
            <NumField layout="stack" label="{fsym}p₂ (pass)" bind:value={$designForm.fp2} edge="fp2" group="pass" unit={uLabel} min={fMin} max={fMax} error={formErrors.fp2} />
            <NumField layout="stack" label="{fsym}a₂ (stop)" bind:value={$designForm.fa2} edge="fa2" group="stop" unit={uLabel} min={fMin} max={fMax} error={formErrors.fa2} />
          </div>
        {:else}
          <div class="pair">
            <NumField layout="stack" label="{fsym}p₁ (pass)" bind:value={$designForm.fp1} edge="fp1" group="pass" unit={uLabel} min={fMin} max={fMax} error={formErrors.fp1} />
            <NumField layout="stack" label="{fsym}a₁ (stop)" bind:value={$designForm.fa1} edge="fa1" group="stop" unit={uLabel} min={fMin} max={fMax} error={formErrors.fa1} />
          </div>
          <div class="pair">
            <NumField layout="stack" label="{fsym}a₂ (stop)" bind:value={$designForm.fa2} edge="fa2" group="stop" unit={uLabel} min={fMin} max={fMax} error={formErrors.fa2} />
            <NumField layout="stack" label="{fsym}p₂ (pass)" bind:value={$designForm.fp2} edge="fp2" group="pass" unit={uLabel} min={fMin} max={fMax} error={formErrors.fp2} />
          </div>
        {/if}
      {/if}
    {/if}

    <div class="pair">
      <NumField layout="stack" label="Ripple" bind:value={$designForm.apDb} edge="apDb" group="pass" unit="dB" min={0.001} max={40} log={false} step={0.1} error={formErrors.apDb} />
      <NumField layout="stack" label="Attenuation" bind:value={$designForm.aaDb} edge="aaDb" group="stop" unit="dB" min={1} max={120} log={false} step={1} error={formErrors.aaDb} />
    </div>
  {/if}

  <!-- ── Output ────────────────────────────────────────────────────────── -->
  <div class="group">Output</div>

  <NumField label="Gain" bind:value={$designForm.gainDb} unit="dB" min={-200} max={200} log={false} step={1} />

  <div class="denorm-row">
    <span class="lbl" title="Where the normalization lands between the passband edge (0 %) and the stopband edge (100 %)">Denorm</span>
    <div class="denorm">
      <input class="slider" type="range" min="0" max="100" step="1" bind:value={$designForm.denorm} aria-label="Denormalization"
        on:input={onDenormInput} on:change={onDenormRelease} on:pointerup={onDenormRelease} on:blur={onDenormRelease} />
      <span class="pct">{$designForm.denorm}%</span>
    </div>
  </div>

  {#if $designError}
    <p class="err">{$designError}</p>
  {/if}

  <div class="design-row">
  <button
    class="btn"
    class:stale
    disabled={!$uiEnabled || $designBusy || hasErrors}
    title={hasErrors ? 'Fix the highlighted fields first' : stale ? 'The form changed since the last design' : ''}
    on:click={design}
  >
    {$designBusy ? 'Computing…' : 'Design Filter'}
    {#if stale && !$designBusy && !$liveMode}<span class="badge">out of date</span>{/if}
  </button>
  <label class="live" class:on={$liveMode} title="Live mode: re-design automatically whenever the form changes (Ctrl+Enter still designs)">
    <input type="checkbox" bind:checked={$liveMode} /> Live
  </label>
  </div>

</div>

<style>
  .fp {
    --lbl-w: 5.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    padding: 0.5rem 0.7rem 0.75rem;
  }

  .group {
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    text-transform: uppercase;
    color: var(--text-dim);
    margin-top: 0.35rem;
    padding-bottom: 0.15rem;
    border-bottom: 1px solid var(--surface-2);
  }
  .group:first-child { margin-top: 0; }

  .pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
    min-width: 0;
    align-items: start;
  }

  .lbl {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.2;
    white-space: nowrap;
  }

  .hint {
    font-size: 0.75rem;
    color: var(--danger);
    margin: -0.2rem 0 0;
    overflow-wrap: anywhere;
  }

  /* Order */
  .order-row {
    display: flex;
    justify-content: space-between;
    align-items: baseline;
    margin-bottom: -0.35rem;
  }
  .order-val {
    font-size: 0.82rem;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    color: var(--text);
  }

  /* Denorm */
  .denorm-row {
    display: grid;
    grid-template-columns: var(--lbl-w) minmax(0, 1fr);
    align-items: center;
    gap: 0.45rem;
  }
  .denorm {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    min-width: 0;
    height: 2rem;
  }
  .pct {
    font-size: 0.82rem;
    color: var(--text-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    min-width: 2.4rem;
    text-align: right;
  }

  /* Tall hit-box so the thumb isn't clipped by a 5px element height */
  .slider {
    -webkit-appearance: none;
    appearance: none;
    flex: 1;
    min-width: 0;
    height: 2rem;
    margin: 0;
    background: transparent;
    outline: none;
    cursor: pointer;
  }
  .slider::-webkit-slider-runnable-track {
    height: 6px;
    border-radius: 3px;
    background: var(--border);
  }
  .slider::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    width: 18px;
    height: 18px;
    margin-top: -6px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
    box-shadow: 0 0 0 1px var(--border);
    cursor: pointer;
  }
  .slider::-moz-range-track {
    height: 6px;
    border-radius: 3px;
    background: var(--border);
  }
  .slider::-moz-range-thumb {
    width: 18px;
    height: 18px;
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
    box-shadow: 0 0 0 1px var(--border);
    cursor: pointer;
  }

  .design-row { display: flex; gap: 0.4rem; align-items: stretch; margin-top: 0.15rem; }
  .design-row .btn { flex: 1; margin-top: 0; }
  .live {
    display: flex; align-items: center; gap: 0.3rem;
    padding: 0 0.55rem; border-radius: 4px; cursor: pointer; user-select: none;
    border: 1px solid var(--border); background: var(--bg);
    font-size: 0.8rem; color: var(--text-muted);
  }
  .live.on { border-color: var(--success); color: var(--success); background: color-mix(in srgb, var(--success) 12%, var(--bg)); }
  .live input { accent-color: var(--success); margin: 0; }
  .btn {
    display: flex;
    align-items: center;
    justify-content: center;
    gap: 0.5rem;
    background: var(--accent-strong);
    border: none;
    border-radius: 4px;
    color: #fff;
    cursor: pointer;
    font-size: 0.9rem;
    font-weight: 600;
    padding: 0.5rem;
    width: 100%;
    margin-top: 0.15rem;
  }
  .btn:hover:not(:disabled) { background: var(--accent-hover); }
  .btn:disabled { background: var(--surface-2); color: var(--disabled); cursor: default; }
  .badge {
    font-size: 0.7rem;
    font-weight: 600;
    background: rgba(255, 255, 255, 0.2);
    border-radius: 999px;
    padding: 0.05rem 0.45rem;
  }

  .err {
    font-size: 0.82rem;
    color: var(--danger);
    background: var(--danger-bg);
    border-radius: 4px;
    padding: 0.4rem 0.5rem;
    word-break: break-word;
    overflow-wrap: anywhere;
    margin: 0;
    max-height: 6rem;
    overflow-y: auto;
  }
</style>
