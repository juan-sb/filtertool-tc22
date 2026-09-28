<script>
  import { getWorkerApi }  from '../lib/worker-client.js'
  import { freqRangeFromParams, TWO_PI } from '../lib/approx.js'
  import { GD, isBand as isBandType, buildParams, formFromParams, rescaleForm, validateForm, switchFilterType } from '../lib/params.js'
  import { withRoots } from '../lib/roots.js'
  import { designForm, filterParams, filterResult, bodeData, stages, bodePoints, uiEnabled, engineStatus, pendingFormHydration, dataUnit } from '../stores/app.js'
  import SciInput from './SciInput.svelte'

  // ── Constants ─────────────────────────────────────────────────────────────
  const FILTER_TYPES  = ['Low-pass', 'High-pass', 'Band-pass', 'Band-reject', 'Group Delay']
  const APPROX_TYPES  = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Cauer', 'Legendre', 'Bessel', 'Gauss']

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
  $: isBand     = isBandType($designForm.filterType)
  $: isGD       = $designForm.filterType === GD
  $: formErrors = validateForm($designForm)

  function onTypeChange(e) {
    designForm.update(f => switchFilterType(f, +e.currentTarget.value))
  }

  // Apply params from Save/Load without re-running Design.
  $: if ($pendingFormHydration) {
    designForm.update(f => formFromParams($pendingFormHydration, toRad, f))
    pendingFormHydration.set(null)
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  let computing = false
  let errorMsg  = ''

  async function design() {
    const errs = Object.values(formErrors)
    if (errs.length) { errorMsg = errs[0]; return }
    computing = true
    errorMsg  = ''
    engineStatus.set('Computing…')
    try {
      const params = buildParams($designForm, toRad)
      const api    = getWorkerApi()
      const result = await api.filterDesign(params)
      if (result.error) { errorMsg = result.error.split('\n').at(-2) ?? result.error; return }
      stages.set([])
      filterParams.set(params)
      filterResult.set(withRoots(result))
      const r = freqRangeFromParams(params)
      bodeData.set(await api.computeBode(result.num, result.den, r.min, r.max, $bodePoints))
      engineStatus.set('Ready')
    } catch (e) {
      errorMsg = e.message
      engineStatus.set('Ready')
    } finally { computing = false }
  }
</script>

<div class="fp">

  <div class="pair">
    <div class="stack">
      <span class="lbl">Type</span>
      <select class="ctl" value={$designForm.filterType} on:change={onTypeChange}>
        {#each FILTER_TYPES as t, i}<option value={i}>{t}</option>{/each}
      </select>
    </div>
    <div class="stack">
      <span class="lbl">Approx</span>
      <select class="ctl" bind:value={$designForm.approxType}>
        {#each APPROX_TYPES as a, i}<option value={i}>{a}</option>{/each}
      </select>
    </div>
  </div>

  <div class="pair">
    <div class="stack">
      <span class="lbl">N min</span>
      <input class="ctl num" type="number" min="1" max="50" bind:value={$designForm.nMin} />
    </div>
    <div class="stack">
      <span class="lbl">N max</span>
      <input class="ctl num" type="number" min="1" max="50" bind:value={$designForm.nMax} />
    </div>
  </div>

  <div class="rule"></div>

  {#if !isGD}
    {#if !isBand}
      <div class="pair">
        <div class="stack">
          <span class="lbl">{fsym}p</span>
          <SciInput bind:value={$designForm.fp} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="stack">
          <span class="lbl">{fsym}a</span>
          <SciInput bind:value={$designForm.fa} unit={uLabel} min={fMin} max={fMax} />
        </div>
      </div>
    {:else}
      <div class="row">
        <span class="lbl">Define</span>
        <select class="ctl" bind:value={$designForm.defineWith}>
          <option value={1}>{fsym}₀ + BW</option>
          <option value={0}>Frequencies</option>
        </select>
      </div>
      {#if $designForm.defineWith === 1}
        <div class="row">
          <span class="lbl">{fsym}₀</span>
          <SciInput bind:value={$designForm.f0} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">BWp</span>
          <SciInput bind:value={$designForm.bwp} unit={uLabel} min={bwMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">BWa</span>
          <SciInput bind:value={$designForm.bwa} unit={uLabel} min={bwMin} max={fMax} />
        </div>
      {:else}
        <div class="row">
          <span class="lbl">{fsym}p₁</span>
          <SciInput bind:value={$designForm.fp1} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">{fsym}p₂</span>
          <SciInput bind:value={$designForm.fp2} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">{fsym}a₁</span>
          <SciInput bind:value={$designForm.fa1} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">{fsym}a₂</span>
          <SciInput bind:value={$designForm.fa2} unit={uLabel} min={fMin} max={fMax} />
        </div>
      {/if}
    {/if}

    <div class="rule"></div>

    <div class="pair">
      <div class="stack">
        <span class="lbl">Ripple</span>
        <SciInput bind:value={$designForm.apDb} unit="dB" min={0.001} max={40} logNudge={false} step={0.5} />
      </div>
      <div class="stack">
        <span class="lbl">Attenuation</span>
        <SciInput bind:value={$designForm.aaDb} unit="dB" min={1} max={120} logNudge={false} step={1} />
      </div>
    </div>

  {:else}
    <div class="row">
      <span class="lbl">τ₀</span>
      <SciInput bind:value={$designForm.tau0} unit="s" min={1e-12} max={1} />
    </div>
    <div class="row">
      <span class="lbl">{fsym} ref</span>
      <SciInput bind:value={$designForm.frg} unit={uLabel} min={fMin} max={fMax} />
    </div>
    <div class="row">
      <span class="lbl">γ</span>
      <div class="with-unit">
        <input class="ctl num" type="number" min="0.01" max="99" step="0.5" bind:value={$designForm.gamma} />
        <span class="unit">%</span>
      </div>
    </div>
  {/if}

  <div class="rule"></div>

  <div class="row">
    <span class="lbl">Gain</span>
    <SciInput bind:value={$designForm.gainDb} unit="dB" logNudge={false} step={1} />
  </div>

  <div class="row">
    <span class="lbl">Denorm</span>
    <div class="denorm">
      <input class="slider" type="range" min="0" max="100" step="1" bind:value={$designForm.denorm} />
      <span class="pct">{$designForm.denorm}%</span>
    </div>
  </div>

  {#if errorMsg}
    <p class="err">{errorMsg}</p>
  {/if}

  <button class="btn" disabled={!$uiEnabled || computing} on:click={design}>
    {computing ? 'Computing…' : 'Design Filter'}
  </button>

</div>

<style>
  .fp {
    --lbl-w: 5.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.45rem;
    padding: 0.5rem 0.7rem 0.75rem;
  }

  .row {
    display: grid;
    grid-template-columns: var(--lbl-w) minmax(0, 1fr);
    align-items: center;
    gap: 0.45rem;
    min-width: 0;
  }

  .pair {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.45rem;
    min-width: 0;
  }

  .stack {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
    min-width: 0;
  }

  .lbl {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.2;
    white-space: nowrap;
    text-align: left;
  }

  .ctl {
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 4px;
    color: var(--text);
    font-size: 0.88rem;
    padding: 0.35rem 0.45rem;
    width: 100%;
    min-width: 0;
    outline: none;
  }
  .ctl:focus { border-color: var(--accent); }
  .ctl.num { font-family: ui-monospace, 'SF Mono', Consolas, monospace; }

  .with-unit {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    min-width: 0;
  }
  .with-unit .ctl { flex: 1; }
  .unit {
    font-size: 0.8rem;
    color: var(--text-dim);
    flex-shrink: 0;
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

  .rule {
    height: 1px;
    background: var(--surface-2);
    margin: 0.15rem 0;
  }

  .btn {
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
