<script>
  import { getWorkerApi }  from '../lib/worker-client.js'
  import { freqRangeFromParams, TWO_PI } from '../lib/approx.js'
  import { filterParams, filterResult, bodeData, stages, bodePoints, uiEnabled, engineStatus, pendingFormHydration, dataUnit } from '../stores/app.js'
  import SciInput from './SciInput.svelte'

  // ── Constants ─────────────────────────────────────────────────────────────
  const FILTER_TYPES  = ['Low-pass', 'High-pass', 'Band-pass', 'Band-reject', 'Group Delay']
  const APPROX_TYPES  = ['Butterworth', 'Chebyshev I', 'Chebyshev II', 'Cauer', 'Legendre', 'Bessel', 'Gauss']

  // ── Form state ────────────────────────────────────────────────────────────
  // Frequencies below are held in the current data unit (Hz or rad/s); params
  // sent to the engine are always rad/s.
  let filterType  = 0
  let approxType  = 0
  let nMin = 1,   nMax = 10
  let apDb = 3,   aaDb = 40,  gainDb = 0
  let denorm      = 0     // 0–100 %

  // LP / HP
  let fp = 1000, fa = 2000

  // BP / BR
  let defineWith = 1
  let f0 = 1000, bwp = 200, bwa = 600
  let fp1 = 800, fp2 = 1200, fa1 = 600, fa2 = 1500

  // Group Delay
  let tau0 = 1e-3, frg = 1000, gamma = 5

  // ── Units ─────────────────────────────────────────────────────────────────
  /** Hz value × uf = value in the current data unit. */
  $: uf     = $dataUnit === 'rad' ? TWO_PI : 1
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
    fp *= k;  fa *= k
    f0 *= k;  bwp *= k; bwa *= k
    fp1 *= k; fp2 *= k; fa1 *= k; fa2 *= k
    frg *= k
  }

  // ── Derived ───────────────────────────────────────────────────────────────
  $: isBand     = filterType === 2 || filterType === 3
  $: isGD       = filterType === 4

  // Apply params from Save/Load without re-running Design.
  $: if ($pendingFormHydration) {
    applyParamsToForm($pendingFormHydration)
    pendingFormHydration.set(null)
  }

  // ── Submit ────────────────────────────────────────────────────────────────
  let computing = false
  let errorMsg  = ''

  async function design() {
    computing = true
    errorMsg  = ''
    engineStatus.set('Computing…')
    try {
      const params = buildParams()
      const api    = getWorkerApi()
      const result = await api.filterDesign(params)
      if (result.error) { errorMsg = result.error.split('\n').at(-2) ?? result.error; return }
      stages.set([])
      filterParams.set(params)
      filterResult.set(result)
      const r = freqRangeFromParams(params)
      bodeData.set(await api.computeBode(result.num, result.den, r.min, r.max, $bodePoints))
      engineStatus.set('Ready')
    } catch (e) {
      errorMsg = e.message
      engineStatus.set('Ready')
    } finally { computing = false }
  }

  function applyParamsToForm(p) {
    const fromRad = w => (Number(w) / TWO_PI) * uf
    filterType = p.filter_type ?? 0
    approxType = p.approx_type ?? 0
    nMin = p.N_min ?? 1
    nMax = p.N_max ?? 10
    apDb = p.ap_dB ?? 3
    aaDb = p.aa_dB ?? 40
    gainDb = 20 * Math.log10(Math.max(p.gain ?? 1, 1e-12))
    denorm = p.denorm ?? 0
    defineWith = p.define_with ?? 1
    gamma = p.gamma ?? 5
    tau0 = p.tau0 ?? 1e-3

    if (filterType === 4) {
      frg = fromRad(p.wrg ?? 0) || 1000 * uf
      return
    }
    if (filterType === 0 || filterType === 1) {
      fp = fromRad(p.wp) || 1000 * uf
      fa = fromRad(p.wa) || 2000 * uf
      return
    }
    // Band-pass / band-reject
    if (defineWith === 1) {
      f0 = fromRad(p.w0) || 1000 * uf
      bwp = fromRad(p.bw?.[0]) || 200 * uf
      bwa = fromRad(p.bw?.[1]) || 600 * uf
    } else {
      fp1 = fromRad(p.wp?.[0]) || 800 * uf
      fp2 = fromRad(p.wp?.[1]) || 1200 * uf
      fa1 = fromRad(p.wa?.[0]) || 600 * uf
      fa2 = fromRad(p.wa?.[1]) || 1500 * uf
    }
  }

  function buildParams() {
    const toRad = v => v * TWO_PI / uf
    const base = {
      filter_type: filterType, approx_type: approxType,
      N_min: nMin, N_max: nMax,
      ap_dB: apDb, aa_dB: aaDb,
      gain: Math.pow(10, gainDb / 20),
      normalization: 'Passband',
      is_helper: false, helper_approx: [], helper_N: -1,
      define_with: defineWith, denorm,
      gamma, tau0,
    }
    if (isGD) return { ...base, wrg: toRad(frg), wp: 0, wa: 0, w0: 0, bw: [0,0] }
    if (!isBand) return { ...base, wp: toRad(fp), wa: toRad(fa), w0: 0, bw:[0,0], wrg:0 }
    if (defineWith === 1) return {
      ...base,
      wp: [toRad(f0 - bwp/2), toRad(f0 + bwp/2)],
      wa: [toRad(f0 - bwa/2), toRad(f0 + bwa/2)],
      w0: toRad(f0), bw: [toRad(bwp), toRad(bwa)], wrg: 0,
    }
    return {
      ...base,
      wp: [toRad(fp1), toRad(fp2)], wa: [toRad(fa1), toRad(fa2)],
      w0: toRad(Math.sqrt(fp1 * fp2)),
      bw: [toRad(fp2 - fp1), toRad(fa2 - fa1)], wrg: 0,
    }
  }
</script>

<div class="fp">

  <div class="pair">
    <div class="stack">
      <span class="lbl">Type</span>
      <select class="ctl" bind:value={filterType}>
        {#each FILTER_TYPES as t, i}<option value={i}>{t}</option>{/each}
      </select>
    </div>
    <div class="stack">
      <span class="lbl">Approx</span>
      <select class="ctl" bind:value={approxType}>
        {#each APPROX_TYPES as a, i}<option value={i}>{a}</option>{/each}
      </select>
    </div>
  </div>

  <div class="pair">
    <div class="stack">
      <span class="lbl">N min</span>
      <input class="ctl num" type="number" min="1" max="50" bind:value={nMin} />
    </div>
    <div class="stack">
      <span class="lbl">N max</span>
      <input class="ctl num" type="number" min="1" max="50" bind:value={nMax} />
    </div>
  </div>

  <div class="rule"></div>

  {#if !isGD}
    {#if !isBand}
      <div class="pair">
        <div class="stack">
          <span class="lbl">{fsym}p</span>
          <SciInput bind:value={fp} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="stack">
          <span class="lbl">{fsym}a</span>
          <SciInput bind:value={fa} unit={uLabel} min={fMin} max={fMax} />
        </div>
      </div>
    {:else}
      <div class="row">
        <span class="lbl">Define</span>
        <select class="ctl" bind:value={defineWith}>
          <option value={1}>{fsym}₀ + BW</option>
          <option value={0}>Frequencies</option>
        </select>
      </div>
      {#if defineWith === 1}
        <div class="row">
          <span class="lbl">{fsym}₀</span>
          <SciInput bind:value={f0} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">BWp</span>
          <SciInput bind:value={bwp} unit={uLabel} min={bwMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">BWa</span>
          <SciInput bind:value={bwa} unit={uLabel} min={bwMin} max={fMax} />
        </div>
      {:else}
        <div class="row">
          <span class="lbl">{fsym}p₁</span>
          <SciInput bind:value={fp1} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">{fsym}p₂</span>
          <SciInput bind:value={fp2} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">{fsym}a₁</span>
          <SciInput bind:value={fa1} unit={uLabel} min={fMin} max={fMax} />
        </div>
        <div class="row">
          <span class="lbl">{fsym}a₂</span>
          <SciInput bind:value={fa2} unit={uLabel} min={fMin} max={fMax} />
        </div>
      {/if}
    {/if}

    <div class="rule"></div>

    <div class="pair">
      <div class="stack">
        <span class="lbl">Ripple</span>
        <SciInput bind:value={apDb} unit="dB" min={0.001} max={40} logNudge={false} step={0.5} />
      </div>
      <div class="stack">
        <span class="lbl">Attenuation</span>
        <SciInput bind:value={aaDb} unit="dB" min={1} max={120} logNudge={false} step={1} />
      </div>
    </div>

  {:else}
    <div class="row">
      <span class="lbl">τ₀</span>
      <SciInput bind:value={tau0} unit="s" min={1e-12} max={1} />
    </div>
    <div class="row">
      <span class="lbl">{fsym} ref</span>
      <SciInput bind:value={frg} unit={uLabel} min={fMin} max={fMax} />
    </div>
    <div class="row">
      <span class="lbl">γ</span>
      <div class="with-unit">
        <input class="ctl num" type="number" min="0.01" max="99" step="0.5" bind:value={gamma} />
        <span class="unit">%</span>
      </div>
    </div>
  {/if}

  <div class="rule"></div>

  <div class="row">
    <span class="lbl">Gain</span>
    <SciInput bind:value={gainDb} unit="dB" logNudge={false} step={1} />
  </div>

  <div class="row">
    <span class="lbl">Denorm</span>
    <div class="denorm">
      <input class="slider" type="range" min="0" max="100" step="1" bind:value={denorm} />
      <span class="pct">{denorm}%</span>
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
