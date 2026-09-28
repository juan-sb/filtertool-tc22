<script>
  // One stage: summary, normalization, editable f0 / Q / gain offset, reset / remove.
  import { createEventDispatcher } from 'svelte'
  import { dataUnit, hoveredStageId } from '../stores/app.js'
  import { TWO_PI } from '../lib/approx.js'
  import { poleSummary, scaleRoots, withQ, NORM_OPTIONS, normLabel, normProblem, Q_MIN, Q_MAX } from '../lib/stage-math.js'
  import { updateStage, resetStage, removeStage, isModified } from '../lib/stages.js'
  import { formatSI } from '../lib/si.js'
  import NumField from './form/NumField.svelte'

  export let stage
  export let color
  export let filterType = 0

  const dispatch = createEventDispatcher()

  $: uf       = $dataUnit === 'rad' ? TWO_PI : 1
  $: uLabel   = $dataUnit === 'rad' ? 'rad/s' : 'Hz'
  $: fsym     = $dataUnit === 'rad' ? 'ω' : 'f'
  $: summary  = poleSummary(stage.poles)
  $: f0       = summary.w0 != null ? (summary.w0 / TWO_PI) * uf : null   // data unit
  $: hasQ     = stage.poles.length === 2 && Number.isFinite(summary.q)
  $: gainDb   = stage.gain > 0 ? 20 * Math.log10(stage.gain) : null
  $: edited   = isModified(stage)
  $: hovered  = $hoveredStageId === stage.id

  // Local copies for the inputs; pushed to the stage on change.
  let f0Edit, qEdit, gEdit
  $: f0Edit = f0
  $: qEdit  = summary.q
  $: gEdit  = stage.gainDb ?? 0

  function setF0(v) {
    if (!(v > 0) || !(f0 > 0)) return
    const r = v / f0
    updateStage(stage.id, s => ({ zeros: scaleRoots(s.zeros, r), poles: scaleRoots(s.poles, r) }))
  }
  function setQ(v) {
    if (!(v > 0)) return
    updateStage(stage.id, s => ({ poles: withQ(s.poles, v) }))
  }
  function setGain(v) {
    if (Number.isFinite(v)) updateStage(stage.id, { gainDb: v })
  }

  $: if (f0Edit != null && f0 != null && Math.abs(f0Edit / f0 - 1) > 1e-9) setF0(f0Edit)
  $: if (hasQ && qEdit != null && Math.abs(qEdit / summary.q - 1) > 1e-9) setQ(qEdit)
  $: if (gEdit != null && Math.abs(gEdit - (stage.gainDb ?? 0)) > 1e-9) setGain(gEdit)

  const orderText = s => `${s.poles.length}P/${s.zeros.length}Z`
</script>

<!-- svelte-ignore a11y_no_static_element_interactions -->
<div
  class="card"
  class:hovered
  style="--c: {color}"
  on:mouseenter={() => hoveredStageId.set(stage.id)}
  on:mouseleave={() => hoveredStageId.set(null)}
>
  <div class="head">
    <span class="grip" title="Drag to reorder" on:pointerdown={e => dispatch('grab', e)}>⋮⋮</span>
    <span class="swatch"></span>
    <span class="name">{stage.name}</span>
    {#if edited}<span class="badge" title="Changed since the stage was built">edited</span>{/if}
    <span class="spacer"></span>
    <button class="icon" title="Reset to the stage as built" disabled={!edited} on:click={() => resetStage(stage.id)}>↺</button>
    <button class="icon danger" title="Remove stage" on:click={() => removeStage(stage.id)}>×</button>
  </div>

  <div class="meta">
    <span>{orderText(stage)}</span>
    {#if f0 != null}<span>{fsym}₀ {formatSI(f0)} {uLabel}</span>{/if}
    {#if hasQ}<span>Q {summary.q.toFixed(3)}</span>{/if}
    {#if gainDb != null}<span title="Stage gain k (normalization · offset)">k {gainDb.toFixed(2)} dB</span>{/if}
  </div>

  <label class="norm">
    <span class="lbl">Norm.</span>
    <select value={stage.normtype ?? 'Passband'} on:change={e => updateStage(stage.id, { normtype: e.currentTarget.value })}>
      {#each NORM_OPTIONS as n}
        {@const why = normProblem(n, filterType, stage.zeros, stage.poles)}
        <option value={n} disabled={!!why && n !== (stage.normtype ?? 'Passband')}>{normLabel(n, filterType)}{why ? ` — n/a (${why})` : ''}</option>
      {/each}
    </select>
  </label>

  <div class="edits">
    {#if f0 != null}
      <NumField layout="stack" label="{fsym}₀" bind:value={f0Edit} unit={uLabel} min={1e-6} max={1e15} />
    {/if}
    {#if hasQ}
      <NumField layout="stack" label="Q" bind:value={qEdit} min={Q_MIN} max={Q_MAX} />
    {/if}
    <NumField layout="stack" label="Gain" bind:value={gEdit} unit="dB" min={-200} max={200} log={false} step={0.5}
      title="Gain offset on top of the normalization; drag to adjust" />
  </div>
</div>

<style>
  .card {
    border: 1px solid var(--border);
    border-left: 3px solid var(--c);
    border-radius: 5px;
    background: var(--bg);
    padding: 0.4rem 0.5rem 0.5rem;
    display: flex;
    flex-direction: column;
    gap: 0.35rem;
    transition: background 0.12s, box-shadow 0.12s;
  }
  .card.hovered {
    background: color-mix(in srgb, var(--c) 10%, var(--bg));
    box-shadow: 0 0 0 1px var(--c);
  }

  .head { display: flex; align-items: center; gap: 0.35rem; min-width: 0; }
  .grip { cursor: grab; color: var(--text-dim); font-size: 0.7rem; letter-spacing: -2px; user-select: none; touch-action: none; }
  .swatch { width: 0.6rem; height: 0.6rem; border-radius: 50%; background: var(--c); flex-shrink: 0; }
  .name { font-size: 0.85rem; font-weight: 600; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }
  .badge {
    font-size: 0.66rem; padding: 0 0.35rem; border-radius: 999px;
    border: 1px solid var(--warning); color: var(--warning);
  }
  .spacer { flex: 1; }
  .icon {
    background: none; border: 1px solid transparent; border-radius: 3px;
    color: var(--text-dim); cursor: pointer; font-size: 0.9rem; line-height: 1; padding: 0.05rem 0.3rem;
  }
  .icon:hover:not(:disabled) { color: var(--text); border-color: var(--border); }
  .icon.danger:hover { color: var(--danger); }
  .icon:disabled { opacity: 0.3; cursor: default; }

  .meta {
    display: flex; flex-wrap: wrap; gap: 0.2rem 0.6rem;
    font-size: 0.74rem; color: var(--text-muted);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }

  .norm { display: flex; align-items: center; gap: 0.4rem; }
  .lbl { font-size: 0.78rem; color: var(--text-muted); }
  select {
    flex: 1; min-width: 0;
    background: var(--bg); border: 1px solid var(--border); border-radius: 4px;
    color: var(--text); font-size: 0.78rem; padding: 0.2rem 0.3rem; outline: none;
  }
  select:focus { border-color: var(--accent); }

  .edits {
    display: grid;
    grid-template-columns: repeat(3, minmax(0, 1fr));
    gap: 0.35rem;
  }
</style>
