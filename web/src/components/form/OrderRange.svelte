<script>
  // Two-thumb N range (N min / N max) with a marker at the designed order.
  // Square-root scale: low orders (the common case) get most of the track.
  import { createEventDispatcher } from 'svelte'

  export let lo = 1
  export let hi = 10
  export let min = 1
  export let max = 50
  /** Order chosen by the last design, or null. */
  export let designed = null
  /** True when the form no longer matches the last design (marker is dimmed). */
  export let stale = false

  const dispatch = createEventDispatcher()
  const RES = 1000   // native range resolution
  const TICKS = [1, 5, 10, 20, 50]

  const toPos   = n => Math.sqrt((n - min) / (max - min)) * RES
  const fromPos = p => Math.round(min + (max - min) * (p / RES) ** 2)
  const pct     = n => (toPos(n) / RES) * 100

  function set(which, v) {
    v = Math.round(Math.min(max, Math.max(min, v)))
    if (which === 'lo') lo = Math.min(v, hi)
    else hi = Math.max(v, lo)
    dispatch('change', { lo, hi })
  }

  // Arrow keys / wheel step exactly one order (the sqrt scale makes native steps uneven).
  function onKeydown(which, e) {
    const d = { ArrowRight: 1, ArrowUp: 1, ArrowLeft: -1, ArrowDown: -1, PageUp: 5, PageDown: -5 }[e.key]
    const jump = { Home: min, End: max }[e.key]
    if (d == null && jump == null) return
    e.preventDefault()
    set(which, jump ?? (which === 'lo' ? lo : hi) + d)
  }

  // Wheel only steps a focused thumb, so it never hijacks sidebar scrolling.
  function onWheel(which, e) {
    if (document.activeElement !== e.currentTarget) return
    e.preventDefault()
    set(which, (which === 'lo' ? lo : hi) + (e.deltaY < 0 ? 1 : -1))
  }

  // When both thumbs sit on the same value, the one on top must be the one that
  // can still move: raise "lo" near the top of the range, "hi" elsewhere.
  $: loOnTop = lo === hi && hi >= (min + max) / 2
</script>

<div class="order">
  <div class="track-wrap">
    <div class="track"></div>
    <div class="fill" style="left: {pct(lo)}%; right: {100 - pct(hi)}%"></div>
    {#if designed != null}
      <div class="mark" class:stale style="left: {pct(Math.min(max, designed))}%">
        <span>{designed}</span>
      </div>
    {/if}
    <input type="range" class="thumb" class:top={loOnTop} min="0" max={RES} step="1" value={toPos(lo)}
      aria-label="Minimum order" aria-valuemin={min} aria-valuemax={max} aria-valuenow={lo} aria-valuetext="N min {lo}"
      on:input={e => set('lo', fromPos(+e.currentTarget.value))}
      on:change={e => (e.currentTarget.value = String(toPos(lo)))}
      on:keydown={e => onKeydown('lo', e)}
      on:wheel={e => onWheel('lo', e)} />
    <input type="range" class="thumb" class:top={!loOnTop} min="0" max={RES} step="1" value={toPos(hi)}
      aria-label="Maximum order" aria-valuemin={min} aria-valuemax={max} aria-valuenow={hi} aria-valuetext="N max {hi}"
      on:input={e => set('hi', fromPos(+e.currentTarget.value))}
      on:change={e => (e.currentTarget.value = String(toPos(hi)))}
      on:keydown={e => onKeydown('hi', e)}
      on:wheel={e => onWheel('hi', e)} />
  </div>
  <div class="ticks">
    {#each TICKS.filter(t => t >= min && t <= max) as t}
      <span style="left: {pct(t)}%">{t}</span>
    {/each}
  </div>
</div>

<style>
  .order {
    --thumb: 16px;
    display: flex;
    flex-direction: column;
    min-width: 0;
    padding-top: 1rem;   /* room for the designed-N label */
  }

  .track-wrap, .ticks {
    position: relative;
    margin: 0 calc(var(--thumb) / 2);
  }
  .track-wrap { height: var(--thumb); }

  .track, .fill {
    position: absolute;
    top: 50%;
    height: 6px;
    margin-top: -3px;
    border-radius: 3px;
  }
  .track { left: 0; right: 0; background: var(--border); }
  .fill  { background: var(--accent); opacity: 0.55; }

  .mark {
    position: absolute;
    top: -0.2rem;
    bottom: -0.2rem;
    width: 2px;
    margin-left: -1px;
    background: var(--success);
    pointer-events: none;
    z-index: 3;
  }
  .mark span {
    position: absolute;
    bottom: 100%;
    left: 50%;
    transform: translateX(-50%);
    font-size: 0.72rem;
    font-weight: 700;
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
    color: var(--success);
    line-height: 1.1;
    white-space: nowrap;
  }
  .mark span::before { content: 'N='; font-weight: 400; }
  .mark.stale { background: var(--text-dim); opacity: 0.7; }
  .mark.stale span { color: var(--text-dim); }

  /* Two stacked native ranges; only their thumbs take pointer events. */
  .thumb {
    -webkit-appearance: none;
    appearance: none;
    position: absolute;
    left: calc(var(--thumb) / -2);
    width: calc(100% + var(--thumb));
    top: 0;
    height: var(--thumb);
    margin: 0;
    background: transparent;
    pointer-events: none;
    outline: none;
    z-index: 1;
  }
  .thumb.top { z-index: 2; }
  .thumb::-webkit-slider-runnable-track { background: transparent; height: var(--thumb); }
  .thumb::-moz-range-track { background: transparent; }
  .thumb::-webkit-slider-thumb {
    -webkit-appearance: none;
    appearance: none;
    pointer-events: auto;
    width: var(--thumb);
    height: var(--thumb);
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
    box-shadow: 0 0 0 1px var(--border);
    cursor: grab;
  }
  .thumb::-moz-range-thumb {
    pointer-events: auto;
    width: var(--thumb);
    height: var(--thumb);
    border-radius: 50%;
    background: var(--accent);
    border: 2px solid var(--surface);
    box-shadow: 0 0 0 1px var(--border);
    cursor: grab;
  }
  .thumb:focus-visible::-webkit-slider-thumb { box-shadow: 0 0 0 2px var(--accent); }
  .thumb:focus-visible::-moz-range-thumb { box-shadow: 0 0 0 2px var(--accent); }

  .ticks {
    height: 0.9rem;
    margin-top: 0.15rem;
    font-size: 0.68rem;
    color: var(--text-dim);
    font-family: ui-monospace, 'SF Mono', Consolas, monospace;
  }
  .ticks span {
    position: absolute;
    transform: translateX(-50%);
    line-height: 1;
  }
</style>
