<script>
  import { createEventDispatcher } from 'svelte'

  /** @type {{ value: any, label: string, title?: string, glyph?: string }[]} */
  export let options = []
  export let value
  export let ariaLabel = ''
  /** 'sm' = compact inline control (e.g. in a section header). */
  export let size = 'md'

  const dispatch = createEventDispatcher()

  function pick(v) {
    if (v === value) return
    value = v
    dispatch('change', v)
  }

  // Roving arrow-key selection, like a native radio group.
  function onKeydown(e) {
    const dir = e.key === 'ArrowRight' || e.key === 'ArrowDown' ? 1
      : e.key === 'ArrowLeft' || e.key === 'ArrowUp' ? -1 : 0
    if (!dir) return
    e.preventDefault()
    const i = options.findIndex(o => o.value === value)
    const next = options[(i + dir + options.length) % options.length]
    pick(next.value)
    e.currentTarget.querySelector(`[data-i="${options.indexOf(next)}"]`)?.focus()
  }
</script>

<!-- svelte-ignore a11y-interactive-supports-focus -->
<div class="seg" class:sm={size === 'sm'} role="radiogroup" aria-label={ariaLabel} on:keydown={onKeydown}>
  {#each options as o, i}
    <button
      type="button"
      role="radio"
      data-i={i}
      aria-checked={o.value === value}
      tabindex={o.value === value ? 0 : -1}
      class:on={o.value === value}
      class:has-glyph={!!o.glyph}
      title={o.title ?? o.label}
      on:click={() => pick(o.value)}
    >
      {#if o.glyph}
        <svg viewBox="0 0 24 12" aria-hidden="true"><path d={o.glyph} /></svg>
      {/if}
      <span>{o.label}</span>
    </button>
  {/each}
</div>

<style>
  .seg {
    display: flex;
    background: var(--bg);
    border: 1px solid var(--border);
    border-radius: 5px;
    padding: 2px;
    gap: 2px;
    min-width: 0;
  }

  button {
    flex: 1 1 0;
    min-width: 0;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    gap: 0.1rem;
    background: transparent;
    border: none;
    border-radius: 3px;
    color: var(--text-dim);
    cursor: pointer;
    font: inherit;
    font-size: 0.78rem;
    font-weight: 600;
    padding: 0.28rem 0.2rem;
    white-space: nowrap;
  }
  button:hover:not(.on) { background: var(--surface-2); color: var(--text-muted); }
  button:focus-visible { outline: 2px solid var(--accent); outline-offset: -2px; }
  button.on {
    background: var(--selected);
    color: var(--accent);
    box-shadow: inset 0 0 0 1px var(--accent);
  }

  .seg.sm { padding: 1px; gap: 1px; border-radius: 4px; }
  .seg.sm button { font-size: 0.68rem; padding: 0.1rem 0.45rem; letter-spacing: 0; text-transform: none; }

  svg {
    width: 1.5rem;
    height: 0.75rem;
    fill: none;
    stroke: currentColor;
    stroke-width: 1.6;
    stroke-linecap: round;
    stroke-linejoin: round;
  }
</style>
