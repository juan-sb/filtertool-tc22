<script>
  // Labelled numeric field: drag the label to scrub, wheel/arrows while focused,
  // SI-prefix entry (SciInput), inline validation hint.
  import SciInput from '../SciInput.svelte'
  import { scrub } from '../../lib/scrub.js'

  export let label
  export let value
  export let unit = ''
  export let min = -Infinity
  export let max = Infinity
  /** Log (×/÷) nudging and scrubbing — frequencies, times. Otherwise linear by `step`. */
  export let log = true
  export let step = 1
  /** Validation message for this field, or '' / undefined. */
  export let error = ''
  /** 'row' = label beside the input, 'stack' = label above. */
  export let layout = 'row'
  export let title = ''
</script>

<div class="nf {layout}">
  <span
    class="lbl"
    title={title || `Drag to adjust ${label}${log ? '' : ` (${step} ${unit} per 4 px)`}; Shift = fine`}
    use:scrub={{ get: () => value, set: v => (value = v), log, step, min, max }}
  >{label}</span>
  <SciInput bind:value {unit} {min} {max} logNudge={log} {step} invalid={!!error} on:change />
  {#if error}<span class="hint">{error}</span>{/if}
</div>

<style>
  .nf { min-width: 0; }
  .nf.row {
    display: grid;
    grid-template-columns: var(--lbl-w, 5.5rem) minmax(0, 1fr);
    align-items: center;
    column-gap: 0.45rem;
    row-gap: 0.2rem;
  }
  .nf.stack {
    display: flex;
    flex-direction: column;
    gap: 0.2rem;
  }

  .lbl {
    font-size: 0.82rem;
    color: var(--text-muted);
    line-height: 1.2;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
    user-select: none;
    touch-action: none;
  }
  .lbl:global(.scrubbable) { cursor: ew-resize; }
  .lbl:global(.scrubbable):hover { color: var(--accent); }
  .lbl:global(.scrubbing) { color: var(--accent); text-decoration: underline dotted; }

  .hint {
    grid-column: 1 / -1;
    font-size: 0.75rem;
    color: var(--danger);
    line-height: 1.25;
    overflow-wrap: anywhere;
  }
  .nf.row .hint { grid-column: 2; }
</style>
