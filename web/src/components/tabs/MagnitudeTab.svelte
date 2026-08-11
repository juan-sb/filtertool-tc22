<script>
  import { bodeData, filterParams, comparisons, theme, compareDash, colorMode, colorShuffle, activeTab, plotUnit } from '../../stores/app.js'
  import { APPROX_NAMES, plotColor, compareLine, freqAxis } from '../../lib/approx.js'
  import BodePlot from '../BodePlot.svelte'

  export let showTemplate = true

  function toDb(v) {
    if (!(v > 0)) return null
    const db = 20 * Math.log10(v)
    return Number.isFinite(db) ? db : null
  }

  $: axis = freqAxis($plotUnit)

  function buildTemplateShapes(params, bode, ax) {
    if (!params || !bode || params.filter_type === 4) return []

    const freq   = bode.freq
    const X0     = freq[0] * ax.scale
    const X1     = freq[freq.length - 1] * ax.scale
    const dbVals = bode.magnitude.map(toDb).filter(v => v != null)
    const Y0     = (dbVals.length ? Math.min(...dbVals) : -100) - 5
    const Y1     = (dbVals.length ? Math.max(...dbVals) : 0) + 5

    // Template bounds live in params as rad/s — map them onto the plot's unit.
    const toX    = w => w * ax.sScale
    const gainDb = 20 * Math.log10(Math.max(params.gain ?? 1, 1e-12))
    const pbBound = gainDb - params.ap_dB
    const sbBound = gainDb - params.aa_dB
    // Soft pink on light (desktop #ffcccb); clear red tint on dark (pastel pink goes muddy)
    const fill = $theme === 'light'
      ? 'rgba(255, 204, 203, 0.45)'
      : 'rgba(248, 81, 73, 0.16)'

    const rect = (x0, x1, y0, y1) => ({
      type: 'rect', xref: 'x', yref: 'y', layer: 'below',
      x0: Math.max(x0, X0), x1: Math.min(x1, X1),
      y0: Math.max(y0, Y0), y1: Math.min(y1, Y1),
      fillcolor: fill,
      line: { width: 0 },
    })

    const ft = params.filter_type
    if (ft === 0) {  // LP
      const fp = toX(params.wp), fa = toX(params.wa)
      return [rect(X0, fp, Y0, pbBound), rect(fa, X1, sbBound, Y1)]
    }
    if (ft === 1) {  // HP
      const fp = toX(params.wp), fa = toX(params.wa)
      return [rect(fp, X1, Y0, pbBound), rect(X0, fa, sbBound, Y1)]
    }
    if (ft === 2) {  // BP
      const [fp1, fp2] = params.wp.map(toX)
      const [fa1, fa2] = params.wa.map(toX)
      return [
        rect(fp1, fp2, Y0, pbBound),
        rect(X0, fa1, sbBound, Y1),
        rect(fa2, X1, sbBound, Y1),
      ]
    }
    if (ft === 3) {  // BR
      const [fp1, fp2] = params.wp.map(toX)
      const [fa1, fa2] = params.wa.map(toX)
      return [
        rect(X0, fp1, Y0, pbBound),
        rect(fp2, X1, Y0, pbBound),
        rect(fa1, fa2, sbBound, Y1),
      ]
    }
    return []
  }

  function buildTemplateRange(params) {
    if (!showTemplate || !params || params.filter_type === 4) return null
    const gainDb = 20 * Math.log10(Math.max(params.gain ?? 1, 1e-12))
    return [gainDb - 2 * params.aa_dB, gainDb]
  }

  $: traces = [
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
  ]

  $: shapes = showTemplate ? buildTemplateShapes($filterParams, $bodeData, axis) : []
  $: yRange = buildTemplateRange($filterParams)
  $: yLabel = $plotUnit === 'rad' ? '$|H(\\omega)|$ [dB]' : '$|H(f)|$ [dB]'
</script>

<BodePlot
  {traces}
  {shapes}
  {yRange}
  {yLabel}
  xLabel={axis.xLabel}
  logX={true}
  filename={showTemplate ? 'filtool_template' : 'filtool_magnitude'}
  active={$activeTab === (showTemplate ? 'template' : 'magnitude')}
/>
