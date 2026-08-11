<script>
  import { bodeData, filterParams, comparisons, theme, compareDash, colorMode, colorShuffle, activeTab, plotUnit } from '../../stores/app.js'
  import { APPROX_NAMES, plotColor, compareLine, freqAxis } from '../../lib/approx.js'
  import BodePlot from '../BodePlot.svelte'

  $: axis = freqAxis($plotUnit)

  $: traces = [
    ...($bodeData ? [{
      x: $bodeData.freq.map(f => f * axis.scale),
      y: $bodeData.groupDelay.map(v => v * 1000),
      mode: 'lines',
      name: APPROX_NAMES[$filterParams?.approx_type ?? 0],
      line: { color: plotColor($filterParams?.approx_type ?? 0, $theme, $colorMode, $colorShuffle), width: 2 },
    }] : []),
    ...$comparisons.map(c => ({
      x: c.bodeData.freq.map(f => f * axis.scale),
      y: c.bodeData.groupDelay.map(v => v * 1000),
      mode: 'lines',
      name: APPROX_NAMES[c.approxType],
      line: compareLine(c.approxType, $theme, { dash: $compareDash, mode: $colorMode, shuffle: $colorShuffle }),
    })),
  ]

  $: yLabel = $plotUnit === 'rad' ? '$\\tau(\\omega)$ [ms]' : '$\\tau(f)$ [ms]'
</script>

<BodePlot
  {traces}
  {yLabel}
  xLabel={axis.xLabel}
  logX={true}
  filename="filtool_groupdelay"
  active={$activeTab === 'groupDelay'}
/>
