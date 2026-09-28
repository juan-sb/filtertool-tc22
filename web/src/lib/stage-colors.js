// One colour per stage, shared by the Stages plot, stage cards and PZ maps.

const DARK  = ['#58a6ff', '#3fb950', '#d29922', '#bc8cff', '#f78166', '#79c0ff', '#ffa657', '#39d353']
const LIGHT = ['#0969da', '#1a7f37', '#9a6700', '#8250df', '#cf222e', '#0550ae', '#bc4c00', '#116329']

export function stageColor(index, theme = 'dark') {
  const p = theme === 'light' ? LIGHT : DARK
  return p[((index % p.length) + p.length) % p.length]
}

/** A stage keeps its colour when stages are reordered or removed. */
export const colorOf = (stage, fallbackIndex, theme) => stageColor(stage.colorIndex ?? fallbackIndex, theme)

/** Colour index for a new stage: one past the highest in use. */
export const nextColorIndex = list => list.reduce((m, s, i) => Math.max(m, (s.colorIndex ?? i) + 1), 0)
