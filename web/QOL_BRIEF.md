# FilterTool web: QoL pass (brief for Claude Code)

> **How to use this file:** start Claude Code at the repo root (`FilterTool/`) and say:
> *"Read `web/QOL_BRIEF.md`. Go through the Open Questions with me before writing code, then implement in the milestone order."*
> Juan can pre-answer any question inline by ticking a box / writing under it.

---

## 0a. Baseline update (2026-09-28): read this before the rest

This brief was written against the Pyodide app (`d4e0a4a`). Since then `main` merged PRs #4/#5
(`9a60cb3`, `61e2076`, `11970bd`) and the `web-qol` branch was rebuilt on top of `4607f77`.
What changed underneath, and what it means for the sections below:

- **Engine:** Pyodide is gone. Design maths lives in the Rust crate `crates/filter-engine`, loaded as
  WASM by `web/src/worker/wasm.worker.js` (API typed in `web/src/lib/engine-api.ts`). `web/public/python/`
  no longer exists. Golden fixtures captured from the old Pyodide engine: `npm run fixtures:test`.
  Treat `crates/` like the PyQt code: don't modify it from this pass, and flag it if a change seems unavoidable.
  Worker calls no longer go through a Python runtime, which makes the Q5 decision below easy: stage/cascade
  responses stay in the worker, no JS port.
- **Already done upstream:** B1 (real `app.css` with light + dark tokens; use `var(--…)` and theme-aware
  Plotly colours), B4 (favicons), E2 Save/Load (`.ftjson`, `lib/design-io.js`; autosave and URL hash are *not*
  done), the T7 `each_key_duplicate` crash (keyed by value+index; the selection bug remained, see M1).
  New features to respect: separate **Template** and **Magnitude** tabs, **Step/Impulse** tabs, **Data/Plots
  units** (Hz or rad/s, persisted; form values are in the data unit, engine params are rad/s),
  **Cursor** toggle, colour modes, collapsible sidebar, keep-alive tabs (`visibility: hidden`).
- **Still valid upstream bugs, fixed in M1:** B2 (the engine designs f0+BW from `w0`/`bw` and never computes
  edges, so the template drew arithmetic edges), B3 (the engine requires `bw[0] < bw[1]` with `bw[0]` = inner
  band, i.e. the stopband width for BR), T7 selection-by-value, Stages x-range (B5).
- **T6 FYI corrected:** the Rust `ω→ω0` evaluates H(j·|p₀|) (correct, matches desktop `TFunction.at`). The old
  Pyodide worker's `at(w0*1j)` was the wrong one (a lone BP biquad peaked at about +30 dB); now it's −0.002 dB.

Constraints 2 and 5 below are obsolete (kept for history). Constraint 3's CRLF noise was not present.

## 0. Hard constraints (read first)

1. **Touch only `web/`.** Do not modify `src/`, `designer/`, `main.py`, `requirements.txt`, `export.*`, `README.md` (root), `venv/`, or anything the PyQt app uses. It's the same repo, so be explicit about paths every time.
2. **Leave `web/public/python/*.py` alone if you can.** These are web-only copies of `src/package/*`, and some already differ from the PyQt versions. Editing them won't break PyQt, but the goal is to keep the design maths identical between the two apps. Do new logic in JS or in the worker's inline Python strings (`web/src/worker/pyodide.worker.js`). If a `.py` change really can't be avoided, flag it first.
3. **Git hygiene:** `git status` already shows ~74 modified files (root + `designer/` + `src/` + `web/`). They're **CRLF-only noise**: `git diff --ignore-cr-at-eol` is empty. Don't stage or commit them. Stage by explicit path (`git add web/src/...`), never `git add -A` / `git add .`. Work on a branch (suggest `web-qol`).
4. **Line endings:** files in the working tree are CRLF. Keep them that way on files you edit, so the diffs stay readable.
5. Pyodide loads from `cdn.jsdelivr.net` at runtime, so testing needs internet. `vite.config.js` sets COOP/COEP headers for the dev server, and `base` is `/TC2-FilterTool/`.

---

## 1. Architecture snapshot

Svelte 5 (legacy `$:` / `export let` syntax throughout) + Vite 8 + `plotly.js-dist` + Pyodide 0.27.5 in a Web Worker via Comlink.

```
web/src/
  App.svelte                 header + Sidebar + TabBar + active tab; boots worker (api.init)
  main.js                    mounts App, imports app.css  ← Vite boilerplate, see Bug B1
  stores/app.js              all shared state (filterParams, filterResult, bodeData, stages,
                             comparisons, bodePoints, remainingPZ, pzKey())
  lib/worker-client.js       singleton Comlink wrap
  lib/approx.js              APPROX_NAMES/COLORS, freqRangeFromParams()
  worker/pyodide.worker.js   init, filterDesign(params), computeBode(num,den,fmin,fmax,n),
                             buildStageFromZPK(z,p,k,normtype,filterType), parseDataset()
  components/
    Sidebar.svelte           FilterPanel + ComparePanel + 2 "Phase 7" placeholders
    FilterPanel.svelte       the whole design form; form state is LOCAL `let`s; design()
    SciInput.svelte          SI-prefix text input (2k2, 4.7n, …), arrow-key nudge
    ComparePanel.svelte      re-designs other approximations whenever $filterParams changes
    BodePlot.svelte          generic Plotly wrapper; Plotly.react on every afterUpdate
    tabs/MagnitudeTab        |H| + template rectangles (from $filterParams at design time)
    tabs/PhaseTab, GroupDelayTab
    tabs/PoleZeroTab         PZ plot + checkbox list + "Norm." select + Add Stage + stage list
    tabs/StagesTab           per-stage |H| + cascade (Bode computed in Pyodide)
  unused boilerplate: lib/Counter.svelte, components/SciInputTest.svelte,
                      components/PlotPanel.svelte, assets/{hero.png,svelte.svg,vite.svg}
```

**Data flow today:** FilterPanel `design()` builds `params` (Hz→rad/s), calls `api.filterDesign` → `filterResult {zeros,poles,num,den,gain,N}` → `api.computeBode` → `bodeData`. It also **clears `stages` and `comparisons`** every time (FilterPanel.svelte ~L50). The template is drawn from `$filterParams`, which is the *snapshot of the last design*, not the live form.

Python side: `AnalogFilter(**params).validate()` does everything (order selection, denorm %, LP/HP/BP/BR transform). `denorm` is 0–100 and interpolates the normalization between the passband edge and the stopband edge (`compute_denormalized_parameters`).

---

## 2. Juan's task list: analysis and suggested approach

### T1. Modernize the design form inputs
The form uses `<select>` for type and approximation, plain `<input type=number>` for N min/max, and `SciInput` text fields for fp/fa/ripple/att. Functional, but dated.

Suggested direction (see Q7 for style choice):
- **Filter type** → segmented control with tiny response-shape glyphs (LP/HP/BP/BR/GD).
- **Approximation** → chip grid with the approximation's plot color swatch (reuse `APPROX_COLORS`, so the colors match the plots).
- **Order** → one compact "N: [min] – [max]" control with −/+ steppers and wheel support, plus an "N = X" readout after design.
- **Numeric fields** (fp, fa, f0, BW, ripple, att, gain, τ0…) → keep the SciInput parser, but add mouse-wheel adjust, drag-to-scrub on the label, floating label + unit suffix, and inline range/validation hints. Give each field a small colored marker that matches its template handle on the plot. Hovering the field should highlight that edge on the plot, and hovering the edge should highlight the field.
- **Validation in the form:** show human messages ("fp must be < fa for low-pass") before calling Python. Today the error shown is `result.error.split('\n').at(-2)`, which is a raw traceback line like `Filter.py - assert self.wp < self.wa`.
- Group the form into "Specs" (type, approx, order), "Template" (freqs, ripple, att) and "Output" (gain, denorm, plot points).

### T2. Template updates live from the form (filter recalculated only on Design)
- Lift the form state out of FilterPanel's local `let`s into a store (e.g. `designForm` in `stores/app.js`). Move `buildParams()` into a pure lib function (e.g. `lib/params.js`) so MagnitudeTab, the drag handles and FilterPanel all share it.
- MagnitudeTab draws the template from the **live** form, not `$filterParams`.
- **Draw the template even before the first design.** The current Y range comes from `bodeData`; without data, fall back to something like `[-(att+20), +5]` dB and an x range from the form.
- Show a "design out of date" indicator (badge on the Design button, or a dimmed curve) when the live form ≠ `$filterParams`.
- Must also fix Bug B2 (BP/BR f0+BW edges), otherwise the live template will be visibly wrong.

### T3. Denormalization slider recalculates the filter live
- On slider `input`, re-run `filterDesign` + `computeBode` with **latest-wins throttling**: if a request is in flight, remember only the newest value and fire it when the current one finishes. Don't queue every tick. Pyodide calls are serialized in one worker anyway.
- Only `denorm` changes. The other params stay as last designed. (Or should a live denorm also pick up the other pending form edits? See Q2.)
- Watch the side effects: `design()` clears stages; ComparePanel re-designs **every** compared approximation on every `$filterParams` change, which is slow while dragging. Suggest comparisons update on slider `change` (release) only. See Q2 and Q9.

### T4. Drag the template limits on the magnitude plot
- Handles: LP has the passband corner (fp, −ripple) and the stopband corner (fa, −att). Vertical edges move frequency only (`ew-resize`), horizontal edges move dB only (`ns-resize`), corners move both. HP mirrors LP. BP/BR have 4 frequency edges + 2 dB levels. In f0+BW mode also add an f0 handle (dashed vertical line) that moves the whole band. GD could get τ0/γ/frg handles on the Group Delay tab (optional).
- "Clear UI": visible grip markers at corners, edge highlight + cursor change on hover, a floating value label while dragging ("fp = 1.23 kHz"), and the form field flashing or updating in sync. Enforce constraints while dragging (fp < fa, fp1 < fp2, ripple < att…) by clamping, not by throwing errors.
- Implementation options:
  - **(a) Custom pointer layer (recommended):** a capture-phase `pointerdown` listener on the plot container that hit-tests template edges in pixel space and calls `stopPropagation()` so Plotly's zoom/pan doesn't start. Convert with `gd._fullLayout.xaxis` / `yaxis` (`l2p`/`p2l`; on log axes the "linear" value is log10, so verify) plus `gd._fullLayout._size` offsets. Write to the `designForm` store and update shapes via `Plotly.relayout(gd, {shapes})` inside rAF.
  - (b) Plotly `editable` shapes + `plotly_relayout`: less code, but rectangles resize from every edge/corner and can be translated freely. Constraining that is awkward.
- `BodePlot.svelte` currently does a full `Plotly.react` on every `afterUpdate`. With 2000–10000 points that's OK for occasional updates but not per mousemove. Either give BodePlot a `shapes`-only fast path or pass shapes separately.

### T5. Stages tab: hovering a stage highlights it on the PZ graph
- The Stages tab has no PZ graph today (the PZ plot lives only in PoleZeroTab). Extract the Plotly PZ rendering from `PoleZeroTab.svelte` (`buildTraces`, `mkX`, `mkO`, layout) into a reusable `PzMap.svelte`.
- New Stages layout (see Q8): Bode on the left; on the right, a PZ mini-map above a list of stage cards.
- Add a shared `hoveredStageId` store. Hovering a stage card or its Bode curve highlights that stage's poles/zeros on every PZ map, bolds its curve and dims the others. Hovering a pole on the map highlights its stage (the reverse direction). Neandertool does the same thing: `renderStageCard` mouseenter/mouseleave → `currentlyHoveredStageId`, and PZMapManager.render colors the hovered stage.

### T6. Stage normalization isn't obvious
Current UX: one "Norm." `<select>` with `Passband | ω→0 | ω→∞ | ω→ω0` above "Add Stage". It's chosen *before* building, isn't stored on the stage, can't be changed afterwards, and gives no feedback. "Passband" silently resolves per filter type in the worker (LP/BR/GD→ω→0, HP→ω→∞, BP→ω→ω0).

Suggestions (see Q6):
- Store `normtype` per stage and make it editable from the stage card after building.
- Use descriptive labels: "Auto (passband): unity gain at DC for this LP", "Unity at DC (ω→0)", "Unity at HF (ω→∞)", "Unity at |p| (ω→ω0)". Show the resolved choice for "Auto".
- Mark the normalization point on the stage Bode curve (a dot at 0 dB at the reference frequency), and show the resulting stage gain (k, dB).
- Show a "cascade gain vs target" readout (e.g. "cascade passband gain −3.01 dB, target 0 dB, Δ −3.01 dB").
- FYI, don't fix: the worker's ω→ω0 evaluates `at(w0*1j)` (correct). PyQt's `Filter.addStage` evaluates `temp_tf.at(np.abs(p_arr[0]))` on the real axis. That's a difference between the two apps. Leave PyQt alone and mention it to Juan.

### T7. Duplicate zeros (e.g. several at 0 Hz): selecting one selects all, and the stage can't be built
**Root cause:** poles/zeros are identified **by value** (`pzKey([r,i])` → `"r.toFixed(10),i.toFixed(10)"`).
- `stores/app.js` `remainingPZ`: builds a Set of used keys → using one zero at 0 marks *all* zeros at 0 as used.
- `PoleZeroTab.svelte` `toggleKey`, `selectedKeys`, `selectedZeros` all use value keys → selecting one selects every duplicate. `selectedZeros` then contains all N zeros at 0, which can make the stage improper (more zeros than poles).
- `{#each … as z (pzKey(z))}` (PoleZeroTab ~L234/L249) → **duplicate keys in a keyed each**. Svelte 5 throws `each_key_duplicate` here, which is probably the "can't build" symptom.
- BP always produces duplicates (N zeros at s=0 from the LP→BP transform); BR produces repeated ±jω0 zeros.

**Fix:** give each root a stable identity at design time, e.g. `{id:'z3', re, im, conjId}`. Pair conjugates once (greedy nearest match among unpaired items) so a complex pair toggles together and real roots toggle one at a time. Stages store ids; `remainingPZ` becomes a multiset by id. Update Pole-Zero tab, stage builder, PZ maps and Save/Load (if added) to use ids.

### T8. Drag stages (gain and frequency) with real-time results, plus Reset
- Neandertool model (`setupCurveDrag`): grab the curve; vertical drag = gain (dB per pixel from the axis scale), horizontal drag = multiply the stage's frequencies by `f(x)/f(x0)` in log space. For a stage that means scaling all its poles **and** zeros by the ratio, so the shape and Q are preserved.
- Optional (see Q4): drag poles directly on the PZ map (neandertool `setupPoleDrag`): the conjugate follows, the pole is clamped to the LHP, it snaps to the real axis within a few px, and Q is capped.
- **Real-time means JS, not Pyodide.** Stages are ≤2nd order. Evaluating H(jω) from zpk at 2000 points in JS is microseconds and avoids a worker round-trip per mousemove. Add a small `lib/tf.js`: `polyFromRoots`, `evalZPK(z,p,k,ω)` → mag/phase/group delay, and the normalization-gain logic ported from `buildStageFromZPK`. Then the Stages tab (per-stage curves + cascade) can be computed in JS. Keep Pyodide for filter design. (See Q5.)
- Per-stage **Reset** restores the original `{zeros, poles, gain, normtype}` captured at build time, plus a "Reset all" button. Mark modified stages visibly (badge / italic name).
- Show the original designed filter as a faint reference curve on the Stages plot, so the effect of dragging on the cascade is obvious.
- The PZ map should show moved roots in their new position, with a ghost at the original (see Q4).

---

## 3. Bugs found along the way (fix regardless)

- **B1: Vite boilerplate CSS is still active.** `src/app.css` (imported in `main.js`) sets `#app { width: 1126px; margin: 0 auto; text-align: center; … }` and `:root { font: 18px/145% … }`. Verified in Chromium at a 1600px viewport: `#app` is **1126px wide and centered**, and the root font size is **18px**. `:root` beats App.svelte's `html { font-size:14px }` on specificity, so every `rem` is 18px (16px under 1024px). Fix: remove or replace `app.css` with a real base stylesheet and design tokens (CSS vars for the GitHub-dark palette that's currently hard-coded in every component). **Expect everything to shrink** when rem goes 18→14px, so recalibrate sizes.
- **B2: BP/BR "f0 + BW" template edges are wrong.** `buildParams()` sends `wp = f0 ± BWp/2` (arithmetic), but Python `validate()` (F0_BW) recomputes the edges geometrically: BP uses `wp = w0·(√(1+1/4Q²) ∓ 1/2Q)` with `Q = w0/bw`; BR uses `wa0 = ½(−bw0+√(bw0²+4w0²))`, `wa1 = wa0+bw0`, and the same for wp from bw1. The magnitude tab draws the JS edges, so the drawn template doesn't match what was designed. Compute edges in JS with the same formulas.
- **B3: BR BW mapping looks swapped.** For BR, Python treats `bw[0]` as the **inner (stop)** width and asserts `bw[0] < bw[1]`. The form sends `bw: [BW pass, BW stop]` for both BP and BR, so for BR the user has to type the stop width into "BW pass". Map `bw = [bwStop, bwPass]` for BR (and sanity-check the defaults when switching type).
- **B4: favicon 404.** `index.html` references `/TC2-FilterTool/favicon.png`, but `public/` only has `favicon.svg`.
- **B5: Inconsistent x-ranges between tabs.** The main Bode uses the ranges from `getFreqRange()`, Compare uses `freqRangeFromParams()`, and Stages uses max|pole| ×0.01…×100. Unify them in one function.
- **B6: `each` key duplicates / identity by value.** Covered in T7.
- Minor: a11y warning `aria-disabled` on `<aside>` (Sidebar); unused boilerplate files (see §1); the "Phase 7" placeholder sections in the sidebar are visual noise (collapse or hide them).

---

## 4. Proposed extras (pick in Q1)

- **E1: Template compliance.** Paint the parts of |H| that break the template red, add a pass/fail chip per band, and color the curve green when compliant (neandertool: `meetsConstraints` in `js/game.js`, `PlotManager.render` curve color, sparks on violation). Updates live while dragging the template or stages.
- **E2: Save / Load + autosave.** Make the disabled header buttons work: export/import `{form, filterParams, stages}` as JSON. Autosave to `localStorage` so a reload doesn't lose work. Optionally encode the form in the URL hash so a design can be shared as a link.
- **E3: Click-select on the PZ plot + Auto-stage.** Use `plotly_click` to select poles/zeros on the plot, not only in the checkbox list. Add an "Auto-stage" button that pairs each complex pole pair with the nearest zeros into 2nd-order sections (real leftovers become 1st order), ordered by Q (the PyQt app has `orderStagesBySos` for inspiration; port the idea, don't import it).
- **E4: Stage fine-tuning.** Mouse wheel over a stage card or curve adjusts Q (for complex pairs: `p = −ω0/2Q ± jω0√(1−1/4Q²)`). The stage card shows f0 / Q / gain as editable SciInputs. Add an "Absorb remaining gain" button that puts the cascade-vs-target difference into the last stage (the PyQt `force_gain` behavior). Add stage reordering (drag in the list).
- **E5: Keyboard and plot niceties.** Ctrl/⌘+Enter = Design, Esc cancels a drag and restores the pre-drag value, Del removes the hovered stage. Add a crosshair readout with f and |H| at the cursor and a "reset view" double-click hint.
- **E6: Live mode toggle.** Optionally auto-redesign (debounced, latest-wins) on *any* form change, not just denorm.
- **E7: GD template.** Draw the τ0 ± γ% tolerance up to f_rg on the Group Delay tab, with drag handles like T4.
- **E8: Cleanup.** Delete unused boilerplate, introduce CSS design tokens, fix the Sidebar a11y warning.

---

## 5. Open questions for Juan

Answer these before implementation. Recommended defaults are marked ★.

**Q1. Which extras (§4) go in this pass?**
- [ ] E1 Template compliance ★
- [ ] E2 Save/Load + autosave ★
- [ ] E3 PZ click-select + Auto-stage
- [ ] E4 Stage fine-tuning (Q wheel, editable f0/Q/gain, absorb gain)
- [ ] E5 Keyboard / crosshair
- [ ] E6 Live-mode toggle
- [ ] E7 GD template
- [ ] E8 Cleanup ★

**Q2. Live denorm and existing stages.** Today every design clears stages. When the denorm slider moves and stages exist:
- [ ] ★ Live preview updates the filter; stages are cleared on release, with a non-blocking "stages cleared, undo" toast
- [ ] Ask for confirmation before clearing
- [ ] Try to remap stages onto the new roots (by nearest root / same pairing). Nicer but more complex.

Also: should a live denorm re-design use only the last designed params with the new denorm ★, or also pick up pending (undesigned) form edits?

**Q3. Template drag vs the committed design.** You said the filter isn't recalculated until Design is pressed. OK to show a "design out of date" badge ★? Or should releasing a template drag auto-redesign (like denorm)?

**Q4. Where can stages be dragged?**
- [ ] ★ Stages tab Bode: drag a stage curve (vertical = gain, horizontal = frequency scale; poles and zeros scaled together)
- [ ] Also drag poles/zeros on the PZ map (conjugate follows, LHP clamp, real-axis snap)
- [ ] Both

And after a stage is moved, should the Pole-Zero tab show the moved roots (with a ghost at the original) ★, or always the designed roots?

**Q5. Compute stage responses in JS instead of Pyodide?** ★ Yes for stages and the cascade, needed for real-time dragging; Pyodide stays for design. This duplicates the stage-normalization maths in JS (small, and `buildStageFromZPK` can remain as a cross-check). OK?

**Q6. Normalization UX.**
- [ ] ★ Per-stage normalization, editable after building, descriptive labels, 0 dB reference marker, cascade-vs-target gain readout
- [ ] One global "normalize stages to…" setting instead
- [ ] Keep the pre-build selector, just relabel it and explain it

**Q7. Input style direction for the form.**
- [ ] ★ Segmented type control + approximation chips + "scrub" numeric fields (wheel, drag-label, arrows), no sliders except denorm
- [ ] Neandertool-like: value + slider per parameter (auto-ranging sliders), wheel to adjust
- [ ] Keep the current layout, restyle only

**Q8. Stages tab layout.**
- [ ] ★ Bode on the left; right column with a PZ mini-map above the stage cards
- [ ] Bode on top, PZ + cards below
- [ ] Merge Pole-Zero and Stages into one "Stages" workspace (builder + cards + Bode)

**Q9. Compare panel during live changes.** ★ Recompute comparisons only on release / Design (not per slider tick)? Or keep them fully live?

**Q10. Housekeeping.** Branch name (`web-qol` ★)? One commit per milestone ★, or one squashed commit? OK to delete the unused boilerplate files ★?

---

## 5b. Decisions (answered 2026-09-28)

- **Q1 extras:** E1, E3, E4, E5, E6, E7, E8. **E2 (Save/Load) is out.** (Save/Load since landed upstream; keep it working: stages now save `normtype` and re-attach to root ids on load.)
- **Q2:** remap existing stages onto the new roots after a denorm change (nearest root / same pairing). Put it behind a flag so it can fall back to "clear on release + undo toast" if performance suffers.
- **Q2b:** live denorm uses the last-designed params + the new denorm.
- **Q3:** releasing a template drag **auto-redesigns** (like denorm). No "out of date" badge needed for drags. It's still useful when live mode (E6) is off and plain form edits are pending.
- **Q4:** drag on **both** the Stages Bode curve (vertical = gain, horizontal = freq scale) **and** the PZ map (conjugate follows, LHP clamp, real-axis snap, Q cap). **Mouse wheel over a pole changes only its Q** (ω0 fixed).
- **Q4b:** the PZ tab shows the moved roots, with a ghost at the original.
- **Q5:** keep the engine worker for stage/cascade responses (no JS port of the maths). Drags use latest-wins throttling on worker calls. (Answered as "keep Pyodide"; the engine is now WASM, which only makes this cheaper.)
- **Q6:** per-stage normalization, editable on the card, descriptive labels (resolved "Auto"), 0 dB marker, cascade-vs-target readout.
- **Q7:** segmented type + approximation chips + scrub numeric fields. **N min/max is a range slider (two thumbs)**. After design, mark where the chosen N landed on that slider.
- **Q8:** Bode on the left; right column with the PZ mini-map above the stage cards.
- **Q9:** comparisons recompute only on release / Design.
- **Q10:** branch `web-qol`, one commit per milestone, delete the unused boilerplate.

## 5c. Status (2026-09-28): all milestones done on `web-qol`

| Milestone | Commits | Notes |
|---|---|---|
| M1 Foundations | `966d1ef` | B2 / B3 / T7 / Stages x-range on the WASM baseline; B1, B4, E2 were done upstream |
| M2 Form | `0f9f14a`, `14e2a0b`…`255c30d` | Segmented type, approximation preview tiles (engine-drawn sketches, see `lib/approx-sketches.js`), two-thumb N slider, scrub fields, inline validation |
| M3 Live template + drag | `a851321` | Handles, compliance chips / red segments (E1), form ↔ plot hover link |
| M4 Live denorm | `e33bc50`, `eff3a05` | Slider and curve drag; stages remapped across redesigns (`REMAP_STAGES` flag); comparisons on release |
| M5 Stages workspace | `ca2cdee`, `368df61`, `80c45f5` | T5 / T6 / T8, E3 (click-select, Auto-stage), E4 (f0 / Q / gain on cards, absorb Δ, reorder) |
| M6 Extras | see `git log` | E5 shortcuts, E6 live mode, E7 GD template, E8 cleanup |

Deviations from the brief worth knowing:
- Curve drag for denorm maps pointer travel across the transition band to 0–100 % (a 1:1 follow would be ~5 px for the whole range).
- Stage roots are draggable on both PZ maps; wheel-Q works on poles and stage curves, not on cards (it would fight card-list scrolling).
- A redesign keeps stage gain offsets / normalization but resets moved roots to the new design (with a notice).
- E5 crosshair: upstream's Cursor toggle already gives the readout; Plotly double-click resets the view.
- Q5 revisited: stage drags on the Stages tab are previewed from JS (`lib/stage-eval.js`, matches the engine to ~1e-13 dB) on a canvas overlay, neandertool-style; the engine still rebuilds once on release. Drag latency went from ~75 ms to ~21 ms (p50).
- Live denorm (slider and Template curve drag) on the Template / Magnitude tabs: each step only fetches poles / zeros from the engine and draws |H| on a canvas (ghosted plot underneath); one real design on release. ~70–99 ms → ~21–24 ms p50. Other tabs keep the per-step redesign. Main-thread WASM was rejected: Legendre N=15 takes ~340 ms per design.

## 6. Suggested milestones

1. **Foundations** (done, rebuilt on the WASM baseline): `lib/params.js` (buildParams/formFromParams incl. B2/B3, validation mirroring the engine, unit rescale), `designForm` store, root identity model (T7, incl. Save/Load), Stages x-range (B5). B1 was fixed upstream; `lib/tf.js` dropped (Q5).
2. **Form redesign:** T1, inline validation, error messages.
3. **Live template + drag:** T2, T4 (+ E1 if chosen).
4. **Live denorm:** T3 with latest-wins throttling (+ Q2/Q9 behavior).
5. **Stages workspace:** `PzMap.svelte` extraction, T5 hover linking, T6 normalization UX, T8 drag + reset (+ E3/E4 if chosen).
6. **Extras and cleanup:** E2, E5–E8 as chosen, B4, a11y.

## 7. Verification

- `cd web && npm run dev`, then open `http://localhost:5173/TC2-FilterTool/`. `npm run build` must pass without new warnings, and `npm run fixtures:test` must pass.
- Manual matrix: LP / HP / BP (f0+BW and Freqs) / BR (both) / GD × Butterworth, Cheb I, Cheb II, Cauer, Bessel.
  - Check that the template edges match the designed filter (B2).
  - Build stages from a BP filter with several zeros at 0 (T7), e.g. Butterworth BP, N=4 → 4 zeros at 0.
  - Drag every template handle; drag stages; press Reset.
  - Move the denorm slider with and without stages.
- Cross-check the JS stage normalization against the worker's `buildStageFromZPK` for a few cases (same gain to ~1e-9).
- Confirm `git status` shows changes **only under `web/`** (ignoring the pre-existing CRLF noise), and that `git diff --stat -- src designer main.py` is empty.

## 8. Neandertool references (github.com/TC-II/neandertool, `js/ui.js`)

- `UIManager.setupCurveDrag()`: curve hit-test within 8 px, vertical = global gain, horizontal = scale all f0 by the log-x ratio, window-level mousemove/mouseup.
- `UIManager.setupPoleDrag()`: pick a pole, the conjugate follows, LHP clamp (`re ≤ −MIN_F0`), real-axis snap (6 px), Q cap 20, and two-distinct-real-poles handling (the grabbed pole moves, the other stays).
- `UIManager.renderStageCard()`: card hover → `currentlyHoveredStageId` → PZ map colors that stage, and the magnitude plot draws the hovered stage as a dashed gray curve under the main one.
- `UIManager.createParameterControl()`: wheel-to-adjust with a user "sensitivity" multiplier, and auto-ranging sliders that re-center on release (`param.autoRange()` in `js/dsp.js`).
- `PlotManager.drawConstraints()` / `render()`: forbidden regions shaded, and edges lying on the plot border aren't outlined (so only real limits look like limits). The curve changes color when `meetsConstraints()` (`js/game.js`) passes.
- Visual style is retro pixel-art. Take the interaction ideas, not the look; keep FilterTool's current dark GitHub-like theme.
