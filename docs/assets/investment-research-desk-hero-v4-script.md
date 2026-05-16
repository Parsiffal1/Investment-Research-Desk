# Investment Research Desk Hero V4 Script

## Goal
Replace the current README hero with a version that is easier to remember after a quick GitHub glance.

This version should answer three questions almost instantly:
1. Is this a finance product?
2. Is it a research/support product rather than an execution tool?
3. What core mental image remains after the GIF ends?

## User-feedback constraints
- Restore the small bottom positioning line in phase 1.
- Phase 2 should still show clear financial motifs, especially K-line / market structure.
- Phase 3 must **not** become a dashboard or full board.
- Phase transitions should feel designed, not like hard cuts.
- Avoid flicker, low-frame-rate feel, and palette churn.

## Memory-point test
After a first-time viewer sees the GIF once, the intended retained fragments are:
- dark institutional finance mood
- the name `Investment Research Desk`
- a calm K-line / price-structure scene
- `macro`, `technical`, and `sentiment` collapsing into one `context` point
- this is a research-support product, not an execution tool

If those five fragments survive, the GIF succeeds.

## Product inference test
A new GitHub visitor should be able to guess:
- this is for **equities and crypto research**
- it helps interpret market structure, macro pressure, and other signals
- it is a **financial research assistant / workflow**, not a broker or bot

## Visual direction
- More finance / bank / professional than v3
- Lower information density than v2 and v3 phase 3
- Keep the dark premium tone
- Use fewer, larger, more legible forms

## Storyboard

### Phase 1 — Institutional brand reveal
- Deep finance palette: graphite, slate, muted steel blue, restrained warm accent.
- Thin dormant market traces and a small footer line at the bottom.
- `Investment Research Desk` resolves calmly.
- Bottom line reinforces positioning:
  - `Local-first financial research workflow · research context only`

### Phase 2 — Market structure field
- One dominant finance surface appears.
- Core motifs only:
  - 5–7 oversized candlesticks
  - one main price path
  - one macro / yield-like curve
  - one sparse ticker rail
- Two restrained callouts only:
  - `K-line structure`
  - `Macro pressure`
- No extra beams, extra charts, or crowded overlays.

### Phase 3 — Signal convergence memory point
- The chart does not turn into a dashboard.
- Instead:
  - candles simplify to 3 key bars
  - price path compresses toward a focal point
  - macro curve bends inward
  - three labels appear as elemental cues:
    - `macro`
    - `technical`
    - `sentiment`
  - all converge into one illuminated `context` ring / crosshair
- Brand returns quietly and the final footer reinforces that this is research support, not execution.

## Transition design
Transitions must feel smooth and premium.

### Transition A: brand -> market field
- Use the same horizontal traces from phase 1 and extend them into chart axes / price paths.
- Fade the title upward while the chart field rises from the same center region.
- Keep background constant to avoid flicker.

### Transition B: market field -> convergence
- Candles and price line contract inward instead of disappearing.
- Macro curve bends and narrows toward the focal point.
- Ticker rail fades first.
- Large shapes shrink and dim into one crosshair / pulse lock.

## Export safety rules
- Use `scripts/render_true30_gif.cjs`
- `window.__gifExport` mode must avoid:
  - animated grain
  - blur-heavy glassmorphism
  - rapid contrast changes
  - crowded microtext
- Keep stable background and limited palette.

## New asset names
- `docs/assets/investment-research-desk-hero-v4.html`
- `docs/assets/investment-research-desk-hero-v4.gif`
