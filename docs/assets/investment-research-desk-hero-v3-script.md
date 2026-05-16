# Investment Research Desk Hero V3 Script

## Direction
A major redesign that moves away from the busier finance-forward v2 and toward a cleaner **finance / bank / professional** visual language.

## Goals
- Lower information density than v2
- Keep the motion atmospheric and premium
- Make the product feel clearly financial, but not like a noisy retail-trading dashboard
- Use fewer visual objects, larger shapes, stronger hierarchy
- Preserve the idea of research assistance, not execution

## Storyboard

### Phase 1 — Institutional brand reveal
- Dark blue / slate / steel finance palette
- Clean bank-grade motion, calm and controlled
- Thin market grid and distant pricing traces
- `Investment Research Desk` resolves with understated authority

### Phase 2 — Professional market structure field
- One main finance surface dominates
- Visible but restrained:
  - K-line structure
  - one primary price curve
  - one macro / yield-like curve
  - subtle ticker references
- Feels closer to institutional research / market intelligence than consumer trading UI

### Phase 3 — Research convergence
- Market structure contracts into one refined research board
- Only a few key outputs remain visible:
  - directional view
  - risk / confidence
  - research context only
  - final context / trace / metrics
- End with calm authority, not visual overload

## Export constraints
- Use `scripts/render_true30_gif.cjs`
- GIF-safe mode should avoid shimmer, blur, grain, and dense microtext
- Asset names:
  - `docs/assets/investment-research-desk-hero-v3.html`
  - `docs/assets/investment-research-desk-hero-v3.gif`
