# Taste Judge Rubric (v0.1)

This rubric defines a 5-axis, 1-5 score for computer-use (CU) agent
trajectories. It is meant to complement (not replace) the binary verifier
reward used in our RL loop. Verifier captures *did the goal happen*; this
rubric captures *was the path good*.

A trajectory is the tuple `(task, actions[], outcome, screenshots?)`. Each
action is a JSON-ish dict like
`{"type": "click"|"type"|"key"|"scroll"|"navigate"|"wait", "target": "...", "text": "...", ...}`.

## Axes

### 1. intent_clarity
**Definition.** For each action in the trajectory, is its purpose obvious in
the context of the user-stated task? Does the path through the UI map onto
sub-goals you would expect a human to take? This is *means-end* clarity, not
correctness.

- **1 (bad).** Half the actions have no clear purpose — random clicks, opening
  unrelated tabs, typing into a search box for an unrelated query, or "let me
  click around to see what happens" behavior.
- **3 (mediocre).** Most actions are interpretable, but 1-3 are head-scratchers
  (e.g., agent visits Settings before Compose for an email task with no
  apparent reason, then comes back).
- **5 (excellent).** Every action lines up with a sub-goal that a competent
  human reviewer can name in one short phrase: "open compose", "address it",
  "type subject", "send".

### 2. efficiency
**Definition.** Action count relative to the shortest reasonable path. Penalize
duplicate clicks, redundant navigations, and unnecessary scrolling. *Don't*
penalize legitimate verification steps (e.g., scrolling once to confirm a
selection).

- **1 (bad).** >2x the optimal action count, with obvious wasted motion (10+
  scrolls when the target is a button on screen; the same field clicked 5
  times).
- **3 (mediocre).** Roughly 1.3-1.7x optimal. Some redundancy (double-clicks
  where one would do, an extra navigation step) but trajectory still
  progresses.
- **5 (excellent).** Within ~1.1x of optimal. Every action moves the world
  state forward.

### 3. recovery
**Definition.** When the agent encounters something unexpected — a popup, a
loading spinner, an error toast, a layout change — does it diagnose the new
state and adjust, or does it blindly retry the same action? If nothing
unexpected happens, this axis defaults to 4 (we lacked signal). Mark 5 only if
*we saw* a graceful recovery.

- **1 (bad).** Agent retries the same failing action 3+ times without
  acknowledging the changed state (e.g., clicks the same now-disabled button
  repeatedly; types into a closed input).
- **3 (mediocre).** Agent recovers eventually but only after some thrashing,
  or recovers from one issue but is fragile to a second.
- **5 (excellent).** Encounters at least one unexpected event and pivots
  cleanly within 1-2 actions: dismisses popup, waits for spinner, retries
  with a corrected selector.

### 4. ui_grounding
**Definition.** Do clicks land on plausible target elements given visible UI
state? Do typed values match the field type and constraints? This is the
"the agent actually sees the screen" axis. Heavy reliance on screenshots when
provided.

- **1 (bad).** Clicks empty space or off-screen coordinates; types a date into
  a phone-number field; selects items by index that doesn't exist; ignores a
  blocking modal.
- **3 (mediocre).** Most clicks hit reasonable targets, but at least one
  obvious miss (clicks the link's row but not the link; types in the wrong
  field) that the agent then has to fix.
- **5 (excellent).** Every click and type is on a plausibly visible, enabled,
  correctly-typed element. No "ghost" interactions.

### 5. coherence
**Definition.** Reading the trajectory top-to-bottom, does it feel like *one
mind pursuing one plan*, or like noise / context-switching / forgetting? This
captures plan stability over the full sequence.

- **1 (bad).** Plan changes mid-stream with no apparent trigger; agent abandons
  partial work; reopens the same form three times to redo what it already did.
- **3 (mediocre).** Plan is mostly coherent but has one clear "where is it
  going?" stretch — e.g., a 4-action detour that doesn't connect.
- **5 (excellent).** A reviewer can summarize the trajectory in a single
  sentence and the actions all fit that sentence without exception.

## Weighted total

`weighted_total = 0.20*intent_clarity + 0.20*efficiency + 0.20*recovery + 0.25*ui_grounding + 0.15*coherence`

Range: 1.00-5.00.

**Weight justification.**
- `ui_grounding` (0.25) gets the highest weight: when the agent is *not*
  looking at the screen, every other axis becomes meaningless. It is also the
  axis with the most signal per screenshot. Mis-grounded trajectories should
  pull the total down hardest.
- `intent_clarity` and `efficiency` (0.20 each): correlated but not
  identical. Clarity is "did each step make sense?" while efficiency is "was
  the *count* of steps right?" — a trajectory can be clear but bloated, or
  efficient but mysterious.
- `recovery` (0.20): high weight because RL training surfaces lots of
  unexpected-state cases; we want the judge to push the policy toward robust
  recoveries even when the verifier still says success.
- `coherence` (0.15): partially redundant with intent_clarity (correlation
  ~0.6 expected), so down-weighted to avoid double-counting.

Weights sum to 1.00.

## Notes for the judge model
- Score independently per axis; do not let `outcome` bleed into per-axis
  scores. Verifier success can coexist with a 2 on efficiency.
- When in doubt between two adjacent scores, pick the lower one — taste should
  be conservative.
- For `recovery` when no unexpected event occurred: default to 4. Reserve 5
  for trajectories that actually demonstrated recovery.
