# Default Masking Regression — Causality Report

**Date:** 2026-04-04  
**Status:** Active investigation  
**Preset:** `default` only  
**Clip family:** `deskTest`  
**Purpose:** Separate the likely historical trigger from the current live bug and the secondary artifact amplifier.

---

## Executive Summary

The current Default preset masking failures are best explained by a **three-part causal chain**, not a single regression:

1. **Historical trigger:** the performance/quality pass introduced Pass 1 direction-estimation changes that made the synthetic pipeline more fragile.
2. **Current live bug:** SAM2 prompt-frame selection is broken and effectively prompts on frame `0` every time.
3. **Secondary amplifier:** fisheye-to-ERP backprojection still has a known sampling weakness that turns a weak synthetic track into hollow or stippled ERP masks.

The important conclusion is:

- The optimization pass likely **triggered or exposed** the regression family.
- But the current failures are **not fully explained** by the old `512px + single-best-box` changes, because those changes are already reverted in code.
- The strongest current bug is the **SAM2 prompt-frame-selection bug**.

---

## What Changed During The Optimization Pass

Two Pass 1 changes were introduced as part of the optimization/quality work:

1. Detection resolution lowered from `min(1024, erp_w // 4)` to `min(512, erp_w // 4)`
2. Union-box direction estimation replaced with highest-confidence single-box direction

Those changes are documented in:

- [2026-04-04-direction-estimation-regression.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-direction-estimation-regression.md)
- [2026-04-04-performance-optimization-results.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-performance-optimization-results.md)

Those changes plausibly made the synthetic fisheye framing less stable by making direction estimation noisier.

That matters because the Default preset depends on accurate synthetic look-at direction for:

- person centering in the synthetic fisheye
- reliable center-click prompting for SAM2
- stable propagation over time
- clean fisheye-to-ERP sampling during backprojection

So the user's instinct is directionally right: **the optimization pass likely contributed to the failure pattern appearing.**

---

## What The Current Code Says Now

The key point is that the two Pass 1 changes above are already reverted:

- detection size is back to `min(1024, erp_w // 4)` in [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L837) and [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1341)
- union-box direction is back in [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1090)

So if masks are still badly broken **after those reversions**, then the optimization-era direction regression is not the whole current explanation.

That points to a second active bug.

---

## The Strongest Current Live Bug

The strongest current live bug is documented in:

- [2026-04-04-sam2-prompt-frame-selection-bug.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-sam2-prompt-frame-selection-bug.md)

### The actual logic problem

Pass 1 now returns an empty ERP mask because Pass 2 is authoritative for mask shape:

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1121)

But `_synthetic_pass()` still tries to choose the best prompt frame by primary-mask area:

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1154)

That means:

- every `mask.sum()` is `0`
- `best_idx` never meaningfully advances
- the SAM2 prompt frame effectively stays at frame `0`

That prompt frame is then passed directly into the tracker:

- [core/masker.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/masker.py#L1198)
- [core/backends.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/backends.py#L404)

### Why this matters

SAM2 is being center-click prompted on a single synthetic fisheye frame.

If frame `0` happens to be a weak synthetic view:

- the person may not be well-centered
- the center click may hit desk/floor/wall/furniture instead of the operator
- SAM2 can then propagate the wrong object through the whole sequence

This is a much stronger explanation for the observed behavior than a tiny residual Pass 1 precision shift.

It also explains why:

- some runs used to look fine
- some frames are correct while others are wildly wrong
- failures correlate with temporal distance from the prompt frame

---

## The Secondary Amplifier

Even with a bad synthetic track, the ERP mask quality is made worse by the known backprojection weakness documented in:

- [2026-04-04-backprojection-sampling-artifact.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-backprojection-sampling-artifact.md)

That artifact is real and still relevant:

- off-center synthetic fisheye masks backproject sparsely into ERP
- point sampling creates hollow, dotted, or fragmented operator shapes
- poor synthetic centering makes the artifact much more visible

So the backprojection issue is still important, but it looks more like an **amplifier** than the primary cause of the current regression.

---

## Evidence From The Broken Frames

Two broken frames reinforce this split:

- `D:\Capture\deskTest\default_test2\extracted\masks\deskTest_trim_00004.png`
- `D:\Capture\deskTest\default_test2\extracted\masks\deskTest_trim_00002.png`

### `00004`

This frame looks like a mixed failure:

- operator only weakly captured
- non-person scene structure appears in the mask
- output resembles bad prompting plus projection weakness

This is consistent with the prompt-frame-selection bug.

### `00002`

This frame is more severe:

- head is partially captured
- torso/body are mostly missing
- remaining operator pixels appear as sparse fragments

This frame looks less like a simple direction wobble and more like:

1. weak synthetic framing or wrong SAM2 target upstream
2. followed by backprojection degradation downstream

That is exactly the pattern expected from:

- a bad prompt frame choice
- plus an aliasing-prone fisheye-to-ERP projection

---

## Best Current Causal Model

The most plausible current causal model is:

1. The optimization pass made direction quality more fragile on some clips.
2. That exposed or made visible an existing prompt-frame-selection bug.
3. Because prompt selection is broken, SAM2 often starts from frame `0` instead of the strongest frame.
4. When frame `0` is not a good synthetic prompt view, SAM2 tracks the wrong thing or a weak/partial target.
5. The known backprojection weakness then turns that weak synthetic result into hollow or fragmented ERP masks.

This model fits all three documents together without contradiction.

---

## Confidence Ranking

### High confidence

- The old Pass 1 optimization changes were real and quality-sensitive.
- Those changes are currently reverted.
- Prompt-frame selection is currently broken because it is based on empty masks.

### Medium confidence

- The prompt-frame-selection bug is now the strongest live cause of the observed Default preset failures.
- The backprojection artifact is amplifying the visual severity of those failures.

### Lower confidence

- The optimization pass alone explains the entire current failure pattern.

It likely contributed historically, but the current code evidence points to a more specific live bug.

---

## What This Means Practically

The investigation should not treat this as:

- "just revert the optimization pass harder"

or as:

- "just fix backprojection sampling"

Instead, the most disciplined framing is:

1. **Fix prompt-frame selection first**
2. **Validate Default preset mask quality again**
3. **Then decide how much residual damage is still due to backprojection sampling**

If prompt selection is still wrong, backprojection work may only treat symptoms.

If backprojection is fixed first, SAM2 may still be tracking the wrong thing cleanly.

---

## Recommended Next Diagnostic Order

### 1. Treat prompt-frame selection as the primary active bug

The immediate question should be:

- is SAM2 being prompted on a weak synthetic frame when stronger frames exist?

The current evidence says yes.

### 2. Inspect synthetic intermediates on a known bad frame family

For the same run, inspect:

- synthetic fisheye image for the chosen prompt frame
- SAM2-tracked synthetic mask on that frame
- SAM2-tracked synthetic masks on nearby good/bad frames
- final ERP mask after backprojection

That will separate:

- wrong-target tracking upstream
- sampling loss downstream

### 3. Keep the direction regression note as historical context, not the only explanation

That note still matters, but it should be treated as:

- the likely trigger

not as:

- the complete present-tense root cause

---

## Recommended Wording Going Forward

The cleanest short summary is:

> The Default preset masking regression appears to be a compounded failure. The optimization pass likely made direction estimation more fragile and exposed the issue, but the strongest current live bug is that SAM2 prompt-frame selection still uses empty Pass 1 masks and therefore effectively prompts on frame 0 every time. The known fisheye-to-ERP backprojection artifact then amplifies those weak synthetic results into hollow or fragmented ERP masks.

---

## Bottom Line

The user's instinct was valid: **something from the optimization era likely set this off.**

But the best current code-level explanation is more precise:

- the optimization-era direction changes were the likely trigger
- the prompt-frame-selection bug is the strongest current live bug
- the backprojection artifact is the secondary amplifier

That is the current best-fit explanation for why Default preset masks are still broken even after the obvious Pass 1 reversions.
