# Performance Optimization Results — Response

**Date:** 2026-04-04  
**Report reviewed:** `docs/2026-04-04-performance-optimization-results.md`

---

## Executive Read

The optimization pass looks like a real success.

The most important outcome is not any single timing number. It is the
structural shift in where time is being spent:

- Pass 1 is no longer the dominant problem
- Stage 3 reframing is much healthier
- the remaining hot spot is now the synthetic-mask backprojection

That is exactly what a good optimization pass should do: remove obvious waste,
then expose the next true bottleneck.

So the high-level conclusion is:

- the implemented changes appear worthwhile
- the optimization strategy was sound
- the next round should focus primarily on backprojection

That said, the report is currently mixing measurements from different runs or
different reporting layers, and that makes some of the headline numbers
internally inconsistent. I would fix that before treating this as the canonical
performance record.

---

## Main Findings

### 1. The optimization work itself looks strong

The reported wins line up with the code changes we would expect to help:

- Pass 1 remap caching
- batched YOLO inference
- Stage 3 remap caching
- nearest-neighbor mask remap switched to `cv2.remap`
- highest-confidence box for direction

Conceptually, this is the right shape:

- eliminate repeated geometry math
- reduce per-view Python/GPU-call overhead
- share geometry between image and mask reprojection

The report's conclusion that Pass 1 is no longer the dominant cost sounds
credible.

### 2. The report's numbers are not fully self-consistent

This is the biggest issue in the document.

Examples:

- Masking substage total is reported as **105.4s**
- Full pipeline masking stage is reported as **59.8s**

Those cannot both describe the same 16-frame run.

Likewise:

- Reframer substage total is **30.2s**
- Full pipeline reframe stage is **26.3s**

Again, those cannot both be the same run unless they are being measured under
different conditions.

There is also a frame-count inconsistency:

- the report header says **11-16 frames**
- the masking and reframe tables use **16 frames**
- the full pipeline quality line says **11/11 complete rig frames**

That strongly suggests the report is summarizing multiple runs in one narrative
without labeling them clearly enough.

This does not mean the optimization results are wrong. It means the report
needs explicit run boundaries so readers know which numbers belong together.

### 3. The default-preset change should be separated from performance attribution

The report includes this as item 7:

- default preset changed from Cubemap (6 views) to Default (16 views)

That may be a good product change, but it is not a pure speed optimization.
It changes the amount of work the pipeline does.

So for reporting purposes, I would keep that separate from the optimization
pass. Otherwise future readers may accidentally compare:

- old runs with 6 views
- new runs with 16 views

and interpret the results as if they were directly comparable.

My recommendation:

- keep the preset change in the implementation log
- keep it out of the optimization impact summary

### 4. The report correctly identifies backprojection as the next frontier

Even with the measurement inconsistencies, the directional conclusion is still
clear: backprojection is now the problem to attack.

That is a good sign. It means the earlier work succeeded enough that the
remaining expensive part is now a deeper geometric kernel, not repeated
high-level overhead.

The candidate list is also in roughly the right order of seriousness:

- Approach D is the most compelling if validation passes
- Approach A is the safest fallback
- Approach B has the highest conceptual upside but the most behavioral risk
- Approach C is cheap but probably not transformative

---

## Thoughts On The Candidate Approaches

### Approach D: Pure NumPy equidistant fisheye projection

This is the most interesting next move.

Why it stands out:

- it targets the actual hot path directly
- it works for moving cameras, not just static ones
- it does not depend on the shared-map assumption
- it could potentially simplify both backprojection and synthetic rendering

The key requirement is validation.

Before replacing `pycolmap.img_from_cam(...)`, I would require:

1. A dense numerical comparison against `pycolmap` over representative rays
2. A round-trip sanity check at center, edge, and near-horizon conditions
3. A mask-equivalence comparison on a few real synthetic frames

If those pass, this is probably the strongest next optimization.

### Approach A: Downsampled backprojection

This is the best low-risk fallback.

What I like about it:

- simple to implement
- easy to benchmark
- easy to guard with a config flag or threshold
- can stack with Approach D later

If Approach D proves harder than expected, A is probably the fastest route to a
meaningful wall-clock improvement.

I would treat it as:

- the safest immediate optimization candidate
- especially attractive if you can prove mask-quality equivalence visually and
  numerically

### Approach B: Forward projection

This has the highest theoretical upside after D, but I would not do it next.

The hole-filling / splatting problem is real, and once you move to forward
projection you are no longer just optimizing the same mapping kernel. You are
changing the sampling behavior.

That makes it a good later-stage optimization experiment, not the first thing
I would reach for.

### Approach C: Cache rotation-independent world rays

The report is probably right that this is modest rather than transformative.

Still, there are two reasons it may be worth doing opportunistically:

1. It is low risk
2. It may reduce memory churn as well as arithmetic

I would not prioritize it ahead of D or A, but if the code is being touched
heavily anyway, it is a reasonable companion optimization.

---

## Additional Ideas I Would Add

### 1. Separate "same-math replacement" from "sampling-behavior change"

The candidate list would be easier to reason about if it explicitly grouped:

**Same-math replacements**
- D: replace `pycolmap` projection with equivalent vectorized NumPy math
- C: cache world-ray grid

**Sampling-behavior changes**
- A: downsample ERP backprojection grid
- B: forward projection / scatter

That split matters because the first group is easier to validate for exactness,
while the second group trades some fidelity assumptions for speed.

### 2. Add a validation harness for backprojection before implementation

Before choosing D or A, I would add a tiny benchmark/validation harness that
can compare candidate outputs against the current implementation on:

- one synthetic mask with stable direction
- one synthetic mask with off-center direction
- one real tracked mask from a handheld clip

Metrics:

- IoU vs current backprojection
- changed-pixel count
- runtime

That will keep the next optimization round evidence-based.

### 3. Treat extraction as the next pipeline-wide watch item

Even though this report is mostly about masking and reframing, the full
pipeline table shows extraction at **74.9s**, which is now in the same league
as the remaining masking cost.

So once backprojection is improved, the next "speed everywhere possible"
question may shift to extraction rather than masking.

That is useful because it means the optimization program is working: the
pipeline is becoming balanced enough that other stages now matter.

### 4. Consider using the same math replacement in synthetic rendering later

If Approach D succeeds for backprojection, it may also be worth revisiting
`_render_synthetic_fisheye`.

That path currently relies on `camera.cam_from_img(...)` to convert fisheye
pixels into normalized rays before ERP sampling. Since the synthetic camera is
an ideal equidistant fisheye with zero distortion, there may be a similar
opportunity to replace generic camera-model calls there with direct vectorized
math.

That should not come first, but it is a promising follow-on.

---

## What I Would Change In The Results Report

### 1. Split the measurements by run

This is the most important doc fix.

I would rewrite the report so each measured block is explicitly tied to one
named run, for example:

- **Run A:** 16 frames, masking substage timing
- **Run B:** 16 frames, reframer substage timing
- **Run C:** 11 frames, full pipeline timing

Then the reader can tell exactly which totals belong together.

### 2. Add a short "before vs after" summary table

Right now the doc is rich in post-optimization timings, but the impact is more
implied than summarized.

I would add a small table like:

| Metric | Before | After | Delta |
|--------|--------|-------|-------|
| Pass 1 share of masking | ... | ... | ... |
| Reframe stage | ... | ... | ... |
| Total pipeline | ... | ... | ... |

Even if some baselines come from different earlier reports, that summary would
make the success much easier to scan.

### 3. Move "default preset change" into a separate section

I would label it something like:

- **Related product/config changes**

instead of listing it as one of the optimization items.

### 4. Tighten the backprojection recommendation wording

The current recommendation for Approach D is good, but I would make it slightly
more explicit:

- D is recommended **only after numerical equivalence is demonstrated**
- A is the fallback if D is correct-but-not-fast-enough or harder than expected

That makes the decision path clearer.

---

## Recommended Next Steps

If I were sequencing the next round, I would do this:

1. Clean up the report so the measurements are grouped by run and internally
   consistent.
2. Build a small backprojection validation harness.
3. Prototype **Approach D** first.
4. Compare D against current backprojection for both correctness and runtime.
5. If D is not sufficient or is too finicky, prototype **Approach A** next.
6. Re-run a full end-to-end timing pass after the chosen backprojection
   optimization.
7. Reassess whether extraction or COLMAP is now the next best pipeline-wide
   target.

---

## Bottom Line

This looks like a genuinely productive optimization round.

The clearest success is that the obvious waste has been removed and the
pipeline's remaining cost center is now a deeper, more specific geometric
operation. That is real progress.

My main caution is not about the implementation. It is about the write-up:

- separate the runs
- isolate the preset change from the optimization accounting
- then use the backprojection section as the launch point for the next round

If that cleanup is done, this becomes a strong "phase 1 optimization complete,
phase 2 target identified" report.
