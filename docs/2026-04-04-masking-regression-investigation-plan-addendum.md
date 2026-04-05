# Default Masking Regression — Investigation Addendum

**Date:** 2026-04-04  
**Status:** New concrete finding  
**Related:** [2026-04-04-masking-regression-investigation-plan.md](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/docs/2026-04-04-masking-regression-investigation-plan.md)

---

## New Finding

The current plugin venv does **not** appear to contain a normal installed `sam2` package.

What was observed:

- `sam2` resolves as a **namespace package**, not a regular package
- `sam2.__file__` is `None`
- `importlib.util.find_spec("sam2.build_sam")` returns `None`
- there is no `sam2*.dist-info` entry in `site-packages`
- `site-packages/sam2/` currently contains only stray `_C.pyd` artifacts, not the normal Python package files

This is materially different from a healthy `sam2` install.

---

## Why This Matters

The masking code only enables the real SAM2 video backend if this import succeeds:

```python
from sam2.build_sam import build_sam2_video_predictor_hf
```

That gate lives in [core/backends.py](/c:/Users/alexm/.lichtfeld/plugins/lichtfeld-360-plugin/core/backends.py#L254).

If that import fails:

- `HAS_SAM2 = False`
- `Sam2VideoBackend` is not selected
- the system falls back to `FallbackVideoBackend`

That means:

- SAM2-specific fixes may appear to have no effect
- prompt-frame-selection fixes may appear not to matter
- regression analysis can become misleading because the runtime is no longer using the path we think it is using

---

## Revised Interpretation

The original investigation plan is still useful, but it is missing one high-priority possibility:

> the runtime environment may no longer match the previously good session, because `sam2` is not actually installed in a normal usable state

This would explain why:

- reverting code did not restore the earlier behavior
- SAM2-specific logic changes did not obviously improve results
- the system still behaves differently even after rolling back optimization changes

---

## Updated Priority

Before spending more time on deeper geometric diagnosis, the environment should be restored to a known-good `sam2` state.

That means verifying:

1. `sam2` is a real installed package, not a namespace stub
2. `sam2.build_sam` imports successfully
3. `Sam2VideoBackend` can actually be selected at runtime
4. `_C.pyd` state is tested only after the base Python package state is healthy

---

## Practical Implication

The next most important question is no longer:

- "did reverting the optimization code help?"

It is:

- "is the runtime even using SAM2 anymore?"

If the answer is no, then environment repair needs to happen before the masking regression investigation can produce trustworthy conclusions.
