# SAM 3 Video Backend Adapter Shape

> Date: 2026-04-03
> Purpose: Draft the backend shape Claude should expect when implementing B5

## Goal

Define the most practical `Sam3VideoBackend` shape for the plugin as it exists today.

This note assumes:

- `Track B2-B4` are already in place
- `Sam2VideoBackend` is the current real video backend
- `process_frames()` in `core/masker.py` already owns teardown, fallback, and retry
- premium Pass 1 already uses native `sam3` image APIs

## Core Recommendation

Implement `Sam3VideoBackend` as an adapter around the native `sam3` video predictor, but do not force the rest of the plugin to learn that session API.

In other words:

- outside the class, keep the current plugin contract
- inside the class, translate that contract into `sam3` session requests

## Important Design Correction

`Sam3VideoBackend` is not just "Sam2VideoBackend with different imports".

The current `Sam2VideoBackend` contract is essentially:

- initialize predictor once
- `track_sequence(frames, initial_frame_idx=...)`
- get back `list[np.ndarray]`

The documented native SAM 3 video flow is session based:

- build predictor
- start session for a resource path
- add prompt to session
- retrieve outputs from the session flow

So the backend should be designed as a translation layer.

## Recommended Public Shape

```python
class Sam3VideoBackend:
    """SAM 3 video tracking on synthetic fisheye views.

    Native adapter around build_sam3_video_predictor().
    Uses text prompting on the synthetic view sequence.
    """

    def __init__(
        self,
        targets: list[str],
        device: str = "cuda",
    ) -> None:
        self._targets = targets
        self._device = device
        self._predictor: Any = None

    def initialize(self) -> None:
        ...

    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
    ) -> list[np.ndarray]:
        ...

    def cleanup(self) -> None:
        ...
```

## Why `targets` Must Be Added

This is the biggest interface detail B5 needs to account for.

Today:

- `VideoTrackingBackend.track_sequence(...)` does not accept `targets`
- `Sam2VideoBackend` does not need them because it prompts with a center click
- `FallbackVideoBackend` gets `targets` in its constructor

But a text-prompted `Sam3VideoBackend` does need target text.

So the factory should instantiate it like this:

```python
def get_video_backend(
    preference: str | None = None,
    fallback_image_backend: MaskingBackend | None = None,
    targets: list[str] | None = None,
) -> VideoTrackingBackend | None:
    if preference == "sam3" and HAS_SAM3 and HAS_TORCH:
        return Sam3VideoBackend(targets=targets or ["person"])
    if HAS_SAM2 and HAS_TORCH and preference != "fallback":
        return Sam2VideoBackend()
    if fallback_image_backend is not None:
        return FallbackVideoBackend(
            fallback_image_backend, targets or ["person"]
        )
    return None
```

This keeps the existing `VideoTrackingBackend` protocol intact while still letting the backend carry prompt text.

## Prompting Policy For V1

Recommended B5 v1 policy:

- use only the first target string from `targets`
- default to `"person"` if empty
- do not try to concatenate multiple prompts
- do not try to run multiple target sessions in one call

Reason:

- the synthetic camera is already subject-centered
- the premium masking workflow is already person-centric in practice
- multi-prompt behavior is harder to interpret and harder to reduce back to one binary mask sequence

Suggested helper:

```python
def _select_prompt_text(self) -> str:
    return self._targets[0].strip() if self._targets else "person"
```

## Recommended Internal Flow

### `initialize()`

Responsibilities:

- import `build_sam3_video_predictor`
- construct the predictor once
- store it on `self`

Suggested sketch:

```python
def initialize(self) -> None:
    from sam3.model_builder import build_sam3_video_predictor

    if not HAS_SAM3:
        raise ImportError("sam3 not installed")

    logger.info("Loading SAM 3 video predictor...")
    self._predictor = build_sam3_video_predictor(device=self._device)
    logger.info("SAM 3 video predictor ready")
```

Note:

- the exact constructor signature still needs runtime confirmation
- if native SAM 3 does not actually accept a `device=` keyword, adapt here and nowhere else

### `track_sequence(...)`

Responsibilities:

1. validate initialized state
2. return `[]` for empty input
3. write numbered JPEGs to a tempdir
4. start a SAM 3 session on that tempdir
5. add a text prompt on `initial_frame_idx`
6. collect framewise masks
7. reduce outputs to a single binary mask per frame
8. clean temp resources in `finally`

Suggested high-level sketch:

```python
def track_sequence(
    self,
    frames: list[np.ndarray],
    initial_mask: np.ndarray | None = None,
    initial_frame_idx: int = 0,
) -> list[np.ndarray]:
    if self._predictor is None:
        raise RuntimeError("Sam3VideoBackend not initialized")
    if not frames:
        return []

    orig_h, orig_w = frames[0].shape[:2]
    prompt_text = self._select_prompt_text()

    with tempfile.TemporaryDirectory() as tmpdir:
        self._write_numbered_jpegs(frames, tmpdir)

        session = self._start_session(tmpdir)
        outputs = self._add_text_prompt(
            session_id=session["session_id"],
            frame_index=initial_frame_idx,
            text=prompt_text,
        )

        masks_by_frame = self._collect_masks_from_outputs(
            outputs=outputs,
            n_frames=len(frames),
            frame_size=(orig_h, orig_w),
        )

    return masks_by_frame
```

## The Real Adapter Work

The most important implementation work is not tempdir writing. It is output normalization.

The backend must turn whatever SAM 3 returns into:

- exactly one `np.ndarray` mask per input frame
- `uint8`
- shape matching the original synthetic frame resolution
- stable frame ordering

That strongly suggests internal helpers like these:

```python
def _start_session(self, frames_dir: str) -> dict[str, Any]: ...
def _add_text_prompt(
    self,
    session_id: str,
    frame_index: int,
    text: str,
) -> dict[str, Any]: ...
def _collect_masks_from_outputs(
    self,
    outputs: Any,
    n_frames: int,
    frame_size: tuple[int, int],
) -> list[np.ndarray]: ...
def _normalize_mask(
    self,
    mask: Any,
    frame_size: tuple[int, int],
) -> np.ndarray: ...
```

## Single-Subject Reduction Policy

This is the most important semantic decision in the adapter.

Text-prompted SAM 3 video may return multiple matching instances for `"person"`.

The plugin, however, wants a single synthetic-pass mask sequence that refines the main subject.

Recommended policy for B5 v1:

1. On the prompt frame, choose the tracked instance nearest the image center.
2. If the response already carries object IDs, keep only that tracked object on later frames.
3. If the response does not expose stable IDs cleanly, fall back to the mask whose centroid stays nearest image center on each frame.
4. Do not union every returned person instance by default.

Why this is the safest choice:

- it matches the synthetic camera's centered-subject design
- it keeps premium behavior aligned with the current two-pass plan
- it avoids expanding masks to background people at the frame edges

Suggested helpers:

```python
def _pick_centered_object_id(
    self,
    prompt_frame_outputs: Any,
    frame_size: tuple[int, int],
) -> Any: ...

def _extract_frame_mask_for_object(
    self,
    frame_outputs: Any,
    object_id: Any,
    frame_size: tuple[int, int],
) -> np.ndarray: ...
```

## `initial_mask` Handling

Keep the protocol behavior aligned with the rest of Track B:

- accept `initial_mask`
- ignore it in B5 v1

Rationale:

- it preserves the protocol
- it keeps B5 v1 focused on the documented text-first path
- it leaves room for a future SAM 3 refinement mode that mixes text with visual prompts

The docstring should say that explicitly.

## `cleanup()`

Responsibilities:

- drop predictor references
- close any session state if the final API exposes session-close behavior
- free CUDA cache if torch is available

Suggested sketch:

```python
def cleanup(self) -> None:
    self._predictor = None
    if HAS_TORCH and torch.cuda.is_available():
        torch.cuda.empty_cache()
    logger.info("SAM 3 video backend cleaned up")
```

## Factory Priority

Recommended `get_video_backend(...)` priority after B5:

```python
if preference == "sam3" and HAS_SAM3 and HAS_TORCH:
    return Sam3VideoBackend(targets=targets or ["person"])
if HAS_SAM2 and HAS_TORCH and preference != "fallback":
    return Sam2VideoBackend()
if fallback_image_backend is not None:
    return FallbackVideoBackend(
        fallback_image_backend, targets or ["person"]
    )
return None
```

This keeps premium behavior explicit:

- if the user asked for SAM 3 and it exists, use SAM 3 video
- otherwise default-video stays SAM2
- otherwise fall back to per-frame image detection

## Tests Claude Should Add

### Interface tests

- `Sam3VideoBackend` has `initialize`, `track_sequence`, `cleanup`
- `get_video_backend(preference="sam3", targets=["person"])` returns `Sam3VideoBackend` when available

### Mocked adapter tests

- mocked native predictor receives:
  - `start_session`
  - `add_prompt`
- `track_sequence(...)` returns one mask per frame
- masks are resized back to original synthetic frame resolution
- empty frame list returns `[]`
- `initial_mask` is accepted and ignored

### Object-selection tests

- when outputs contain multiple persons, adapter chooses centered instance
- frame ordering is stable
- missing frame outputs become zero masks instead of crashing

### Failure tests

- predictor init failure bubbles cleanly to `process_frames()` fallback logic
- session/prompt failure bubbles cleanly to `process_frames()` fallback logic
- tempdir is cleaned on exception

## Open Questions To Resolve During Runtime Research

1. What exact request types exist beyond `start_session` and `add_prompt`?
2. Does `add_prompt` itself return all needed framewise outputs, or is there another propagation request?
3. What exact shape do `response["outputs"]` items have?
4. How are tracked object IDs represented?
5. Does the predictor expose an explicit session-close request?
6. Does native SAM 3 video work in the plugin's Windows/LichtFeld runtime without NCCL-related issues?
7. Does it require resized frames, or can it run directly on the synthetic 2048 sequence?

## Contingency Shape If Native SAM 3 Is Unusable

If native SAM 3 video turns out to be too difficult in practice, the backup adapter should target:

- `transformers.Sam3TrackerVideoModel`
- `transformers.Sam3TrackerVideoProcessor`

That backup path would look much more like `Sam2VideoBackend`, but it should be treated as a deliberate fallback architecture, not silently swapped in under the same assumptions.

## Bottom Line

The backend Claude should expect is:

- premium-only
- text-prompted
- session-adapted
- single-subject by policy
- factory-injected with `targets`
- still conforming to the plugin's existing `VideoTrackingBackend` protocol

That is the shape most likely to fit the current plugin cleanly without forcing Track B to re-architect the rest of the masking pipeline.
