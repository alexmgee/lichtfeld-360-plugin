# SAM 3 Video API Research Note

> Date: 2026-04-03
> Scope: Head start research for Track B5 in `docs/specs/2026-04-03-masking-v1-plan-final.md`

## Executive Summary

As of April 3, 2026, there are two credible official ways to bring SAM 3 video tracking into the plugin:

1. the native `sam3` package from Meta's repo / Hugging Face model card
2. the official Hugging Face Transformers integration (`Sam3Video*` and `Sam3TrackerVideo*`)

For this plugin, the best first direction is still the native `sam3` package, because:

- the existing premium image backend already uses native `sam3` image APIs in `core/backends.py`
- the plugin's premium tier is already conceptually "SAM 3 native", not "Transformers SAM 3"
- the official native docs explicitly show a video predictor entry point

The main correction to the current B5 assumption is this:

- native SAM 3 video does not appear to follow the same shape as `Sam2VideoBackend`
- instead of `init_state()` + `add_new_points_or_box()` + `propagate_in_video()`, the documented native API is session/request based around `handle_request(...)`

That does not block integration. It just means `Sam3VideoBackend` should be designed as an adapter that hides a session-style API behind the plugin's current `VideoTrackingBackend.track_sequence(...)` contract.

## Current Plugin Fit

The current plugin already has the right seams for a SAM 3 video backend:

- `core/backends.py` already imports native SAM 3 image APIs:
  - `build_sam3_image_model`
  - `Sam3Processor`
- `core/backends.py` already defines the sequence-level protocol:

```python
class VideoTrackingBackend(Protocol):
    def initialize(self) -> None: ...
    def track_sequence(
        self,
        frames: list[np.ndarray],
        initial_mask: np.ndarray | None = None,
        initial_frame_idx: int = 0,
    ) -> list[np.ndarray]: ...
    def cleanup(self) -> None: ...
```

- `core/masker.py` already routes Pass 2 through `self._video_backend.track_sequence(...)`
- `core/backends.py` already has the intended B5 seam in `get_video_backend(...)`

Important local reality check from this workspace on 2026-04-03:

- the plugin venv currently does not have top-level `sam3` installed
- the plugin venv currently does not have `transformers` installed
- the plugin venv currently does not have `accelerate` installed

So this note is about official API shape and integration planning, not a completed local runtime proof.

## Official API Findings

### 1. Native `sam3` video predictor exists

The official PyPI page and Hugging Face model card both show native video usage like this:

```python
from sam3.model_builder import build_sam3_video_predictor

video_predictor = build_sam3_video_predictor()
video_path = "<YOUR_VIDEO_PATH>"  # a JPEG folder or an MP4 video file

response = video_predictor.handle_request(
    request=dict(
        type="start_session",
        resource_path=video_path,
    )
)

response = video_predictor.handle_request(
    request=dict(
        type="add_prompt",
        session_id=response["session_id"],
        frame_index=0,
        text="<YOUR_TEXT_PROMPT>",
    )
)

output = response["outputs"]
```

Implications:

- the native predictor accepts a JPEG folder or an MP4
- that fits the plugin's existing synthetic-view tempdir approach well
- the prompt model shown in official docs is text-first, not point-first
- the response is session based, not a single fire-and-forget `track_sequence(...)` call

### 2. Native SAM 3 video is not a SAM 2 style drop-in

The current plan text for B5 still assumes "verify it follows SAM v2's pattern". Based on official docs, that is probably false for the native API.

SAM2 in this plugin currently looks like:

- build predictor
- `init_state(temp_frames_dir)`
- `add_new_points_or_box(...)`
- `propagate_in_video(...)`

Native SAM 3, by contrast, is documented around:

- `build_sam3_video_predictor()`
- `handle_request({"type": "start_session", ...})`
- `handle_request({"type": "add_prompt", ...})`
- session-owned outputs

So the integration problem is not "copy `Sam2VideoBackend` and rename imports". It is "wrap a different API style in the same plugin interface".

### 3. There are two distinct official Hugging Face SAM 3 video paths

The official Transformers docs expose both:

- `Sam3VideoModel` / `Sam3VideoProcessor`
- `Sam3TrackerVideoModel` / `Sam3TrackerVideoProcessor`

These are meaningfully different:

- `Sam3Video*` is concept segmentation over videos, driven primarily by text prompts
- `Sam3TrackerVideo*` is the SAM2-like tracker path for points / boxes / masks and is explicitly described as maintaining the same API family as SAM2 Video

That means there are really three viable B5 paths:

1. native `sam3` video predictor
2. Transformers `Sam3Video*` concept segmentation
3. Transformers `Sam3TrackerVideo*` visual tracker

For this plugin, the cleanest primary choice is still native `sam3`, with `Sam3TrackerVideo*` as the best fallback candidate if native runtime turns out to be awkward.

### 4. Official docs strongly support text-prompted video, more than point/box examples

The native docs prominently show text-prompted video:

- start session
- add prompt with `text="person"`

The Transformers docs show both text-driven and tracker-style workflows, but the native public examples are much clearer on text-first usage than on box-only or point-only native video prompting.

This matters because the plugin's premium image backend is already text-prompted. For B5 v1, that makes a text-first video backend the most coherent premium-tier continuation.

## Recommended Interpretation For This Plugin

### Recommendation: B5 v1 should be text-first

For the synthetic fisheye sequence, B5 v1 should treat SAM 3 video as:

- one promptable concept
- one centered target
- one binary output mask per frame

Recommended prompt strategy:

- use the first premium target text from plugin config
- in practice, that will almost always be `"person"`
- do not try to multiplex multiple text prompts in B5 v1

Why this is the best fit:

- the synthetic camera is already aimed at the subject, so a centered `"person"` prompt is a good match
- it preserves the premium-tier identity: SAM 3 stays the text-prompted path in both Pass 1 and Pass 2
- it avoids betting B5 v1 on under-documented native point/box behaviors

### Recommendation: keep JPEG-folder input, do not switch to MP4

The native docs explicitly allow a JPEG folder or MP4.

The plugin should keep the existing synthetic-frame tempdir pattern because:

- it already exists for SAM2
- it avoids adding encode/decode work
- it keeps debugging simpler
- it allows one implementation shape for both `Sam2VideoBackend` and `Sam3VideoBackend`

### Recommendation: do not assume "all persons" semantics are desirable

One subtle risk in premium SAM 3 video is that text-prompted concept segmentation can return all matching instances, not necessarily just the single centered actor.

That is potentially different from the plugin's synthetic-camera intention, which is to refine the primary subject found in Pass 1.

So B5 v1 should decide this explicitly:

- if SAM 3 returns multiple tracked person instances, the backend should not blindly union everything
- the safest policy is to select the instance nearest image center on the prompt frame and then keep that same tracked object identity across the session

This is one of the biggest under-explained areas going into implementation.

## Risks And Constraints To Expect

### 1. Native SAM 3 video may be more environment-sensitive than SAM2

The official PyPI "Getting Started" guidance describes:

- Python 3.12+
- PyTorch 2.7+
- CUDA 12.6+

That lines up reasonably well with this plugin's Python 3.12 requirement and current CUDA-oriented stack, but it also suggests that native SAM 3 video is not designed around a lightweight CPU fallback.

### 2. Windows / backend support is a real concern

Official repo issue:

- `facebookresearch/sam3` issue #150: Windows / no-NCCL concerns for the video predictor

This should be treated as a serious runtime-research item before B5 implementation is considered done on Windows.

### 3. Video memory pressure is likely to be real

Official repo issue:

- `facebookresearch/sam3` issue #169: CUDA OOM in the official video predictor example

That fits the plugin's current VRAM-management concerns exactly. `process_frames()` should continue to own teardown and recovery.

### 4. Box-prompt support may not be the most stable first path

Official repo issue:

- `facebookresearch/sam3` issue #193: box prompt behavior confusion for video

This is another reason to keep B5 v1 text-first unless local testing proves box/point prompting to be reliable and useful.

### 5. Local-vs-demo quality gaps have already been reported

Official repo issue:

- `facebookresearch/sam3` issue #275: quality mismatch between local predictor use and the web playground

So local quality validation should be part of B5 acceptance, not just "it runs".

## Best Integration Direction

### Primary recommendation

Use native `sam3` first:

- package: `sam3`
- backend style: adapter around `build_sam3_video_predictor()`
- prompt style: text prompt, probably `"person"`
- transport: numbered synthetic JPEG folder

### Contingency recommendation

If native `sam3` video proves awkward in the plugin runtime, the best fallback path is not "abandon SAM 3 video" but:

- evaluate `Sam3TrackerVideoModel` / `Sam3TrackerVideoProcessor`

Why that contingency is attractive:

- it is explicitly framed as the SAM2-style tracker replacement
- its API is closer to the plugin's current `Sam2VideoBackend`
- it may require less custom session adaptation

Why it should still be Plan B instead of Plan A:

- it adds `transformers` and probably `accelerate`
- it diverges from the current premium image backend's native `sam3` dependency family
- it creates a mixed premium stack: native for image, Transformers for video

## Recommended Research Steps Before Writing B5 Code

1. Install `sam3` in the plugin venv and confirm import of `build_sam3_video_predictor`.
2. Run a minimal session on a tiny numbered JPEG directory.
3. Inspect the exact `response` schema from:
   - `start_session`
   - `add_prompt`
4. Determine how framewise outputs are retrieved:
   - immediate `response["outputs"]`
   - iterator
   - later `handle_request(...)`
   - session-owned state
5. Determine how tracked object identities are represented.
6. Decide the single-subject reduction policy:
   - centered-object only
   - union all persons
7. Check whether an explicit session-close request exists and should be used in `cleanup()`.
8. Only if native runtime proves problematic, evaluate `Sam3TrackerVideo*` as a fallback architecture.

## Research Conclusions For Claude

If Claude is continuing Track B from the current codebase, the safest interpretation is:

- `Sam3VideoBackend` should be a new adapter, not a small SAM2 clone
- B5 v1 should be text-first
- the backend must accept prompt text from plugin configuration
- the backend should target one centered subject, not all detected people by default
- native `sam3` is the right first package to try
- Transformers tracker video is the right fallback option if native runtime is difficult

## Sources

Official sources used in this note:

- SAM 3 PyPI page: https://pypi.org/project/sam3/
- SAM 3 Hugging Face model card: https://huggingface.co/facebook/sam3
- Transformers SAM3 Video docs: https://huggingface.co/docs/transformers/main/model_doc/sam3_video
- Transformers SAM3 Tracker Video docs: https://huggingface.co/docs/transformers/model_doc/sam3_tracker_video
- Official repo issue #150 (Windows / NCCL): https://github.com/facebookresearch/sam3/issues/150
- Official repo issue #169 (video OOM): https://github.com/facebookresearch/sam3/issues/169
- Official repo issue #193 (box prompt uncertainty): https://github.com/facebookresearch/sam3/issues/193
- Official repo issue #275 (local quality mismatch): https://github.com/facebookresearch/sam3/issues/275

Local plugin files inspected while writing this note:

- `core/backends.py`
- `core/masker.py`
- `core/setup_checks.py`
- `docs/specs/2026-04-03-masking-v1-plan-final.md`
