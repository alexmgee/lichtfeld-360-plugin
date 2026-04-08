# Masking Diagnostics Guide

When masking is enabled, the 360 Plugin can write a detailed diagnostics file that shows exactly what the masking system saw, tracked, and decided on every frame. This is the fastest way to figure out why masking worked on some frames and not others.

## Enabling Diagnostics

1. Open the **360 Plugin** panel in LichtFeld Studio
2. Expand the **Operator Masking** section
3. Check **Diagnostics**
4. Run the pipeline as normal

When the run finishes, a `masking_diagnostics.json` file appears in your output folder alongside the masks.

## Sending Feedback

If masking isn't working well on your footage, here's how to help us fix it:

1. Enable **Diagnostics** and re-run the pipeline on the problem video
2. Find `masking_diagnostics.json` in your output folder
3. Send that file along with a description of what went wrong:
   - Which frames looked bad (e.g. "frames 3, 7, and 12 had no mask" or "the mask was on a statue instead of me")
   - What the scene looked like (indoors/outdoors, other people visible, statues or mannequins nearby)

The diagnostics file contains detection confidence scores and tracking outcomes — no images or personal data. It's small (a few KB) and safe to share.

## What the Diagnostics File Contains

### Default Preset (16-view layout)

The Default preset uses a two-pass pipeline:

- **Pass 1** runs YOLO person detection across 16 virtual cameras arranged around the 360 frame. Its job is to find the operator and estimate their direction on the sphere.
- **Pass 2** renders a synthetic fisheye camera aimed at the detected person and runs SAM2 video tracking across all frames. If tracking drops a frame, a per-frame rescue detection attempts recovery.

The diagnostics file records what happened at each stage for every frame:

```
{
  "version": 1,
  "backend": "YoloSamBackend",
  "video_backend": "Sam2VideoBackend",
  "total_frames": 12,
  "frames_with_direction": 10,
  "frames_tracked": 8,
  "frames_rescued": 2,
  "frames_empty": 2,
  "frames": [ ... per-frame details ... ]
}
```

Each frame entry contains:

**Pass 1 — Person detection**
- `views_detected` / `views_total` — how many of the 16 detection cameras saw a person (e.g. 6/16)
- `max_confidence` / `mean_confidence` — YOLO confidence scores across detecting views
- `by_view` — per-camera breakdown with confidence and bounding box coverage percentage
- `direction_yaw` / `direction_pitch` — estimated person direction on the sphere (null if no detection)

**Pass 2 — Tracking**
- `tracked` — whether SAM2 video tracking produced a mask for this frame
- `rescued` — whether per-frame re-detection recovered a dropped tracking frame
- `mask_pixels` — size of the mask produced
- `box_confidence` — YOLO confidence of the rescue detection (only on rescued frames)
- `box_center_dist` — how far the rescue detection was from frame center in pixels (helps identify wrong-subject detections)

**Final**
- `mask_pixels` — size of the final mask written to disk
- `source` — which stage produced the final mask:

| Source | What happened |
|--------|--------------|
| `pass2_tracked` | SAM2 tracking held — best case |
| `pass2_rescued` | Tracking dropped the frame, but re-detection recovered it |
| `pass1_fallback` | Pass 2 failed entirely, fell back to Pass 1 |
| `no_direction` | Pass 1 found no person in any of the 16 views |
| `empty` | Person was found but neither tracking nor rescue produced a mask |

### Cubemap Preset (6-view layout)

Cubemap runs detection directly on each reframed face — no two-pass pipeline. The diagnostics show per-view results:

```
{
  "version": 1,
  "mode": "cubemap",
  "backend": "YoloSamBackend",
  "views_per_frame": 6,
  "total_frames": 12,
  "frames": [
    {
      "stem": "video_00003",
      "views_detected": 2,
      "views_total": 6,
      "by_view": {
        "00_00": { "detected": true,  "mask_pixels": 34521, "confidence": 0.82 },
        "01_00": { "detected": false, "mask_pixels": 0,     "confidence": null }
      }
    }
  ]
}
```

## Common Patterns

### "Masking works on some frames but not others"

Look at `pass1.views_detected` across frames. If the count swings between 0 and 6+, YOLO detection is inconsistent — the operator may be at the edge of multiple detection views or partially occluded in some frames.

Also check `final.source`: if most frames are `pass2_tracked` but a few are `pass2_rescued` or `empty`, the issue is SAM2 tracking dropout rather than detection.

### "The wrong thing gets masked"

Look at `pass2.box_center_dist` on rescued frames. High values (hundreds of pixels) suggest the rescue detection latched onto something far from where the operator should be — like a statue, mannequin, or bystander.

Also compare `pass1.direction_yaw` across frames. If the direction jumps around erratically, YOLO may be alternating between detecting the real operator and a decoy.

### "No mask at all on any frame"

Check `frames_with_direction` in the top-level summary. If it's 0, Pass 1 never found a person. This usually means the operator is too small in the 360 frame (very distant from the camera), or the detection confidence threshold is filtering out weak detections.

### "Cubemap misses the operator on specific views"

Check `by_view` to see which face names consistently show `"detected": false`. If the same view always misses, the operator is likely at the edge of that view's field of view.

## Performance

Diagnostics adds no overhead when disabled (the default). When enabled, the cost is negligible — just dict writes during processing and a single JSON dump at the end. On rescued frames, an extra detection call captures the box confidence data.
