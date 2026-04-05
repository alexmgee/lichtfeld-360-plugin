# Issue Draft: Support `tool.lichtfeld.display_name` for Local Plugins

## Title

Support a separate human-facing display name for local plugins

## Summary

LichtFeld currently uses `project.name` from a plugin's `pyproject.toml` as both:

- the internal plugin/package identifier
- the visible plugin title in the Plugin Marketplace / Local Installation card

That makes it hard to use a safe internal slug such as `lichtfeld-360-plugin` while presenting a clean user-facing title such as `360 Plugin`.

LichtFeld's registry path already has this separation conceptually, because registry entries support a separate `display_name`. Local installed plugins should support the same pattern.

## Problem

For local installed plugins, LichtFeld currently parses plugin metadata from `pyproject.toml` and sets:

- `PluginInfo.name = project.name`

That same `name` is then used as the visible card title for local plugins in the Plugin Marketplace panel.

This means plugin authors must choose between:

- a safe internal identifier like `lichtfeld-360-plugin`, or
- a human-facing title like `360 Plugin`

Using `project.name` as the visible title works visually, but it also ties the package/plugin identifier, directory identity, and UI label together more tightly than necessary.

## Why This Matters

Plugin authors need two different concepts:

1. Internal/plugin/package id
   - stable
   - slug-safe
   - suitable for packaging, installation, updates, and internal references

2. Display name
   - human-friendly
   - can contain spaces and branding
   - should be what users see in the UI

Without this separation, local plugin titles are constrained by packaging concerns.

## Current Behavior

Observed local-plugin behavior:

- local plugin parsing reads `project.name` into `PluginInfo.name`
- the local marketplace card uses `plugin.name` as the visible title

Observed registry behavior:

- registry metadata already supports a separate `display_name`
- registry-backed marketplace entries already use `display_name` when available

So the platform already has the right idea for registry plugins; local plugins just do not have parity yet.

## Proposed Solution

Add optional support for:

```toml
[tool.lichtfeld]
display_name = "360 Plugin"
```

### Behavior

- `project.name` remains the internal plugin identifier
- `tool.lichtfeld.display_name` becomes the human-facing UI title
- if `display_name` is absent or empty, LichtFeld falls back to `project.name`

This keeps the change fully backward-compatible.

## Suggested Implementation

### 1. Extend local plugin metadata model

Add a new field to the local plugin info structure:

- `PluginInfo.display_name: str = ""`

### 2. Parse `tool.lichtfeld.display_name`

When parsing a local plugin manifest:

- read `tool.lichtfeld.display_name`
- normalize it with something like `.strip()`
- store `display_name or project.name`

### 3. Use display name in local marketplace cards

When building local marketplace entries / local installation cards:

- use `plugin.display_name` for the visible title
- keep `plugin.name` for internal/plugin identity

### 4. Preserve fallback behavior

If `display_name` is missing:

- show `project.name`, exactly as today

## Example

Plugin manifest:

```toml
[project]
name = "lichtfeld-360-plugin"
version = "0.1.0"
description = "360 video to Gaussian Splat scene with automated masking and rig support."

[tool.lichtfeld]
display_name = "360 Plugin"
hot_reload = true
plugin_api = ">=1,<2"
lichtfeld_version = ">=0.5.0"
required_features = []
```

Expected UI result:

- visible title: `360 Plugin`
- internal plugin id remains: `lichtfeld-360-plugin`

## Backward Compatibility

This should be low risk:

- existing plugins without `display_name` continue working unchanged
- no registry behavior needs to change
- no plugin packaging behavior needs to change

## Acceptance Criteria

1. A local plugin can specify `tool.lichtfeld.display_name`.
2. The local plugin card title uses `display_name` when present.
3. `project.name` remains the internal plugin identifier.
4. Plugins without `display_name` still show `project.name`.
5. Registry plugins continue to work exactly as they do today.

## Rationale

This brings local plugin behavior into line with the existing registry/marketplace model and avoids forcing plugin authors to choose between clean branding and safe internal naming.
