# 360 Plugin Registry Submission Draft

Date: 2026-04-04

## Goal

Publish the plugin so users see:

- Visible title: `360 Plugin`
- Internal/plugin/package slug: `lichtfeld-360-plugin`

This matches how existing marketplace plugins behave:

- human-readable title on the first line
- technical slug or source on the second line

## Key Finding

The installed plugins already on disk do **not** use spaced `project.name` values:

- `360_record`
- `colmap_plugin`
- `densification`

Their readable titles come from registry/catalog metadata, not from the Python package name in `pyproject.toml`.

For this plugin, the right long-term split is:

- `project.name = "lichtfeld-360-plugin"`
- registry `display_name = "360 Plugin"`

## Recommended Registry Identity

Use the same slug everywhere technical:

- package name: `lichtfeld-360-plugin`
- local plugin name: `lichtfeld-360-plugin`
- registry plugin id: `community:lichtfeld-360-plugin`

Use the human-facing title only in registry display metadata:

- `display_name = "360 Plugin"`

## Proposed Registry Index Entry

This is the metadata shape LichtFeld's registry search path expects for the index list.

```json
{
  "namespace": "community",
  "name": "lichtfeld-360-plugin",
  "display_name": "360 Plugin",
  "summary": "360 video to Gaussian Splat scene with automated masking and rig support.",
  "author": "Alex Gee",
  "latest_version": "0.1.0",
  "repository": "https://github.com/alexmgee/lichtfeld-360-plugin",
  "keywords": [
    "360",
    "gaussian-splat",
    "masking",
    "video",
    "reconstruction",
    "colmap",
    "rig"
  ]
}
```

## Proposed Registry Detail File

This is the per-plugin detail shape LichtFeld resolves when installing or checking compatibility.

Recommended file path in the registry:

```text
plugins/community/lichtfeld-360-plugin.json
```

Draft:

```json
{
  "namespace": "community",
  "name": "lichtfeld-360-plugin",
  "display_name": "360 Plugin",
  "summary": "360 video to Gaussian Splat scene with automated masking and rig support.",
  "author": "Alex Gee",
  "repository": "https://github.com/alexmgee/lichtfeld-360-plugin",
  "versions": {
    "0.1.0": {
      "version": "0.1.0",
      "plugin_api": ">=1,<2",
      "lichtfeld_version": ">=0.5.0",
      "required_features": [],
      "dependencies": [],
      "git_ref": "v0.1.0"
    }
  }
}
```

## Publish-Time Notes

Before submitting, confirm these values:

1. Replace `git_ref` with the real release tag you publish.
2. Keep `display_name` exactly `360 Plugin`.
3. Keep `name` exactly `lichtfeld-360-plugin`.
4. Keep `repository` pointed at the public GitHub repo users should install from.
5. Update `latest_version` to match the tag/release you publish.

If the registry prefers archive installs over Git installs, the version block can instead use `download_url` and optionally `checksum`.

Example archive-style version block:

```json
{
  "version": "0.1.0",
  "plugin_api": ">=1,<2",
  "lichtfeld_version": ">=0.5.0",
  "required_features": [],
  "dependencies": [],
  "download_url": "https://github.com/alexmgee/lichtfeld-360-plugin/archive/refs/tags/v0.1.0.tar.gz",
  "checksum": "sha256:REPLACE_WITH_REAL_SHA256"
}
```

## Expected User-Facing Result

Once the plugin is represented in the LichtFeld registry/catalog and installed through that path:

- users should see `360 Plugin` as the card title
- the technical slug can remain `lichtfeld-360-plugin`
- the plugin remains valid for `uv`/Python packaging

## Current Repo Values

The current local manifest is already aligned with this plan:

- `project.name = "lichtfeld-360-plugin"`
- `tool.lichtfeld.display_name = "360 Plugin"`

The missing piece is registry/catalog publication, not another local manifest rename.
