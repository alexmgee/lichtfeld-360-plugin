# LichtFeld Plugin Title And Registry System Report

Date: 2026-04-04

## Purpose

Explain why some plugins in LichtFeld show a readable title like `360 Record`, while this plugin currently shows the technical slug `lichtfeld-360-plugin`, and document the system so we do not have to rediscover it later.

## Short Version

There are two display paths in practice:

1. Registry/catalog-backed plugin cards
2. Local fallback plugin cards

Registry/catalog-backed entries already support:

- technical id/slug
- readable display title

Local fallback entries currently use the discovered plugin manifest directly, which means the visible card title comes from `project.name`.

That is why:

- published plugins can show spaced titles
- this local plugin currently shows the slug

## What We Verified

### 1. Existing installed plugins use slug-style package names

The working installed plugins on disk do **not** use spaced Python package names.

Examples:

- `C:\Users\alexm\.lichtfeld\plugins\360_record\pyproject.toml`
  - `name = "360_record"`
- `C:\Users\alexm\.lichtfeld\plugins\colmap_plugin\pyproject.toml`
  - `name = "colmap_plugin"`
- `C:\Users\alexm\.lichtfeld\plugins\densification\pyproject.toml`
  - `name = "densification"`

So their visible titles are not coming from `project.name`.

### 2. Installed marketplace plugins carry source metadata

Those plugins also include `.lichtfeld-source.json` files with remote-source info and a `registry_id`.

Examples:

- `community:record-360`
- `community:colmap`
- `community:densification`

This is a strong signal that LichtFeld is matching those installed plugins back to registry/catalog entries.

### 3. Registry/catalog entries have a separate display title

In LichtFeld source:

- `RegistryPluginInfo` includes both `name` and `display_name`
- the marketplace path uses `display_name or name`

That means published plugins already support the exact split we want:

- readable title for people
- technical slug for the system

### 4. Local manifest parsing still uses `project.name`

In the local discovery path, LichtFeld parses `pyproject.toml` and sets:

- `PluginInfo.name = project["name"]`

So for local fallback entries, the visible title defaults to the package/plugin slug.

### 5. The marketplace panel merges installed plugins with catalog entries by remote URL

If an installed plugin's remote URL matches a known catalog URL, the panel uses the catalog entry and does **not** create a plain `Local Installation` fallback card.

If it does **not** match the catalog, the panel creates a local-only entry and shows:

- `name = plugin.name`
- description from the local manifest

That is the behavior currently affecting this plugin.

## Why This Plugin Shows `lichtfeld-360-plugin`

Current local state:

- `project.name = "lichtfeld-360-plugin"`
- `.lichtfeld-source.json` exists
- `tool.lichtfeld.display_name = "360 Plugin"` is present in the manifest

But the plugin is still falling back to a local-only card instead of being matched to a registry/catalog entry.

As a result, the panel uses:

- `plugin.name`

which is the technical slug:

- `lichtfeld-360-plugin`

## Why `360 Plugin` In `project.name` Broke Loading

`uv` treats `project.name` as a Python package/distribution name.

Spaces are not valid there, so:

- `360 Plugin` works as a human title
- `360 Plugin` fails as a Python package name

That is why putting `360 Plugin` directly into `[project].name` caused `uv sync` parse failures.

## Correct Long-Term Model

For this plugin:

- visible title: `360 Plugin`
- package/internal slug: `lichtfeld-360-plugin`

That is not a weird request. It is already how the published plugins effectively behave.

## Practical Rule

If the goal is:

- "What should users see in LichtFeld?"

then the answer is:

- registry/catalog `display_name`

If the goal is:

- "What should `uv` and Python use internally?"

then the answer is:

- `project.name`

## What Needs To Exist For Users To See `360 Plugin`

This plugin needs a registry/catalog entry with:

- `name = "lichtfeld-360-plugin"`
- `display_name = "360 Plugin"`
- repository metadata pointing at the public GitHub repo

When users install through that registered path, LichtFeld can render the human title from the catalog entry.

## Important Caveat

Local manual development installs can still show the slug if they are rendered as local-only cards rather than matched marketplace entries.

So there are two separate experiences:

1. Local dev card before registry/catalog recognition
   - likely shows slug
2. Published/registered install path used by end users
   - can show `360 Plugin`

## Decision

For this plugin, keep:

- `project.name = "lichtfeld-360-plugin"`

Publish with:

- registry `display_name = "360 Plugin"`

Do not try to force the visible human title through the Python package name again.

## Related Draft

See:

- `docs/2026-04-04-360-plugin-registry-submission-draft.md`

That document contains the exact draft metadata to submit/publish.
