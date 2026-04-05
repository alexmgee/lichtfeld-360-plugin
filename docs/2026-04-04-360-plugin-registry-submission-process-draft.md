# 360 Plugin Registry Submission Process Draft

Date: 2026-04-04

## Goal

Get LichtFeld users to see this plugin as:

- `360 Plugin`

while keeping the technical/plugin/package slug as:

- `lichtfeld-360-plugin`

This is the process that controls the user-facing title for published installs.

## Desired Outcome

After this process:

- LichtFeld registry/catalog knows the plugin as `community:lichtfeld-360-plugin`
- the visible marketplace title is `360 Plugin`
- installs can stay technically valid with slug-style package metadata

## Why This Process Exists

The working installed plugins already prove that LichtFeld supports:

- a readable title on the first line
- a separate technical slug/source on the second line

That title comes from registry/catalog metadata, not from putting spaces into `project.name`.

## Required Metadata Split

Keep this split:

- registry/plugin name: `lichtfeld-360-plugin`
- registry `display_name`: `360 Plugin`

That is the exact behavior users want:

- technical slug for the system
- readable title for people

## Draft Process

### 1. Finish GitHub publication first

Before registry submission:

- the repo should be pushed
- the public GitHub URL should be final
- the release/tag strategy should be known

Reference:

- `docs/2026-04-04-360-plugin-github-publication-process-draft.md`

### 2. Prepare the registry index entry

Create or propose the index metadata that the LichtFeld registry search path uses.

Required fields:

- `namespace`
- `name`
- `display_name`
- `summary`
- `author`
- `latest_version`
- `repository`
- `keywords`

For this plugin, the intended values are drafted in:

- `docs/2026-04-04-360-plugin-registry-submission-draft.md`

### 3. Prepare the per-plugin detail file

Create or propose the plugin detail JSON that includes:

- `namespace`
- `name`
- `display_name`
- `summary`
- `author`
- `repository`
- `versions`

Each version block should include compatibility fields like:

- `plugin_api`
- `lichtfeld_version`
- `required_features`

and one install path:

- `git_ref`

or:

- `download_url`
- `checksum`

### 4. Submit the registry change

Depending on how the LichtFeld registry is managed, this will likely be one of:

1. a PR against the plugin registry repo
2. an issue requesting registry inclusion
3. a direct metadata submission to the registry maintainers

The payload should use:

- `name = "lichtfeld-360-plugin"`
- `display_name = "360 Plugin"`

### 5. Wait for catalog recognition

Once the registry/catalog entry exists and matches the plugin repository/source URL:

- LichtFeld can match the installed plugin to the catalog entry
- the card can use the registry display metadata
- users should see `360 Plugin`

## Submission Checklist

- [ ] GitHub repo is public and final
- [ ] repo URL in metadata is correct
- [ ] registry name is `lichtfeld-360-plugin`
- [ ] registry display title is `360 Plugin`
- [ ] latest version matches the release/tag
- [ ] compatibility fields are correct
- [ ] install method is valid (`git_ref` or archive URL)

## Suggested Registry Identity

- full id: `community:lichtfeld-360-plugin`
- display title: `360 Plugin`

## Acceptance Criteria

This process is complete when:

- the plugin exists in the LichtFeld registry/catalog
- the catalog entry uses `display_name = "360 Plugin"`
- an installed user sees `360 Plugin` instead of the slug on the plugin card

## Important Caveat

This process is about the published user path.

Your local manual dev install may still show the slug if it is being rendered as a plain local fallback card rather than as a matched catalog entry.

That does not mean the registry setup is wrong; it means local fallback and published catalog installs are different display paths.

## Related Docs

- `docs/2026-04-04-360-plugin-registry-submission-draft.md`
- `docs/2026-04-04-lichtfeld-plugin-title-registry-system-report.md`
- `docs/2026-04-04-lichtfeld-display-name-issue-draft.md`
