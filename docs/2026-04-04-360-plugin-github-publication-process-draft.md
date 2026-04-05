# 360 Plugin GitHub Publication Process Draft

Date: 2026-04-04

## Goal

Prepare and publish the plugin repository so it is safe to ship and ready for LichtFeld registry/catalog submission.

This process is about:

- the GitHub repo
- the plugin manifest and release metadata
- making sure the backend/package identity stays stable

This process is **not** the step that makes the marketplace card show `360 Plugin`. That happens in the registry/catalog process.

## Desired Outcome

After this process:

- the repo is pushed safely to GitHub
- the plugin manifest remains valid for `uv` and Python packaging
- the public repo URL is ready to reference from registry metadata
- the technical slug stays `lichtfeld-360-plugin`

## Required Naming

Use this naming split consistently:

- visible human title: `360 Plugin`
- technical/plugin/package slug: `lichtfeld-360-plugin`
- repository: `alexmgee/lichtfeld-360-plugin`

Do not put spaces into:

- `[project].name`
- the repo slug
- any package/distribution identifier

## Current Repo State

The current manifest is already aligned correctly:

- `project.name = "lichtfeld-360-plugin"`
- `tool.lichtfeld.display_name = "360 Plugin"`

That is the correct repo-side state to keep.

## Draft Process

### 1. Stabilize the working tree

Before publishing:

- verify the plugin loads locally again
- verify the current venv is healthy
- avoid running plain `uv sync` unless the dependency model is finalized

Suggested checks:

```powershell
git status --short
& '.\.venv\Scripts\python.exe' -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
& '.\.venv\Scripts\python.exe' -c "from sam2.build_sam import build_sam2_video_predictor_hf; print('sam2 ok')"
```

### 2. Confirm manifest metadata

Verify these fields in `pyproject.toml`:

- `name = "lichtfeld-360-plugin"`
- `version = "0.1.0"` or the release version you intend to publish
- `description` is the public-facing short summary
- `Repository` points to the final GitHub repo URL
- `tool.lichtfeld.display_name = "360 Plugin"`

### 3. Confirm repo-facing content

Before pushing:

- update `README.md` so the public landing page matches the actual plugin state
- make sure install and usage instructions are current
- decide whether any temporary/dev docs should stay in the repo

### 4. Create a clean release checkpoint

Recommended sequence:

1. review `git status`
2. commit the current working state
3. push the branch
4. create a release tag once the plugin state is stable

Suggested tag shape:

- `v0.1.0`

### 5. Verify the public repo URL

After pushing, verify the canonical URL you want the registry to use:

```text
https://github.com/alexmgee/lichtfeld-360-plugin
```

This is the URL that should appear in registry metadata.

## Publish Checklist

- [ ] `pyproject.toml` uses `lichtfeld-360-plugin`
- [ ] `tool.lichtfeld.display_name` is `360 Plugin`
- [ ] `README.md` is publishable
- [ ] repo URL is correct
- [ ] version number is intentional
- [ ] branch is pushed
- [ ] release tag exists or planned

## Acceptance Criteria

This process is complete when:

- the plugin repo is safely pushed to GitHub
- the manifest remains valid for `uv`
- the repo URL is stable enough to submit to the LichtFeld registry

## What This Does Not Solve

This process alone does **not** change the local fallback card title in LichtFeld.

It only prepares the plugin to be published correctly.

For the user-facing `360 Plugin` title, see:

- `docs/2026-04-04-360-plugin-registry-submission-process-draft.md`
- `docs/2026-04-04-360-plugin-registry-submission-draft.md`

