# Masking Setup UX — Product Contract and Flow Proposal

**Date:** 2026-04-08  
**Status:** Proposal  
**Scope:** Fresh-install behavior, FullCircle availability, SAM 3 onboarding UX, setup-state model  
**Related docs:**  
- `docs/specs/2026-04-08-sam3-cubemap-masking-design.md`  
- `docs/specs/2026-04-08-sam3-cubemap-implementation-plan.md`

---

## Summary

This proposal defines the intended user experience for masking setup from first plugin install through optional SAM 3 onboarding.

The desired contract is:

1. The plugin is usable immediately after install even if the user never touches masking.
2. FullCircle is the built-in local masking path and is ready immediately on a normal install.
3. SAM 3 Cubemap is an optional alternate path with guided onboarding.
4. The UI must always make it clear that FullCircle remains available even if SAM 3 setup is incomplete.

This is not the same as the current behavior. Today, a fresh install can run the plugin without masking, but FullCircle is not fully ready because the shipped dependency layout does not include the full FullCircle runtime by default.

---

## Product Contract

### What the plugin should feel like on first launch

Fresh install should feel like:

- `Plugin ready`
- `Masking optional`
- `FullCircle ready now`
- `SAM 3 Cubemap optional setup`

The user should not have to install anything or create any accounts just to use the plugin normally.

If the user never enables masking, nothing about SAM 3 or FullCircle should block their first run.

If the user does want masking, FullCircle should already be available.

SAM 3 should be presented as an optional upgrade path with better onboarding, not as the default way to get masking working.

### Important clarification

There are two different meanings of "works without masking installed":

- The plugin can run with masking disabled.
- The plugin ships without any masking runtime at all.

This proposal assumes the first meaning, not the second.

If FullCircle is supposed to be ready immediately, then the plugin must ship with the FullCircle runtime already provisioned. If the product instead ships with no masking runtime at all, FullCircle cannot honestly be described as ready out of the box.

---

## Current Behavior vs Intended Behavior

### Current behavior

Current dependency layout:

- base install includes `torch`, `ultralytics`, and `segment-anything`
- `sam2` is still optional under `video-tracking`
- `sam3` is optional under `sam3-masking`

Current UX effect:

- plugin core works after install
- masking can remain disabled
- FullCircle appears as the default method but is not actually ready
- the panel asks the user to install more runtime for FullCircle
- SAM 3 setup is available as an alternate branch, but its status model is too compressed

### Intended behavior

Desired UX effect:

- plugin core works after install
- masking is off by default
- FullCircle is ready immediately if the user wants it
- SAM 3 is clearly optional and guided
- the user always has a working local masking path even if SAM 3 setup is blocked

### Gap

The current packaging and setup gating do not yet support the intended contract.

If the intention is "FullCircle ready on fresh install," then FullCircle dependencies cannot remain partially optional.

---

## Recommended Fresh-Install State

### Fresh install state

After the plugin is installed and opened for the first time:

- extraction/reframe/alignment work with masking disabled
- masking section is visible
- masking master toggle is off
- FullCircle is shown as `Ready`
- SAM 3 Cubemap is shown as `Optional setup`

### FullCircle presentation

FullCircle should be framed as:

- local
- no account required
- recommended default masking path
- ready now

Suggested copy:

`FullCircle is ready on this install. No HuggingFace account required.`

### SAM 3 presentation

SAM 3 should be framed as:

- optional
- gated by HuggingFace access
- cubemap-only for now
- guided setup available

Suggested copy:

`SAM 3 Cubemap is optional. It requires HuggingFace approval and model download.`

---

## End-to-End User Flows

## Flow A: User never uses masking

1. User installs plugin.
2. User opens plugin.
3. User selects video and output path.
4. User leaves masking disabled.
5. User runs pipeline successfully.

Expected UX:

- no install prompt
- no HuggingFace messaging forced on the user
- no implication that the plugin is incomplete

## Flow B: User wants FullCircle immediately

1. User installs plugin.
2. User opens masking section.
3. FullCircle is already selected or clearly available.
4. User enables masking.
5. User runs pipeline.

Expected UX:

- no extra install step
- no account step
- no ambiguity about readiness

If FullCircle weights still need first-use download, that must be communicated as normal runtime preparation, not as setup failure.

## Flow C: User wants SAM 3 and has never used HuggingFace

1. User selects `SAM 3 Cubemap`.
2. UI shows setup checklist and step-by-step guidance.
3. User creates HuggingFace account.
4. User requests access to the SAM 3 model.
5. User generates a token.
6. User pastes token into the panel.
7. User clicks `Verify Access`.
8. UI confirms either:
   - access granted, or
   - access still pending
9. Once access is granted, user clicks `Install SAM 3`.
10. Plugin installs runtime and downloads weights.
11. UI shows `Ready`.
12. User enables masking and runs.

Expected UX:

- each step has a visible status
- the next required action is always obvious
- FullCircle remains visibly available throughout

## Flow D: User pasted a bad token

1. User enters token.
2. Verification fails.

Expected UX:

- status says `Token invalid`
- guidance says `Generate a new HuggingFace token and try again`
- does not say `pending`
- does not imply the install is broken

## Flow E: User access is pending

1. User enters a valid token.
2. HuggingFace responds that access is not yet granted.

Expected UX:

- status says `Access pending approval`
- guidance says `Wait for HuggingFace approval, then click Re-check Setup`
- install button stays disabled
- FullCircle remains visibly available

## Flow F: SAM 3 install fails locally

1. User has valid token and approved access.
2. Runtime installation fails or the package imports as broken.

Expected UX:

- status says `SAM 3 install failed`
- report explains whether the failure was package install, import validation, or weight download
- user gets `Retry install`
- FullCircle still shown as available

---

## Proposed Setup State Model

The current state model is too compressed for SAM 3 onboarding UX. Booleans are still useful, but the panel also needs a richer SAM 3 setup report.

### Keep the existing broad readiness booleans

- `fullcircle_ready`
- `sam3_ready`
- `default_tier_ready`
- `is_ready`

These remain useful for pipeline gating and coarse UI branching.

### Add a richer SAM 3 setup report

Introduce a dedicated report object returned by setup-check helpers and by explicit `Check Setup` / `Verify Access` actions.

Suggested shape:

```python
@dataclass
class Sam3SetupReport:
    token_status: str          # missing | invalid | verified | network_error
    access_status: str         # unknown | pending | granted | denied | network_error
    runtime_status: str        # missing | installing | installed | broken
    weights_status: str        # missing | downloading | present | failed
    overall_stage: str         # needs_token | needs_access | ready_to_install | installing | needs_weights | ready | error
    message: str
    next_action: str
```

This object is for UX. It does not replace the runtime booleans used by the pipeline.

### Why this matters

Today, different failure modes collapse into the same user-facing branch:

- bad token
- access not approved yet
- network error
- install/import failure

Those must become distinct states with distinct messages.

---

## Proposed Panel UX

## Overall layout

The masking section should always show both paths clearly:

- `FullCircle`
- `SAM 3 Cubemap`

This can remain a dropdown for implementation simplicity, but visually it should behave like two named modes rather than one masking feature that changes identity.

### FullCircle state block

When FullCircle is selected:

- show `Ready` badge if ready
- show `Enable masking`
- show `Diagnostics`
- show short backend info

If FullCircle is unexpectedly not ready, show a repair-oriented message:

`FullCircle runtime is missing or damaged. Repair install.`

This should be treated as a repair path, not the normal fresh-install state.

### SAM 3 setup block

When SAM 3 is selected and not ready:

- show a setup checklist
- show clear current status
- show one next action
- show `Check Setup` / `Re-check Setup`
- show `Use FullCircle instead` helper text

Suggested visible checklist:

- HuggingFace account
- SAM 3 access approved
- Token verified
- SAM 3 runtime installed
- Weights downloaded

### SAM 3 ready block

When SAM 3 is ready:

- show `Ready` badge
- show `Enable masking`
- show `Diagnostics`
- show prompts field
- show backend info
- optionally retain `Re-check Setup` for reassurance and repair

---

## Doctor / Check Setup Action

The `SAM 3 Doctor` should be a user-facing diagnostic and repair assistant.

Suggested button labels:

- `Check Setup`
- `Re-check Setup`

### What it should do

1. Inspect cached/local SAM 3 state.
2. Inspect token presence and, if appropriate, verify access.
3. Inspect runtime importability.
4. Inspect weight presence.
5. Return a structured report with status lines and one next action.

### What it should output

Example success:

```text
SAM 3 Setup
Token: Verified
Access: Granted
Runtime: Installed
Weights: Present
Status: Ready
```

Example pending:

```text
SAM 3 Setup
Token: Verified
Access: Pending approval
Runtime: Not installed
Weights: Not downloaded
Next step: Wait for HuggingFace approval, then click Re-check Setup.
```

Example invalid token:

```text
SAM 3 Setup
Token: Invalid
Access: Unknown
Runtime: Not installed
Weights: Not downloaded
Next step: Create a new HuggingFace token and verify again.
```

### Important product rule

This action should reduce user anxiety. It should never make the user guess whether the plugin itself is broken.

---

## Packaging and Install Contract Changes

This is the most important non-UI change in the proposal.

### Recommendation

If FullCircle is meant to be ready on a normal install, move the FullCircle runtime into the shipped install contract.

That means the plugin should ship with:

- `torch`
- `ultralytics`
- `segment-anything`
- `sam2`
- any runtime glue required for the current FullCircle path

SAM 3 remains optional.

### Consequence

This likely increases install size and install time.

That is the cost of making FullCircle honestly "ready now."

### Alternative

If install size is unacceptable, the product contract must change to:

- plugin works without masking
- FullCircle requires first-time install
- SAM 3 remains optional

But that is not the contract this proposal is optimizing for.

### Recommendation for this project

Optimize for:

- turnkey FullCircle
- optional SAM 3

That matches the intended user story more closely.

---

## Copy and Messaging Recommendations

## Fresh install messaging

Use:

- `Masking is optional.`
- `FullCircle is ready now.`
- `SAM 3 Cubemap is optional and requires HuggingFace approval.`

Avoid:

- `Not installed` on first launch if FullCircle is supposed to be shipped ready
- generic `Install Masking` language when only SAM 3 is optional

## SAM 3 messaging

Use explicit status text:

- `Token missing`
- `Token invalid`
- `Access pending approval`
- `Access granted`
- `Installing SAM 3 runtime`
- `Downloading SAM 3 weights`
- `SAM 3 ready`

Avoid merging these into a single vague sentence like:

- `Access denied or pending`
- `Install failed`

unless a more specific cause is truly unavailable.

## FullCircle reassurance

Always keep a reassurance line visible somewhere in the SAM 3 path:

`FullCircle remains available while SAM 3 setup is incomplete.`

---

## Proposed Implementation Sequence

## Phase 1: Lock the contract

1. Decide that fresh install means `plugin ready + FullCircle ready + SAM 3 optional`.
2. Update dependency/install strategy to make FullCircle honestly ready out of the box.
3. Keep masking disabled by default in the UI.

## Phase 2: Refactor state reporting

1. Keep current coarse readiness booleans.
2. Add structured SAM 3 setup report objects.
3. Change token verification helpers to return structured outcomes instead of bare `True/False`.
4. Separate access-pending, invalid-token, and install-failure states.

## Phase 3: Redesign panel flow

1. Keep FullCircle and SAM 3 visibly separate.
2. Make FullCircle the stable local path.
3. Add checklist-driven SAM 3 setup UI.
4. Add `Check Setup` / `Re-check Setup`.
5. Keep FullCircle reassurance visible during SAM 3 setup.

## Phase 4: Validate with real user scenarios

1. Fresh install, never use masking.
2. Fresh install, use FullCircle immediately.
3. SAM 3 with no HuggingFace account.
4. SAM 3 with access pending.
5. SAM 3 with invalid token.
6. SAM 3 install/import failure.
7. SAM 3 ready.

---

## Acceptance Criteria

This proposal is successful if:

- the plugin can be used normally after install without touching masking
- FullCircle is ready immediately on a normal install
- SAM 3 is clearly presented as optional
- the user can always tell what SAM 3 setup step is incomplete
- the user can always tell what to do next
- the user never has to infer whether the plugin itself is broken
- the user can always fall back to FullCircle while SAM 3 setup is incomplete

---

## Recommended Follow-Up

Once this contract is accepted, the next implementation spec should cover:

1. the exact dependency/install changes needed to make FullCircle ship ready
2. the exact `Sam3SetupReport` API and supporting helpers
3. the panel/RML changes for checklist-driven SAM 3 onboarding
4. the `Check Setup` action and copy
5. the first-run acceptance tests
