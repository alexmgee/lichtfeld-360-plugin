# Masking Setup UX — Implementation Plan

**Date:** 2026-04-08  
**Status:** Proposed implementation plan  
**Spec:** `docs/specs/2026-04-08-masking-setup-ux-proposal.md`

---

## Goal

Implement the masking setup experience so the plugin lands with this product contract:

1. The plugin is usable immediately after install even if masking is never enabled.
2. FullCircle is the built-in local masking path and is ready immediately on a normal install.
3. SAM 3 Cubemap is an optional alternate path with guided onboarding.
4. The UI always makes it clear that FullCircle remains available even if SAM 3 setup is incomplete.

---

## Decision Gate

This plan assumes the project accepts the following product choice:

- `FullCircle ships ready`
- `SAM 3 remains optional`

If that contract changes, this plan should be revised before implementation begins.

The main consequence is that the FullCircle runtime can no longer be treated as a first-run optional install path.

---

## Current Mismatch

Today:

- the plugin can run with masking disabled
- FullCircle is selected by default in the UI
- FullCircle is not actually ready on a fresh install because `sam2` is still optional
- the SAM 3 setup flow exists, but its state model is too compressed for clear user guidance

This plan fixes both problems:

- align packaging with the intended FullCircle contract
- make SAM 3 onboarding explicit and low-anxiety

---

## Scope

## In scope

- FullCircle fresh-install readiness
- masking section first-run UX
- SAM 3 onboarding checklist and status model
- `Check Setup` / `Re-check Setup` action
- clearer token/access/install/weights messaging
- tests for first-run and setup-state behavior

## Out of scope

- changing SAM 3 masking quality or geometry logic
- replacing SAM 2 with SAM 3.1 video tracking
- redesigning unrelated extraction/reframe/alignment UI

---

## File Map

| Action | File | Responsibility |
|--------|------|---------------|
| Modify | `pyproject.toml` | Move FullCircle runtime into the shipped install contract |
| Modify | `uv.lock` | Regenerate lock for the new FullCircle default dependency set |
| Modify | `core/setup_checks.py` | Add richer SAM 3 setup reporting and repair-oriented helpers |
| Modify | `panels/prep360_panel.py` | Implement first-run messaging, setup checklist state, and doctor action |
| Modify | `panels/prep360_panel.rml` | Add explicit FullCircle/SAM 3 setup blocks and checklist UI |
| Create or Modify | `tests/test_setup_checks.py` | Cover FullCircle-ready fresh-install state and SAM 3 status distinctions |
| Create | `tests/test_prep360_setup_flow.py` | Cover panel-facing setup-state rendering logic without requiring the full UI runtime |

---

## Phase 1: Lock the Fresh-Install Contract

### Task 1: Make FullCircle part of the normal shipped runtime

**Files:**
- `pyproject.toml`
- `uv.lock`

**Intent:**

If FullCircle is supposed to be ready immediately, its runtime must ship with the normal install.

**Implementation:**

- move the current FullCircle-required runtime into the base dependency contract
- treat `sam2` as part of the shipped FullCircle path, not as an optional first-run add-on
- keep SAM 3 under the optional `sam3-masking` extra

**Notes:**

- this increases install size and initial sync time
- that is an explicit product tradeoff in exchange for turnkey FullCircle

**Acceptance criteria:**

- a fresh install reports `fullcircle_ready == True`
- no FullCircle install prompt appears in the normal first-run path

---

## Phase 2: Refactor Setup State Reporting

### Task 2: Keep coarse readiness booleans, add a richer SAM 3 setup report

**Files:**
- `core/setup_checks.py`

**Intent:**

The pipeline still needs simple booleans, but the UI needs richer SAM 3 setup state.

**Implementation:**

- preserve:
  - `default_tier_ready`
  - `fullcircle_ready`
  - `sam3_ready`
  - `is_ready`
- add a dedicated `Sam3SetupReport` dataclass or equivalent structured return type

Suggested fields:

```python
@dataclass
class Sam3SetupReport:
    token_status: str
    access_status: str
    runtime_status: str
    weights_status: str
    overall_stage: str
    message: str
    next_action: str
```

### Task 3: Split ambiguous SAM 3 failure modes

**Files:**
- `core/setup_checks.py`

**Intent:**

Replace generic success/failure SAM 3 checks with distinct outcomes.

**Implementation:**

- change token verification helpers so they can distinguish:
  - missing token
  - invalid token
  - network error
  - access pending
  - access granted
- add runtime/install checks that distinguish:
  - runtime missing
  - runtime broken
  - weights missing
  - weights present

**Acceptance criteria:**

- UI logic can distinguish `token invalid` from `access pending`
- UI logic can distinguish `runtime broken` from `weights missing`

---

## Phase 3: Redesign the Masking Section Flow

### Task 4: Make the masking section reflect the product contract

**Files:**
- `panels/prep360_panel.py`
- `panels/prep360_panel.rml`

**Intent:**

The masking section should clearly separate:

- `FullCircle`
- `SAM 3 Cubemap`

without making SAM 3 setup feel like the plugin itself is blocked.

**Implementation:**

- keep method selection visible at all times
- make masking disabled by default on fresh install
- when FullCircle is selected and ready:
  - show `Ready`
  - show local/no-account language
  - show enable toggle
- when SAM 3 is selected and not ready:
  - show checklist-driven setup
  - show current status
  - show next action
  - show FullCircle reassurance text

Suggested reassurance copy:

`FullCircle remains available while SAM 3 setup is incomplete.`

### Task 5: Add `Check Setup` / `Re-check Setup`

**Files:**
- `panels/prep360_panel.py`
- `panels/prep360_panel.rml`
- `core/setup_checks.py`

**Intent:**

Give the user a simple way to understand the real SAM 3 status without guessing.

**Implementation:**

- add button:
  - `Check Setup`
  - `Re-check Setup`
- wire it to produce a fresh `Sam3SetupReport`
- render:
  - token status
  - access status
  - runtime status
  - weights status
  - next action

**Acceptance criteria:**

- user can tell exactly which SAM 3 setup step is incomplete
- no generic “denied or pending” message remains if a more specific cause is known

---

## Phase 4: Simplify Copy and Messaging

### Task 6: Replace vague setup copy with status-driven copy

**Files:**
- `panels/prep360_panel.py`
- `panels/prep360_panel.rml`

**Implementation goals:**

- FullCircle fresh install:
  - `FullCircle is ready on this install. No HuggingFace account required.`
- SAM 3 setup:
  - `Token missing`
  - `Token invalid`
  - `Access pending approval`
  - `Access granted`
  - `Installing SAM 3 runtime`
  - `Downloading SAM 3 weights`
  - `SAM 3 ready`

**Acceptance criteria:**

- no first-run FullCircle state says `Not installed` in the normal shipped path
- no SAM 3 failure path merges obviously different states into a single vague sentence

---

## Phase 5: Validate the End-to-End Flows

### Task 7: Add setup-state and first-run tests

**Files:**
- `tests/test_setup_checks.py`
- `tests/test_prep360_setup_flow.py`

**Test matrix:**

1. Fresh install, masking disabled:
   - plugin usable
   - no SAM 3 blocking state forced
2. Fresh install, FullCircle selected:
   - `fullcircle_ready == True`
   - no install prompt
3. SAM 3 with no token:
   - `needs_token`
4. SAM 3 with invalid token:
   - `invalid token` state
5. SAM 3 with pending access:
   - `access pending` state
6. SAM 3 with granted access but runtime missing:
   - `ready_to_install`
7. SAM 3 with runtime present but weights missing:
   - `needs_weights`
8. SAM 3 ready:
   - `sam3_ready == True`

### Task 8: Manual UX validation

**Manual checklist:**

1. Fresh install opens without masking pressure.
2. FullCircle path is immediately usable.
3. SAM 3 path clearly explains each setup step.
4. `Check Setup` updates status without requiring guesswork.
5. FullCircle remains visibly available throughout SAM 3 onboarding.

---

## Suggested Implementation Order

1. Phase 1 first: packaging contract
2. Phase 2 second: richer setup report
3. Phase 3 third: panel flow
4. Phase 4 fourth: copy cleanup
5. Phase 5 last: tests and manual validation

This order matters because the panel UX should reflect the true shipped contract, not work around an old packaging model.

---

## Risks

## Risk 1: Install-size increase

Moving FullCircle runtime into the default contract likely increases install size and sync time.

Mitigation:

- accept that cost explicitly as the tradeoff for turnkey FullCircle

## Risk 2: SAM 2 environment fragility

The plugin has already seen SAM 2 environment fragility.

Mitigation:

- keep existing SAM 2 repair/validation logic
- validate fresh-install FullCircle readiness in a clean environment

## Risk 3: UI complexity

Adding richer setup states can make the masking section feel too busy.

Mitigation:

- show one primary next action at a time
- keep FullCircle compact and reassuring
- reserve checklist detail for the SAM 3 branch only

---

## Acceptance Criteria

This plan is complete when:

- fresh install means the plugin is usable without touching masking
- FullCircle is ready immediately on a normal install
- SAM 3 is clearly optional
- SAM 3 setup shows explicit state and next action
- FullCircle remains visibly available while SAM 3 setup is incomplete
- tests cover the main first-run and SAM 3 setup-state branches

---

## Recommended Next Step After This Plan

Start with **Phase 1: packaging contract alignment**.

If we do not first decide and implement whether FullCircle truly ships ready, the rest of the setup UX risks being a polished explanation for the wrong product behavior.
