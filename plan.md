# Option 1: Human Participant in TUI — Implementation Plan

## Goal

Add a human participant to the TUI roundtable discussion with inline modal prompts, supporting:
- Human input during model responses (typing window)
- Sequential model responses (one at a time)
- Visual differentiation of human responses in the transcript

---

## Part 1: Fix the Orchestrator (sequential loop)

### 1. `core/discussion.py`

- Add `from datetime import datetime` import (used on line 359, currently missing)
- Rewrite `run()` method to use **sequential** model loop instead of `asyncio.gather`:
  - Each model responds one at a time
  - `await asyncio.sleep(0)` between models to keep TUI event loop responsive
  - `await self._notify_progress()` after each model so TUI updates in real-time
- Remove `_generate_single_response()` helper (no longer needed; inlined into sequential loop)
- Human input logic at lines 429-472 stays the same (fires after all models complete)

### 2. `core/config.py` — `HumanParticipantConfig`

- Add `display_name: str = "Human"` field (default display name in the TUI)

---

## Part 2: Add Human Participant Support to TUI

### 3. `tui/screens.py` — `HumanInputScreen` (new modal)

- Title shows round number + display name (e.g., "Round 2 — Your Turn to Respond")
- `TextArea` pre-filled with: the discussion prompt + round context (previous summary for round 2+)
- Submit/Cancel buttons
- `Submitted` message with the response text as payload

### 4. `tui/widgets.py`

- `TranscriptDisplay.set_human_display_name(name)` — new method to customize "human" display
- `TranscriptDisplay.add_response(model, content, round_num, human_display_name=None)` — detect `model == "human"` → render with yellow styling + display name
- `StatusPanel`: add a `has_human` reactive that shows the human display name alongside model info

### 5. `tui/app.py` — `RoundtableApp`

Add TUI state tracking:

- `self._last_known_round: int = 0` — detect new rounds
- `self._human_response: str | None = None` — store submitted text
- `self._human_pending: asyncio.Future[str] | None = None` — wait for submission if not yet typed

Create `human_input_callback`:

- If `_human_response` exists (submitted during typing window) → return it
- If not → create a Future, await it (TUI event loop keeps processing widget events)
- When `HumanInputScreen` is submitted, resolve the Future with the text

At detection of new round via `_on_progress_update` → push `HumanInputScreen` (with context from `session` + `current_round`)

On `HumanInputScreen.Submitted` → store response, resolve Future, pop screen
On `HumanInputScreen.Cancelled` → return empty string / handle gracefully

Pass `human_input_callback` when creating `DiscussionOrchestrator`

### 6. `tui/styles.css`

- Styles for `HumanInputScreen` modal (matching existing modal styling)
- Styles for human responses in the transcript (yellow/gray panel)

---

## Execution order

1. Fix orchestrator sequential loop + missing import
2. Add config display_name
3. Add HumanInputScreen
4. Update transcript/status-panel widgets
5. Wire up TUI app integration
6. Add CSS styles

---

# Option 2: Sequential CLI Orchestrator + Human Participant

## Goal

Rewrite the CLI orchestrator to run models sequentially (one at a time) for memory-constrained hardware, and fully implement human participant support across CLI. TUI functionality follows once CLI works correctly.

## Key Decisions

- **Sequential model execution** — models run one-by-one, not in parallel (required for hardware with limited memory)
- **CLI-first** — build and verify in CLI, then layer TUI on top
- **Human inputs during model sequence** — user can type while models respond; moderator summarizes after all participants (models + human)
- **Human position** — responds after all models (position N+1 of N+1)

---

## Phase 1: Fix CLI Bugs + Rewrite to Sequential Loop

### 1. `core/discussion.py` — Fix `datetime` import

`datetime.now()` is used on line 359 but not imported. Add:

```python
from datetime import datetime
```

### 2. `core/discussion.py` — Fix `model_configs[0]` bug

Line 404 uses `model_configs[0]` (first model config) for every model. Fix to index by position:

```python
model_configs = [m for m in self.config.models if m.name == model_name][0]
```

### 3. `core/discussion.py` — Rewrite `run()` to sequential

Replace `asyncio.gather(*tasks)` with a sequential loop:

```
For each round (1..max_rounds):
    1. Set up model order (rotation)
    2. For each model_name in model_order:
        a. Build context (previous summary + current round responses from models 1..i)
        b. Print "[Round N] {model_name} responding..."
        c. Generate response (await, do not parallelize)
        d. Save response to session, auto-save to disk
        e. Print completion info
    3. If human_participant.enabled:
        a. Start AsyncInputReader (background thread during model responses)
        b. Build human context (all models + previous summary)
        c. Get human input (timeout 90s, fallback to sync input)
        d. Stop reader, add human response
    4. Generate moderator summary (moderator sees all + human)
    5. Update session with attributed summary
    6. Check consensus
    7. If consensus reached → break, mark completed
    8. If not all rounds done → continue
```

**Context per model**: Each model sees the previous round's attributed summary + current round responses from models 1 through (position-1). `ContextConfig` mode (`full` / `summary_only` / `summary_plus_last_n`) determines exactly what's included.

**Human context**: Human sees the previous attributed summary + all current round model responses.

**Stoppability**: Check `self.state.is_running` between model responses so user can stop mid-round.

### 4. `core/discussion.py` — Verify human inclusion in summary

`_generate_summary()` at line 179 already combines `get_round_responses()` + `get_round_human_responses()`. Confirm moderator prompt includes human response.

### 5. `core/discussion.py` — Verify consensus includes human

`_check_consensus()` at line 270 already combines model + human responses. No changes needed.

---

## Phase 2: Human Participant CLI Flow

### 6. `core/discussion.py` — AsyncInputReader during model responses

Currently only started when `self.input_buffer` is truthy (TUI mode). Fix: always start during CLI when `human_participant.enabled`. During sequential model loop:

```
[Round 1] Model 1 responding...
[Round 1] Model 1 completed (2341 chars)
Your response (type while models respond): ← typing prompt during sequence
[Round 1] Model 2 responding...
...
[Round 1] Human completed (142 chars)
[Round 1] Moderator generating summary...
```

### 7. `core/discussion.py` — Fallback input (already exists)

Lines 436-455 handle fallback: if no input during model responses, show sync `input()` prompt after all models. Works as-is.

### 8. `core/config.py` — Optional: add `display_name` to `HumanParticipantConfig`

```python
class HumanParticipantConfig(BaseModel):
    enabled: bool = False
    prompt: str = "Share your perspective on: {prompt}"
    display_name: str = "Human"  # for IDs in session data / exports
```

---

## Phase 3: CLI Output Improvements

### 9. `main.py` / `core/discussion.py` — Start-of-discussion header

Print clearly:

```
Starting roundtable discussion...
Models: qwen3.5:35b, qwen3.5:27b, ...
Human: enabled
Max rounds: 10
Context: summary_only
```

### 10. `core/discussion.py` — Model output (already done)

Lines 345-353 already print model-by-model. Sequential naturally shows one-at-a-time.

### 11. `core/discussion.py` — Summary display (already done)

Lines 474-491 print summary + moderator assessment. Optionally add human contribution line.

### 12. `main.py` — Final results (already done)

Lines 196-218 handle "DISCUSSION COMPLETE". Works as-is.

---

## Phase 4: TUI (After CLI Works)

### 13. `tui/widgets.py` — Human response styling

`TranscriptDisplay.add_response()`: add optional `is_human` parameter for yellow/green highlighting.

### 14. Remaining TUI

No structural changes needed — `app.py`, `screens.py`, `core/config.py`, `storage/session.py`, `core/consensus.py`, `core/similarity.py` are all compatible. The orchestrator's more-frequent progress updates (model-by-model vs round-by-round) benefit UI responsiveness.

---

## File Change Summary

| File | Changes |
|---|---|
| `core/discussion.py` | Add `datetime` import; fix `model_configs[0]` bug; rewrite `run()` to sequential loop; ensure context building is incremental; start `AsyncInputReader` unconditionally in CLI |
| `core/config.py` | Optional: add `display_name` to `HumanParticipantConfig` |
| `tui/widgets.py` | Minor: add human response styling |
| `tui/screens.py` | No changes |
| `tui/app.py` | No changes |
| `storage/session.py` | No changes |
| `storage/export.py` | No changes |
| `tests/test_config.py` | Optional: add test for `display_name` |
| `.gitignore` | No changes |

---

## Execution Order

1. Fix `datetime` import
2. Fix `model_configs[0]` bug
3. Rewrite `run()` to sequential, verify with `--prompt`
4. Verify human participant input flow (enable `human_participant.enabled: true` in config)
5. Verify moderation includes human responses
6. Verify consensus detection with human
7. Run `pytest` — all existing tests + add human participant integration test
8. TUI: verify compatibility, add human response widget styling
