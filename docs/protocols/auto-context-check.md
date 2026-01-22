# Auto-Context Check (Token-Waste Guard)

Purpose: avoid burning context on sequential waiting/polling when no new
information is arriving.

## Trigger Signals

Run this check whenever you notice any of the following:

- You executed 2+ status-only commands in a row with no state change.
  Examples: `gt hook`, `gt mail inbox`, `bd ready`, `gt trail`.
- You are "waiting" on another agent or human without a concrete next action.
- You are monitoring logs/feeds without taking action for several minutes.

## Decision

1) **Action available?** Do the next real action immediately.
2) **No action available?** Hand off with a short summary so a fresh session
   can resume when new information arrives.

### Handoff

```bash
gt handoff -s "Auto-context check" -m "Waiting on <thing>. Next action: <what to do when it arrives>."
```

## Notes

- This is a *token-waste* guard, not a completeness gate.
- If you're blocked by a decision, escalate instead of waiting.
