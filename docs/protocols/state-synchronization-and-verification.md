# State Synchronization and Verification Standards

Purpose: keep all review passes grounded in the same repo state and ensure
claims are reproducible.

Scope: Brains Trust planning/completion reviews, polecat work claims, and
handoffs that assert test results or behavior changes.

## State Synchronization (Before Review or Claim)

1) Identify the target bead and branch.
2) Sync git state:
   - `git fetch --all --prune`
   - If your branch tracks a remote and you are expected to review the latest
     head, `git pull --rebase`.
   - If you cannot pull (detached head, no remote, or permissions), record the
     exact commit you reviewed.
3) Sync beads state:
   - If you created or updated beads, run `bd sync` before ending the session.
   - If you rely on issue state, run `bd sync` to import the latest JSONL.
4) Record a **state stamp** in your notes:
   - branch name
   - commit SHA
   - dirty/clean working tree
   - sync time (UTC)
   - target bead id

## Verification Standards

- Every behavioral claim must be backed by a reproducible command or a direct
  file reference (path + line).
- Never state "tests pass" unless you ran them.
- Prefer deterministic commands (exact flags, exact example names).
- If a claim is based on manual inspection, describe the exact steps.

### Verification Block (Required for Claims)

```
Verification:
- Command: `<command>`
- Expected: <what should happen>
- Result: <pass/fail + summary>
- Commit: <sha>
- When: <UTC timestamp>
- Env: <toolchain/OS if relevant>
- Notes: <any caveats>
```

### Manual / Visual Verification

Include:
- Steps and inputs
- Expected output
- Observed output
- Files or screenshots referenced

### If Not Run

Be explicit:
```
Not run: <reason>
Risk: <what could be wrong>
```

## Re-run Triggers

If you pull new commits or rebase after verification, re-run affected checks
and update the verification block.
