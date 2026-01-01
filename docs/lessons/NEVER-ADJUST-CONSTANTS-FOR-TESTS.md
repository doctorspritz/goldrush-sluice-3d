# NEVER ADJUST PHYSICS CONSTANTS TO MAKE TESTS PASS

## The Rule

If a physics constant doesn't pass a sanity test, **THE SIMULATION IS BROKEN ELSEWHERE**.

Do NOT change the constant. Find and fix the actual bug.

## Why This Matters

Physics constants come from literature, research, and real-world measurements. They have correct values for a reason:

- Vorticity confinement epsilon: 0.01-0.1 (Fedkiw 2001)
- Gravity: 9.81 m/s^2
- Water density: 1000 kg/m^3

When a test fails with correct constants, the failure is telling you something is wrong in the simulation logic, not the constants.

## What Happens When You Ignore This

Example: Vorticity confinement was set to 40.0 instead of 0.05 to make tests pass.

Result: 400-800x the recommended value caused massive artificial turbulence throughout the simulation. The water looked "genuinely horrible" - chaotic velocity vectors, fragmented surface, complete garbage.

The "fix" that made tests pass introduced a far worse bug that took significant debugging to find.

## The Correct Approach

1. Test fails with correct constant
2. STOP - do not change the constant
3. Ask: "Why does the simulation produce wrong output with correct physics?"
4. Debug the simulation logic, not the constants
5. Fix the actual bug
6. Tests now pass with correct constants

## Sanity Checks for Constants

Before committing any constant change, ask:
- Is this value within literature-recommended ranges?
- Does this value make physical sense?
- Am I changing this to make tests pass, or because the value is actually wrong?

If the answer to #3 is "to make tests pass", you're about to create a cascading disaster. Stop and find the real bug.
