---
title: "NEVER Patch Fix Problems - Debug The Actual New Code"
category: process-errors
tags: [debugging, discipline, new-features, critical]
date: 2025-12-27
severity: critical
component: development-process
---

# NEVER Patch Fix Problems - Debug The Actual New Code

## Problem Symptom

When adding new feature (sediment particles) to working system (water simulation), the new feature didn't work. Claude's response was to:

1. Look at unrelated todos about damping and gravity
2. Change particle diameters multiple times
3. Add velocity boosts
4. Add untested entrainment mechanisms
5. Change gravity constants
6. Modify existing working code

**None of this was correct. It made everything worse.**

## Root Cause

The bug was in the NEW code, not the existing systems.

When something breaks after adding new code:
- The EXISTING system was working
- The NEW code is the problem
- Patching the existing system breaks more things

## The Absolute Rule

```
Working System + New Code = Problem
                    ↓
            Problem is in New Code
                    ↓
            Debug New Code ONLY
                    ↓
            DO NOT touch working systems
```

## What To Do Instead

### 1. Acknowledge the new code is the problem
- Water worked → water code is fine
- Sediment is new → sediment code is suspect

### 2. Debug the actual new function
- Read it carefully line by line
- Add debug output to understand behavior
- Compare to expected physics

### 3. Check what already exists
- Don't add new mechanisms without checking if they exist
- Don't duplicate functionality

### 4. Write tests for the new code
- Before or alongside implementation
- Verify expected behavior in isolation

## What NOT To Do

- ❌ Change particle diameters
- ❌ Add velocity boosts
- ❌ Modify gravity constants
- ❌ Touch damping values
- ❌ Add new mechanisms to working systems
- ❌ Look at unrelated todos

## Key Takeaway

> **When something doesn't work after adding new code, the bug is in the NEW code, not the existing system. Debug the new code. Don't patch everything else.**
