---
id: "003"
status: pending
priority: p3
title: Fix gaps in deposited piles
created: 2025-12-28
---

# Fix gaps in deposited piles

## Problem

DEM settling produces mostly coherent piles, but some have holes where particles didn't pack tightly enough before deposition threshold was met.

## Potential Fixes

1. **Lower MIN_NEIGHBORS**: Currently 3, could try 2
2. **Smaller particle radius**: Tighter packing geometry
3. **Flood-fill post-process**: After deposition, fill isolated gaps
4. **Two-pass deposition**: First pass marks candidates, second pass fills gaps
5. **Higher MASS_PER_CELL**: Require more particles before converting

## Notes

Low priority - piles are forming in correct locations and mostly coherent. This is polish.
