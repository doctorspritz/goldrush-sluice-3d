# WARP.md

This file provides guidance to WARP (warp.dev) when working with code in this repository.

## Repository structure (big picture)
- This repo currently contains two standalone Rust crates under `crates/`:
  - `crates/sim`: a library crate intended for the core simulation logic (public API in `crates/sim/src/lib.rs`).
  - `crates/game`: a binary crate intended for the game / executable (entrypoint in `crates/game/src/main.rs`).
- There is no top-level `Cargo.toml` workspace. Use `--manifest-path` (recommended) or run `cargo` from within each crate directory.

## Common dev commands
All commands below can be run from the repo root.

### Build / check
- Build `sim`:
  - `cargo build --manifest-path crates/sim/Cargo.toml`
- Build `game`:
  - `cargo build --manifest-path crates/game/Cargo.toml`
- Fast typecheck (no artifacts beyond checks):
  - `cargo check --manifest-path crates/sim/Cargo.toml`
  - `cargo check --manifest-path crates/game/Cargo.toml`

### Run
- Run the `game` binary:
  - `cargo run --manifest-path crates/game/Cargo.toml`

### Tests
- Run all tests in `sim`:
  - `cargo test --manifest-path crates/sim/Cargo.toml`
- Run a single test by substring match (example: `it_works`):
  - `cargo test --manifest-path crates/sim/Cargo.toml it_works`

### Format / lint
- Format (Rustfmt):
  - `cargo fmt --manifest-path crates/sim/Cargo.toml`
  - `cargo fmt --manifest-path crates/game/Cargo.toml`
- Lint (Clippy), failing on warnings:
  - `cargo clippy --manifest-path crates/sim/Cargo.toml -- -D warnings`
  - `cargo clippy --manifest-path crates/game/Cargo.toml -- -D warnings`

## Architecture notes
- `sim` is where reusable simulation logic should live; keep its public surface area explicitly `pub` from `crates/sim/src/lib.rs`.
- `game` is the executable; it should orchestrate IO/UI and call into `sim` rather than duplicating simulation logic.
- If `game` needs to depend on `sim`, add a local path dependency in `crates/game/Cargo.toml` pointing at `../sim` (then `use sim::...` from `main.rs`).
