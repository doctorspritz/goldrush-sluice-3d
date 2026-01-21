# Native arm64 build (macOS)

## Check shell + toolchain

```bash
arch
uname -m
sysctl -n sysctl.proc_translated
which rustc
rustc -vV
```

If `arch`/`uname -m` report `x86_64` (or `sysctl.proc_translated` is `1`), the shell is under Rosetta.
If `which rustc` resolves to `/usr/local/bin/rustc` (Homebrew x86_64), prefer the rustup-managed toolchain:

```bash
~/.cargo/bin/rustc -vV
arch -arm64 ~/.cargo/bin/rustc -vV
```

## Use the arm64 toolchain

This repo pins the toolchain via `rust-toolchain.toml`. Ensure the rustup toolchain is installed and the
arm64 cargo is used:

```bash
rustup target add aarch64-apple-darwin
```

If the current shell is x86_64, run the arm64 cargo explicitly:

```bash
arch -arm64 ~/.cargo/bin/cargo -V
```

## Build

```bash
arch -arm64 ~/.cargo/bin/cargo build -p game --release
```

## Verify

```bash
file target/release/game
```

Expected: `Mach-O 64-bit executable arm64`.
