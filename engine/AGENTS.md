# Repository Guidelines

## Project Structure & Module Organization
- `src/` holds all Rust sources. Core chess logic lives in `chess.rs`, the NNUE interface in `nnue.rs`/`nnue_runtime.rs`, and search/time/TT logic in `search.rs`, `time_manager.rs`, and `tt.rs`.  
- Executables: `src/main.rs` runs the engine; `bin/` contains helper binaries (e.g., training scripts).  
- Data and configs: `dataset.csv`, `rustfmt.toml`, and `build.rs`. Test cases live alongside the modules they cover (see `#[cfg(test)]` blocks at file bottoms).

## Build, Test, and Development Commands
All commands require LibTorch; set the variables every run (or source `~/.bashrc`):
```bash
LIBTORCH=/home/undec/libtorch \
LD_LIBRARY_PATH=/home/undec/libtorch/lib:${LD_LIBRARY_PATH:-} \
cargo build --release
```
- `cargo fmt` – applies the repo’s formatting rules (runs rustfmt with `rustfmt.toml`).  
- `cargo clippy` – lints with the default configuration; fix or allow every warning before sending patches.  
- `cargo test` – executes unit tests in `src/`; prefer targeted runs (`cargo test chess::tests::pinned_piece_cannot_move`) when iterating.
- Regression check after `cargo test` (same env):  
  ```bash
  cargo run --release --bin suckfish -- --nnue-path ../nnue.ot
  ```  
  When prompted, send `go 60000 3r3k/5r1p/4Rp1P/2b5/5PQ1/2q2N2/1p4P1/4R2K w - - 4 34` on one line. Confirm the engine reports `bestmove d6d8` and note the nps/depth for monitoring.

## Coding Style & Naming Conventions
- Follow idiomatic Rust: 4-space indentation, snake_case for functions/vars, UpperCamelCase for types/traits.  
- Keep helper functions private unless a public API is required.  
- Prefer explicit bitboard helpers (e.g., `sliding_attacks_from`) and document non-obvious math with short comments.  
- Always run `cargo fmt` before committing; rustfmt is the source of truth.

## Testing Guidelines
- Tests use Rust’s built-in framework (`#[test]`). Place small unit tests near the code; integration tests can live under `tests/` if they grow complex.  
- Name tests descriptively (`pinned_piece_cannot_move`). Cover both rules logic and search heuristics when feasible.  
- Ensure `cargo test` passes with the LibTorch env set; this loads NNUE layers even when not directly exercised.

## Commit & Pull Request Guidelines
- Commit messages follow short, imperative summaries (see log: “Fix env problem”, “Upgrade board legality”). Keep them under ~70 characters and focus on “what”.  
- For pull requests: describe the change, include reproduction steps or profiling numbers if relevant, link issues, and attach screenshots/logs for UI or perf tools.  
- Rebase rather than merge when updating a branch; keep history linear and clean.

## Environment & Security Tips
- NNUE evaluation dynamically loads shared libraries; never commit proprietary weights.  
- Check `.env` or shell exports for secrets before sharing logs.  
- Use `LIBTORCH=/home/undec/libtorch` rather than system installs to avoid ABI mismatches.
