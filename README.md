# zig-half-rs

**f16/bf16 ML operations for Rust** — Companion to https://github.com/gHashTag/zig-half

## Features

- **Ternary pack/unpack** (2-bit encoding)
  - `pack_16` / `unpack_16` — 16 trits ↔ 32 bits
  - Encoding: -1→`01`, 0→`00`, +1→`10`
  - `count_trits` — Count -1, 0, +1 occurrences
  - `compression_ratio` — Calculate 4× memory savings

- **Sparse ternary operations**
  - `ternary_dot_sparse` — Zero-skip optimization (30-50% faster on 66% sparse)
  - `count_zero_chunks` / `sparsity_ratio` / `estimate_speedup` — Analysis

- **Shadow weight storage**
  - `ShadowStorage` — f16 gradient accumulation with periodic sync
  - 2× memory savings vs f32, better cache locality
  - `quantize_to_ternary` — f16 → {-1, 0, +1}

## Installation

Add to `Cargo.toml`:

```toml
[dependencies]
zig-half = "0.1.0"
half = "2.4"
```

## Usage

### Ternary Packing

```rust
use zig_half::ternary::{pack_16, unpack_16};

fn main() {
    let trits: [i8; 16] = [-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0, 1, -1, 0];
    let packed = pack_16(trits);
    let unpacked = unpack_16(packed);

    assert_eq!(trits, unpacked);

    // Count trits
    let counts = zig_half::ternary::count_trits(&trits);
    println!("neg: {}, zero: {}, pos: {}", counts.neg, counts.zero, counts.pos);

    // Compression ratio (4×)
    let ratio = zig_half::ternary::compression_ratio(16);
    println!("Compression ratio: {:.2}×", ratio);
}
```

### Sparse Ternary Dot Product

```rust
use zig_half::sparse::ternary_dot_sparse;

fn main() {
    let trits: Vec<i8> = vec![1, 0, -1, 0, 1];
    let values: Vec<f16> = vec![0.5, 0.3, -0.7, 0.2, 0.5];
    let indices: Vec<usize> = vec![0, 2, 4]; // Only compute at these indices

    let result = ternary_dot_sparse(&trits, &values, &indices);

    println!("Sparse dot: {}", result);

    // Analysis
    let sparsity = zig_half::sparse::sparsity_ratio(&trits);
    let speedup = zig_half::sparse::estimate_speedup(&trits);
    println!("Sparsity: {:.2}%, Estimated speedup: {:.2}×",
             sparsity * 100.0, speedup);
}
```

### Shadow Weight Storage

```rust
use zig_half::shadow::ShadowStorage;

fn main() {
    let mut storage = ShadowStorage::with_defaults(128);

    // Initialize with ternary weights
    let initial_trits: Vec<i8> = vec![1, 0, -1; 128];
    storage.load_from_ternary(&initial_trits);

    // Add gradients
    let gradients: Vec<f16> = vec![0.1, -0.2, 0.05; 128];
    storage.add_gradients(&gradients);

    // Check sync status
    if storage.should_sync() {
        let result = storage.quantize_to_ternary();
        println!("Updated {} weights", result.updated);

        // Get statistics
        let stats = storage.stats();
        println!("Stats: min={:.3}, max={:.3}, mean={:.3}, std={:.3}",
                 stats.min, stats.max, stats.mean, stats.std);
    }
}
```

## Performance

- **Pure Rust** — No CGo overhead
- **Memory efficient** — Uses f16 (2 bytes) instead of f32 (4 bytes)
- **Zero-skip** — 30-50% faster on sparse data (66% zeros)

## Origin

Companion to https://github.com/gHashTag/zig-half — Full Zig implementation.

Extracted from https://github.com/gHashTag/trinity training infrastructure.

## License

MIT License — see LICENSE file.

## Contributing

PRs welcome! Please:
1. Follow Rust standard conventions
2. Add tests for new functions
3. Run `cargo clippy` before commit
4. Update documentation

## See Also

- https://github.com/gHashTag/zig-half — Zig implementation
- https://github.com/gHashTag/go-half — Pure Go version
- https://github.com/gHashTag/trinity — Full HSLM training
