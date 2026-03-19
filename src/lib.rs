//! f16/bf16 ML operations for Rust
//! Companion to https://github.com/gHashTag/zig-half

use half::{f16, bf16};

// TERNARY PACKING — 2-bit encoding {-1, 0, +1} -> {01, 00, 10}
pub mod ternary {
    pub const TRIT_NEG: u8 = 0b01;
    pub const TRIT_ZERO: u8 = 0b00;
    pub const TRIT_POS: u8 = 0b10;

    pub fn pack_16(trits: [i8; 16]) -> u32 {
        let mut result: u32 = 0;
        for (i, &t) in trits.iter().enumerate() {
            let bits: u8 = match *t {
                -1 => TRIT_NEG,
                0 => TRIT_ZERO,
                1 => TRIT_POS,
                _ => TRIT_ZERO,
            };
            result |= (bits as u32) << (i * 2);
        }
        result
    }

    pub fn unpack_16(packed: u32) -> [i8; 16] {
        let mut trits: [i8; 16] = [0; 16];
        for (i, t) in trits.iter_mut().enumerate() {
            let bits: u8 = ((packed >> (i * 2)) & 0b11) as u8;
            *t = match bits {
                TRIT_NEG => -1,
                TRIT_ZERO => 0,
                TRIT_POS => 1,
                _ => 0,
            };
        }
        trits
    }

    pub fn count_trits(trits: &[i8]) -> TritCounts {
        let mut result = TritCounts { neg: 0, zero: 0, pos: 0 };
        for &t in trits {
            match *t {
                -1 => result.neg += 1,
                0 => result.zero += 1,
                1 => result.pos += 1,
                _ => {}
            }
        }
        result
    }

    pub fn compression_ratio(trit_count: usize) -> f64 {
        let original_bytes = trit_count;
        let packed_bytes = (trit_count * 2 + 7) / 8;
        original_bytes as f64 / packed_bytes as f64
    }
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TritCounts {
    pub neg: usize,
    pub zero: usize,
    pub pos: usize,
}

pub mod sparse {
    pub fn ternary_dot_sparse(trits: &[i8], values: &[f16], indices: &[usize]) -> f64 {
        let mut sum: f64 = 0.0;
        for &idx in indices {
            let w = trits[*idx];
            if w == 0 {
                continue;
            }
            let a: f32 = values[*idx].to_f32();
            sum += (w as f64) * (a as f64);
        }
        sum
    }

    pub fn count_zero_chunks(trits: &[i8]) -> usize {
        const CHUNK_SIZE: usize = 16;
        let mut zero_count: usize = 0;
        for chunk in trits.chunks(CHUNK_SIZE) {
            let all_zero = chunk.iter().all(|&w| *w == 0);
            if all_zero {
                zero_count += 1;
            }
        }
        zero_count
    }

    pub fn sparsity_ratio(trits: &[i8]) -> f64 {
        if trits.is_empty() {
            return 0.0;
        }
        let zero_count = trits.iter().filter(|&&w| **w == 0).count();
        zero_count as f64 / trits.len() as f64
    }

    pub fn estimate_speedup(trits: &[i8]) -> f64 {
        let total_chunks = trits.len() / 16;
        if total_chunks == 0 {
            return 1.0;
        }
        let zero_chunks = count_zero_chunks(trits);
        let zero_chunk_ratio = zero_chunks as f64 / total_chunks as f64;
        1.0 + zero_chunk_ratio * 0.5
    }
}

pub mod shadow {
    use half::f16;

    pub const DEFAULT_SYNC_INTERVAL: usize = 100;
    pub const DEFAULT_QUANTIZE_THRESHOLD: f16 = f16::from_f32(0.5);

    pub struct ShadowStorage {
        weights: Vec<f16>,
        step: usize,
        sync_interval: usize,
        threshold: f16,
    }

    impl ShadowStorage {
        pub fn new(capacity: usize, sync_interval: usize, threshold: f16) -> Self {
            Self {
                weights: vec![f16::from_f32(0.0); capacity],
                step: 0,
                sync_interval,
                threshold,
            }
        }

        pub fn with_defaults(capacity: usize) -> Self {
            Self::new(capacity, DEFAULT_SYNC_INTERVAL, DEFAULT_QUANTIZE_THRESHOLD)
        }

        pub fn add_gradients(&mut self, gradients: &[f16]) {
            let count = std::cmp::min(gradients.len(), self.weights.len());
            for i in 0..count {
                self.weights[i] += gradients[i];
            }
            self.step += 1;
        }

        pub fn should_sync(&self) -> bool {
            self.step >= self.sync_interval
        }

        pub fn quantize_to_ternary(&mut self) -> QuantizeResult {
            let mut trits: Vec<i8> = Vec::with_capacity(self.weights.len());
            let mut updated: usize = 0;
            let threshold_f32: f32 = self.threshold.to_f32();

            for &w in &self.weights {
                let w_abs = w.to_f32().abs();
                if w_abs < 1e-6 {
                    trits.push(0);
                } else if w > self.threshold {
                    trits.push(1);
                    updated += 1;
                } else if w < -self.threshold {
                    trits.push(-1);
                    updated += 1;
                } else {
                    trits.push(0);
                    updated += 1;
                }
            }

            self.step = 0;

            QuantizeResult { trits, updated }
        }

        pub fn load_from_ternary(&mut self, trits: &[i8]) {
            let count = std::cmp::min(trits.len(), self.weights.len());
            for i in 0..count {
                self.weights[i] = f16::from_i32(trits[i] as i32);
            }
        }

        pub fn sparsity(&self) -> f64 {
            let zero_count = self.weights.iter()
                .filter(|&&w| w.to_f32().abs() < 1e-6)
                .count();
            zero_count as f64 / self.weights.len() as f64
        }

        pub fn stats(&self) -> WeightStats {
            let mut min: f32 = f32::INFINITY;
            let mut max: f32 = f32::NEG_INFINITY;
            let mut sum: f64 = 0.0;

            for w in &self.weights {
                let w_f32 = w.to_f32();
                if w_f32 < min {
                    min = w_f32;
                }
                if w_f32 > max {
                    max = w_f32;
                }
                sum += w_f32 as f64;
            }

            let mean = sum / self.weights.len() as f64;
            let mean_f32 = mean as f32;

            let mut var_sum: f64 = 0.0;
            for w in &self.weights {
                let w_f32 = w.to_f32();
                let diff = w_f32 - mean_f32;
                var_sum += diff * diff;
            }

            let std_f32 = (var_sum / self.weights.len() as f64).sqrt() as f32;

            WeightStats { min, max, mean: mean_f32, std: std_f32 }
        }

        pub fn reset(&mut self) {
            for w in self.weights.iter_mut() {
                *w = f16::from_f32(0.0);
            }
            self.step = 0;
        }
    }

    #[derive(Debug)]
    pub struct QuantizeResult {
        pub trits: Vec<i8>,
        pub updated: usize,
    }

    #[derive(Debug)]
    pub struct WeightStats {
        pub min: f32,
        pub max: f32,
        pub mean: f32,
        pub std: f32,
    }
}
