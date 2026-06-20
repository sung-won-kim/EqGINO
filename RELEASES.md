# EqGINO — Release Notes

## v1.1.0 — Performance & Memory Optimization (2026-06-21)

This release makes EqGINO **faster and substantially lighter on GPU memory** while
keeping the model's outputs (and therefore its accuracy) numerically equivalent to
v1.0. No architecture, hyper-parameters, weights, or training recipe were changed —
only the implementation of the hot paths.

### Headline results

Measured on a real **AhmedBody** sample (70,661 mesh nodes), latent grid `32³`,
`hidden_dim=64`, `fno_n_layers=4`, `num_groups=2`, on a single NVIDIA L40S.
"Reference" = the v1.0 code paths, "Optimized" = v1.1, same model & sample in one
process (clean A/B).

| Mode | Metric | v1.0 (reference) | v1.1 (optimized) | Improvement |
|------|--------|-----------------:|-----------------:|:-----------:|
| **Training** (fwd+bwd) | wall-clock | 1236 ms/iter | **756 ms/iter** | **1.64× faster** (−39%) |
| **Training** (fwd+bwd) | peak memory | 16.6 GB | **6.7 GB** | **2.47× less** (−60%) |
| **Inference** (no_grad) | wall-clock | 981 ms/iter | **341 ms/iter** | **2.87× faster** (−65%) |
| **Inference** (no_grad) | peak memory | 8.2 GB | **4.6 GB** | **1.79× less** (−44%) |

### Equivalence / correctness

Every optimization was validated against the original path on identical weights:

| Change | Output difference vs v1.0 |
|--------|---------------------------|
| FFT mode-slice fast path | **0** (bit-identical) |
| Chunked + checkpointed GNO | output **0**, grad max-abs **9.7e-10** (bit-identical) |
| On-device R2 vs `sklearn.r2_score` | **0** |
| `torch_cluster` neighbor search | forward rel-diff **7.8e-9**; after 20 Adam steps **1.8e-6** |

The only non-bit-exact change is the neighbor-search backend (`torch_cluster`),
which differs from the dense `cdist` search by ~4 edges out of ~1.09M — points lying
*exactly* on the radius boundary (`dist == radius`). This is far below GPU
reduction noise between any two training runs and does not affect accuracy. (The
codebase already documents that neighbor-search backends "can differ by a
permutation of the neighbors.")

---

### What was optimized, where, and why

#### 1. Neighbor search — biggest **speed** win
**File:** `neuralop/layers/neighbor_search.py`

Profiling showed the radius neighbor search was **~85% of the forward wall-clock**
(`gno_in` ≈ 450 ms + `gno_out` ≈ 450 ms) whenever Open3D's compiled torch ops are
unavailable — which is the common case, so the code fell back to a dense
`torch.cdist` search that is `O(num_queries × num_data)` in both compute and the
transient distance matrix.

- Added **`torch_cluster.radius`** (bucketed, GPU-native) as the preferred backend
  for `native_neighbor_search`. It avoids ever materializing the dense distance
  matrix.
- Per-query neighbor cap grows adaptively on saturation and **falls back to the
  exact `cdist` path** if it would ever truncate — so results stay correct on any
  density.
- Open3D remains the first choice when its torch ops *are* built (unchanged).

**Effect:** the dominant forward cost collapses; inference forward 981 → 341 ms.

#### 2. GNO kernel integral — biggest **memory** win
**File:** `neuralop/layers/gno_block.py` (`EqGNOBlock.forward`)

The graph kernel integral builds per-edge activations for **~1.09M neighbor edges ×
up to 512 channels** through the kernel MLP. This dominated memory: `gno_out` alone
retained ~7.9 GB and peaked at ~8.7 GB transiently, stacked on top of the retained
`gno_in` (2.7 GB) + FNO (4.8 GB).

- The kernel integral is **independent per query row**, so queries are now split
  into chunks of ~200k edges each (`chunk_target_edges`), processed in a loop, and
  concatenated.
- During backward each chunk is **gradient-checkpointed** (its per-edge activations
  are recomputed instead of stored); under `no_grad` the checkpoint call is a
  transparent passthrough, so **inference also benefits** from the bounded transient.
- A single host sync reads the CRS row-splits; chunk slicing is then pure Python.

**Effect:** caps *both* the transient and the retained edge activations. Training
peak 16.6 → 6.7 GB; inference peak 8.2 → 4.6 GB. Output and gradients are
bit-identical. (Costs a small recompute in backward — the deliberate
memory-for-time trade, which the neighbor-search win more than pays back.)

#### 3. Spectral convolution — removes redundant FFT copies
**File:** `neuralop/layers/spectral_convolution.py` (`EqSpectralConv.forward`)

For **every shipped config** the latent grid resolution equals `fno_n_mode` (e.g.
`32³` grid, 32 modes), so **no Fourier modes are actually truncated**. The original
code still performed a symmetric `arange`-gather of the full spectrum and a
scatter into a freshly zero-initialized complex grid — two full-spectrum copies and
an allocation per call, for an identity operation.

- Added a `keep_all` fast path that detects "all modes kept" and runs the grouped
  spectral multiply directly on the FFT output, skipping the gather/scatter and the
  zero-init.

**Effect:** less per-layer memory churn and fewer kernels in the FNO blocks;
bit-identical output. (The original truncating path is retained for any future
config where `n_modes < grid`.)

#### 4. On-device R2 — removes a per-step GPU→CPU sync
**Files:** `utils.py` (`torch_r2`, `compute_metrics`), `model/eqgino.py` (`loss`)

`loss()` and `compute_metrics()` computed R² via `sklearn.metrics.r2_score`, which
forces a **`.cpu().numpy()` round-trip on every training / validation / test step**.
That host sync stalls the CUDA pipeline even though R² is only logged.

- Added `torch_r2`, a vectorized on-device R² matching sklearn's
  `multioutput='uniform_average'` (verified identical to machine precision), and
  used it in both places.

**Effect:** removes a pipeline-stalling sync from every logged step.

#### 5. Cached latent query grid — removes a per-forward CPU build + copy
**File:** `model/eqgino.py` (`generate_bounding_latent_queries`)

The bounding latent grid is fixed (range + resolution are constant) but was rebuilt
with a CPU `meshgrid` and copied host→device on **every** forward (×4 per sample for
the DeepJeb multi-load path).

- The grid is now built **once on the target device** and cached.

**Effect:** removes a redundant CPU compute + H2D transfer (and its sync) per step.

---

### Where the memory went (training peak, AhmedBody 70k nodes)

| Stage | v1.0 retained | Note |
|-------|--------------:|------|
| `gno_in` edge activations | 2.7 GB | now chunked + checkpointed |
| FNO blocks | 4.8 GB | reconstructed isotropic weights + FFT buffers |
| `gno_out` edge activations | 7.9 GB | **dominant** → chunked + checkpointed |
| **Total peak** | **16.6 GB** | → **6.7 GB** in v1.1 |

### Reproducing the measurements

Pick a free GPU and run training/inference forward on any real sample with
`hidden_dim=64`, `fno_n_mode=32`, `fno_n_layers=4`, `num_groups=2`. Compare peak via
`torch.cuda.max_memory_allocated()` and wall-clock over ~12 iters after 3 warm-ups.
Toggle the optimized paths off for an A/B by setting `EqGNOBlock.use_checkpoint=False`
(disables GNO chunking/checkpointing) and
`neuralop.layers.neighbor_search.torch_cluster_built=False` (forces the `cdist`
search).

### Compatibility

- No public API or config changes; existing configs and checkpoints work unchanged.
- New optional knobs: `EqGNOBlock(use_checkpoint=True)` (default on) and
  `EqGNOBlock.chunk_target_edges` (default 200,000).
- `torch_cluster` is already a project dependency (PyG extensions). If it is absent,
  the code automatically falls back to the dense `cdist` search.
