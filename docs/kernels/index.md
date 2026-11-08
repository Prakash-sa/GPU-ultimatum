# Classic custom kernels

Ten kernels that people write by hand again and again, because a library call is either
missing or slower once it is fused into the code around it. Each page has one picture,
the few lines that matter, and a link to a complete, tested CUDA C++ file in
[`docs/kernels/src/`](https://github.com/Prakash-sa/GPU-ultimatum/tree/main/docs/kernels/src).

| # | Kernel | Used in | The technique |
|---|---|---|---|
| [1](01-activations.md) | Fused bias + GELU, SwiGLU | every transformer MLP | elementwise fusion, `float4` |
| [2](02-rope.md) | Rotary position embedding | LLaMA, GPT-NeoX, most LLMs | pair rotation, angles computed on the fly |
| [3](03-cross-entropy.md) | Fused cross-entropy | LM training loss | online log-sum-exp in one pass |
| [4](04-topk.md) | Top-k per row | sampling, beam search, MoE routing | k rounds of block argmax in shared memory |
| [5](05-bitonic-sort.md) | Bitonic sort | small sorts, sort inside a block | sorting network: shared memory, then global stages |
| [6](06-radix-sort.md) | Radix sort (key, value) | argsort, binning particles, MoE dispatch | count → scan → scatter, `__match_any_sync` ranks |
| [7](07-dropout.md) | Dropout with Philox | training | counter-based RNG: no state, reproducible |
| [8](08-int8-quant.md) | Int8 quantize / dequantize | LLM inference, compression | per-row absmax, exact rounding |
| [9](09-embedding-bag.md) | Embedding bag | recommender models | gather, then reduce per bag (CSR offsets) |
| [10](10-maxpool2d.md) | 2-D max pooling + argmax | CNNs | windows, indices for the backward pass |

These sit next to the ones already on the site: transpose, reduction, softmax, the
norms, GEMM, scan, attention and SpMV are on the [LeetGPU](../leetgpu/index.md) pages;
histogram and convolution are in the [CUDA crash course](../cuda-course/index.md);
stencils and warp primitives are in the [notes](../notes/index.md).

## How they were checked

Every kernel exposes an `extern "C"` function on device pointers and is tested against
PyTorch by `tests/cuda/run_tests.py`:

```bash
python tests/cuda/run_tests.py docs/kernels          # CPU emulation, no GPU needed
python tests/cuda/run_tests.py --gpu docs/kernels    # on a GPU
```

- **Correctness:** the CPU emulator passes all checks against PyTorch, including bit-exact
  ones: sort order, top-k indices, int8 codes, pooling argmax, and the dropout mask
  against a pure-Python Philox.
- **The RNG is the real Philox:** its output matches the published Random123
  known-answer test vectors.
- **The tests catch races:** removing a `__syncthreads()` from radix sort, top-k or
  bitonic sort makes their tests fail.
- **Compiler:** every file builds with nvcc 12.6 for sm_75, sm_80 and sm_90, with
  warnings as errors and no register spills.
- **Not yet:** none of it has run on a GPU, so no timings are given.
