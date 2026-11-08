# CUDA crash course

Five classic kernels, each improved one step at a time. Every page leads with a picture
of what changes and shows only the lines that change. The complete programs are in
[`docs/cuda-course/src/`](https://github.com/Prakash-sa/GPU-ultimatum/tree/main/docs/cuda-course/src)
and are tested.

The modules and steps follow **[CUDA Crash Course (v3)](https://github.com/CoffeeBeforeArch/cuda_programming)**
by CoffeeBeforeArch (a YouTube series; its code is GPL-3.0). The implementations,
figures and explanations here are original. The series narrates each step; watch it
alongside.

| Module | Steps | The lesson |
|---|---|---|
| [1. Vector addition](01-vector-add.md) | pageable → pinned → unified → unified + prefetch | the kernel is trivial; **moving the data** is the problem |
| [2. Matrix multiplication](02-matmul.md) | baseline → no aliasing → any size → rectangular → tiled | reuse: shared-memory tiles turn 1 FMA per load into T |
| [3. Sum reduction](03-reduction.md) | diverged → bank conflicts → sequential → add on load → warp shuffles → cooperative groups | divergence, bank conflicts and idle threads, removed one at a time |
| [4. Histogram](04-histogram.md) | global atomics → shared-memory atomics | contention: aggregate locally, then merge |
| [5. Convolution](05-convolution.md) | naive → constant mask → tiled → cache → 2-D | where each input comes from: constant, shared, L1 |

## How the code is organized

Each module is one `.cu` file. Every step is a kernel plus an `extern "C"` function
that does the whole round trip: allocate, copy in, launch, copy out, free. So every step
can be tested on its own:

```text
docs/cuda-course/src/
├── 01_vector_add.cu    vadd_baseline · vadd_pinned · vadd_unified · vadd_unified_prefetch
├── 02_matmul.cu        mm_baseline · mm_no_alias · mm_non_multiple · mm_rectangular · mm_tiled
├── 03_reduction.cu     sum_diverged · sum_bank_conflicts · sum_no_conflicts · sum_reduce_idle · sum_warp · sum_cg
├── 04_histogram.cu     histogram_global · histogram_shared
└── 05_convolution.cu   conv1d_step(…, step = 1..4) · conv2d
```

```bash
python tests/cuda/run_tests.py cuda-course          # CPU emulation, no GPU needed
python tests/cuda/run_tests.py --gpu cuda-course    # on a machine with nvcc and a GPU
```

**What has been verified:**

- **Emulator:** all 21 steps pass 108 checks against PyTorch in the repo's CPU emulator,
  at awkward sizes (1, 255, 257, 70,001, …).
- **The tests catch races:** removing a `__syncthreads()` from the reduction, the tiled
  matmul or the tiled convolution makes their tests fail.
- **Compiler:** all five files build with nvcc 12.6 for sm_75, sm_80 and sm_90, with
  warnings as errors and no register spills.
- **Not yet:** none of it has run on a GPU, so these pages give no timings. Each page
  says what to measure instead.
