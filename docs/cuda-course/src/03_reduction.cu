// CUDA crash course, module 3: sum reduction, six steps.
// Every step is a block-level sum into shared memory; the host relaunches on the
// partial sums until one value is left. Integers keep the answer exact.
//   1. diverged        interleaved addressing: if (tid % (2*s) == 0) → divergent warps
//   2. bank conflicts  strided index idx = 2*s*tid: no divergence, but bank conflicts
//   3. no conflicts    sequential addressing: tid < s adds tid + s
//   4. reduce idle     each thread adds two elements while loading: half the blocks
//   5. warp shuffles   the last 32 values with __shfl_down_sync (a device function)
//   6. coop. groups    the same with cooperative_groups tiles and cg::reduce
#include <cuda_runtime.h>
#include <cooperative_groups.h>
#include <cooperative_groups/reduce.h>
#include <vector>

namespace cg = cooperative_groups;
constexpr int B = 256;                                   // threads per block (power of two)

// --8<-- [start:diverged]
__global__ void reduce_diverged(const int* in, int* out, int n) {
    __shared__ int s[B];
    int tid = threadIdx.x, i = blockIdx.x * B + tid;
    s[tid] = i < n ? in[i] : 0;
    __syncthreads();
    for (int stride = 1; stride < B; stride *= 2) {
        if (tid % (2 * stride) == 0) s[tid] += s[tid + stride];   // threads 0, 2, 4, …: every
        __syncthreads();                                          // warp diverges
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}
// --8<-- [end:diverged]

// --8<-- [start:bank_conflicts]
__global__ void reduce_bank_conflicts(const int* in, int* out, int n) {
    __shared__ int s[B];
    int tid = threadIdx.x, i = blockIdx.x * B + tid;
    s[tid] = i < n ? in[i] : 0;
    __syncthreads();
    for (int stride = 1; stride < B; stride *= 2) {
        int idx = 2 * stride * tid;                      // contiguous active threads: no
        if (idx < B) s[idx] += s[idx + stride];          // divergence, but idx jumps by 2*stride:
        __syncthreads();                                 // many threads hit the same bank
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}
// --8<-- [end:bank_conflicts]

// --8<-- [start:no_conflicts]
__global__ void reduce_no_conflicts(const int* in, int* out, int n) {
    __shared__ int s[B];
    int tid = threadIdx.x, i = blockIdx.x * B + tid;
    s[tid] = i < n ? in[i] : 0;
    __syncthreads();
    for (int stride = B / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];     // consecutive threads, consecutive
        __syncthreads();                                 // addresses: one bank each
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}
// --8<-- [end:no_conflicts]

// --8<-- [start:reduce_idle]
__global__ void reduce_idle(const int* in, int* out, int n) {
    __shared__ int s[B];
    int tid = threadIdx.x, i = blockIdx.x * (2 * B) + tid;      // each block covers 2*B inputs
    s[tid] = (i < n ? in[i] : 0) + (i + B < n ? in[i + B] : 0); // first add during the load
    __syncthreads();
    for (int stride = B / 2; stride > 0; stride >>= 1) {
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid == 0) out[blockIdx.x] = s[0];
}
// --8<-- [end:reduce_idle]

// --8<-- [start:warp]
__device__ int warp_sum(int v) {                         // 32 lanes, registers only
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(0xffffffffu, v, o);
    return v;                                            // lane 0 holds the warp's sum
}

__global__ void reduce_warp(const int* in, int* out, int n) {
    __shared__ int s[B];
    int tid = threadIdx.x, i = blockIdx.x * (2 * B) + tid;
    s[tid] = (i < n ? in[i] : 0) + (i + B < n ? in[i + B] : 0);
    __syncthreads();
    for (int stride = B / 2; stride > 32; stride >>= 1) {   // shared-memory steps down to 64
        if (tid < stride) s[tid] += s[tid + stride];
        __syncthreads();
    }
    if (tid < 32) {                                      // the last warp finishes in registers
        int v = warp_sum(s[tid] + s[tid + 32]);
        if (tid == 0) out[blockIdx.x] = v;
    }
}
// --8<-- [end:warp]

// --8<-- [start:cg]
__global__ void reduce_cg(const int* in, int* out, int n) {
    __shared__ int warp_totals[B / 32];
    cg::thread_block block = cg::this_thread_block();
    auto warp = cg::tiled_partition<32>(block);
    int i = blockIdx.x * (2 * B) + threadIdx.x;
    int v = (i < n ? in[i] : 0) + (i + B < n ? in[i + B] : 0);
    v = cg::reduce(warp, v, cg::plus<int>());            // every lane gets its warp's sum
    if (warp.thread_rank() == 0) warp_totals[warp.meta_group_rank()] = v;
    block.sync();
    if (warp.meta_group_rank() == 0) {
        v = cg::reduce(warp, warp.thread_rank() < B / 32 ? warp_totals[warp.thread_rank()] : 0,
                       cg::plus<int>());
        if (warp.thread_rank() == 0) out[blockIdx.x] = v;
    }
}
// --8<-- [end:cg]

// ---------------------------------------------------------------- host side

using Kernel = void (*)(const int*, int*, int);

// Launch the block reduction until one value is left; `per_block` is how many inputs
// one block consumes (B, or 2*B for the steps that add while loading).
static int run(const int* h_in, int n, Kernel k, int per_block) {
    int *d_a, *d_b;
    cudaMalloc(&d_a, size_t(n > 0 ? n : 1) * sizeof(int));
    cudaMalloc(&d_b, size_t((n + per_block - 1) / per_block + 1) * sizeof(int));
    cudaMemcpy(d_a, h_in, size_t(n) * sizeof(int), cudaMemcpyHostToDevice);
    int count = n;
    if (count == 0) {
        cudaMemset(d_a, 0, sizeof(int));
        count = 1;
    }
    while (count > 1) {
        int blocks = (count + per_block - 1) / per_block;
        k<<<blocks, B>>>(d_a, d_b, count);
        int* t = d_a; d_a = d_b; d_b = t;                // partials become the next input
        count = blocks;
    }
    int result = 0;
    cudaMemcpy(&result, d_a, sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    return result;
}

extern "C" int sum_diverged(const int* x, int n) { return run(x, n, reduce_diverged, B); }
extern "C" int sum_bank_conflicts(const int* x, int n) { return run(x, n, reduce_bank_conflicts, B); }
extern "C" int sum_no_conflicts(const int* x, int n) { return run(x, n, reduce_no_conflicts, B); }
extern "C" int sum_reduce_idle(const int* x, int n) { return run(x, n, reduce_idle, 2 * B); }
extern "C" int sum_warp(const int* x, int n) { return run(x, n, reduce_warp, 2 * B); }
extern "C" int sum_cg(const int* x, int n) { return run(x, n, reduce_cg, 2 * B); }
