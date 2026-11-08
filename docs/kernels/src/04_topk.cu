// Classic kernel 4: top-k per row (values and indices, descending).
// For small k (sampling, beam search, MoE routing: k = 1..64) and rows that fit in shared
// memory (V <= 8192): load the row once, then k rounds of a block-wide argmax; the
// winner is recorded and knocked out with -inf. Cost: one DRAM read of the row + k
// passes over shared memory. Ties go to the lower index (deterministic).
#include <cuda_runtime.h>
#include <math.h>

constexpr int kThreads = 256;
constexpr int kMaxV = 8192;

struct VI { float v; int i; };

__device__ __forceinline__ VI better(VI a, VI b) {     // larger value wins; ties: lower index
    return (b.v > a.v || (b.v == a.v && b.i < a.i)) ? b : a;
}

// --8<-- [start:topk]
__global__ void topk_kernel(const float* __restrict__ x, float* __restrict__ vals, int* __restrict__ idx,
                            int V, int k) {
    __shared__ float row[kMaxV];
    __shared__ VI warp_best[kThreads / 32];
    const float* src = x + (long long)blockIdx.x * V;
    for (int j = threadIdx.x; j < V; j += blockDim.x) row[j] = src[j];   // one DRAM read
    __syncthreads();
    for (int r = 0; r < k; ++r) {
        VI best{-INFINITY, 0x7fffffff};
        for (int j = threadIdx.x; j < V; j += blockDim.x) best = better(best, VI{row[j], j});
        for (int o = 16; o > 0; o >>= 1)                                 // warp argmax
            best = better(best, VI{__shfl_xor_sync(0xffffffffu, best.v, o),
                                   __shfl_xor_sync(0xffffffffu, best.i, o)});
        if ((threadIdx.x & 31) == 0) warp_best[threadIdx.x >> 5] = best;
        __syncthreads();
        if (threadIdx.x == 0) {
            VI b = warp_best[0];
            for (int w = 1; w < kThreads / 32; ++w) b = better(b, warp_best[w]);
            vals[(long long)blockIdx.x * k + r] = b.v;
            idx[(long long)blockIdx.x * k + r] = b.i;
            row[b.i] = -INFINITY;                                        // knock it out
        }
        __syncthreads();                                                 // before the next round
    }
}
// --8<-- [end:topk]

// x, vals, idx are device pointers. x: N x V; vals, idx: N x k.
extern "C" void topk(const float* x, float* vals, int* idx, int N, int V, int k) {
    topk_kernel<<<N, kThreads>>>(x, vals, idx, V, k);
    cudaDeviceSynchronize();
}
