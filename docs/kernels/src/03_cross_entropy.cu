// Classic kernel 3: fused cross-entropy loss.
//   loss[r] = logsumexp(logits[r, :]) - logits[r, target[r]]        logits: N x V
// Unfused: softmax writes N x V probabilities, log + gather read them back. Fused: one
// pass over each row with an *online* logsumexp (running max m and running sum s,
// rescaled when the max grows), so the N x V intermediate never exists.
#include <cuda_runtime.h>
#include <math.h>

constexpr int kThreads = 256;

// Merge two (max, sum-of-exp relative to max) pairs.
__device__ __forceinline__ void merge(float& m, float& s, float m2, float s2) {
    float mm = fmaxf(m, m2);
    if (mm == -INFINITY) return;                         // both empty
    s = s * expf(m - mm) + s2 * expf(m2 - mm);
    m = mm;
}

// --8<-- [start:ce]
__global__ void cross_entropy_kernel(const float* __restrict__ logits, const int* __restrict__ target,
                                     float* __restrict__ loss, int V) {
    __shared__ float sm[kThreads / 32], ss[kThreads / 32];
    const float* row = logits + (long long)blockIdx.x * V;
    float m = -INFINITY, s = 0.0f;
    for (int j = threadIdx.x; j < V; j += blockDim.x) {  // online: one read of each logit
        float v = row[j];
        if (v > m) { s = s * expf(m - v) + 1.0f; m = v; }
        else s += expf(v - m);
    }
    for (int o = 16; o > 0; o >>= 1)                     // merge (m, s) across the warp
        merge(m, s, __shfl_xor_sync(0xffffffffu, m, o), __shfl_xor_sync(0xffffffffu, s, o));
    if ((threadIdx.x & 31) == 0) { sm[threadIdx.x >> 5] = m; ss[threadIdx.x >> 5] = s; }
    __syncthreads();
    if (threadIdx.x == 0) {
        float M = sm[0], S = ss[0];
        for (int w = 1; w < kThreads / 32; ++w) merge(M, S, sm[w], ss[w]);
        loss[blockIdx.x] = M + logf(S) - row[target[blockIdx.x]];
    }
}
// --8<-- [end:ce]

// logits, target, loss are device pointers. One block per row.
extern "C" void cross_entropy(const float* logits, const int* target, float* loss, int N, int V) {
    cross_entropy_kernel<<<N, kThreads>>>(logits, target, loss, V);
    cudaDeviceSynchronize();
}
