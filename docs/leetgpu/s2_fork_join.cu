// S2. Fork-join with events (practice problem in LeetGPU's format; not on LeetGPU).
// Page: s2-fork-join.md
//
// Given x of length n on the device, write out[i] = (x[i] - mean(x)) / max|x|, or 0 when
// max|x| is 0. The two statistics don't depend on each other, so compute them on two
// streams at once, then join both before the normalize kernel.
#include <cuda_runtime.h>

#define FULL 0xffffffffu

__device__ float warp_sum(float v) {
    for (int o = 16; o > 0; o >>= 1) v += __shfl_down_sync(FULL, v, o);
    return v;
}
__device__ float warp_max(float v) {
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_down_sync(FULL, v, o));
    return v;
}

__global__ void sum_kernel(const float* x, int n, float* sum) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x) v += x[i];
    v = warp_sum(v);
    if ((threadIdx.x & 31) == 0) atomicAdd(sum, v);
}

// |x| >= 0, and non-negative floats order the same way as their bit patterns.
__global__ void absmax_kernel(const float* x, int n, unsigned* amax) {
    float v = 0.0f;
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        v = fmaxf(v, fabsf(x[i]));
    v = warp_max(v);
    if ((threadIdx.x & 31) == 0) atomicMax(amax, __float_as_uint(v));
}

__global__ void normalize(const float* x, float* out, int n, const float* sum, const unsigned* amax) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float m = __uint_as_float(*amax);
    if (i < n) out[i] = m > 0.0f ? (x[i] - *sum / n) / m : 0.0f;
}

extern "C" void solve(const float* x, float* out, int n) {
    const int T = 256, G = (n + T - 1) / T < 128 ? (n + T - 1) / T : 128;
    cudaStream_t a, b;
    cudaStreamCreateWithFlags(&a, cudaStreamNonBlocking);    // non-blocking: no implicit
    cudaStreamCreateWithFlags(&b, cudaStreamNonBlocking);    // sync with stream 0
    cudaEvent_t ready, sum_done, max_done;
    cudaEventCreateWithFlags(&ready, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&sum_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&max_done, cudaEventDisableTiming);
    float* sum;
    unsigned* amax;
    cudaMalloc(&sum, sizeof(float));
    cudaMalloc(&amax, sizeof(unsigned));

    cudaEventRecord(ready, 0);                    // fork: x was written on stream 0
    cudaStreamWaitEvent(a, ready, 0);
    cudaStreamWaitEvent(b, ready, 0);

    cudaMemsetAsync(sum, 0, sizeof(float), a);
    sum_kernel<<<G, T, 0, a>>>(x, n, sum);
    cudaEventRecord(sum_done, a);

    cudaMemsetAsync(amax, 0, sizeof(unsigned), b);
    absmax_kernel<<<G, T, 0, b>>>(x, n, amax);
    cudaEventRecord(max_done, b);

    cudaStreamWaitEvent(0, sum_done, 0);          // join: stream 0 waits for both branches,
    cudaStreamWaitEvent(0, max_done, 0);          // the host does not
    normalize<<<(n + T - 1) / T, T>>>(x, out, n, sum, amax);

    cudaFree(sum);                                // cudaFree waits for the device
    cudaFree(amax);
    cudaEventDestroy(ready);
    cudaEventDestroy(sum_done);
    cudaEventDestroy(max_done);
    cudaStreamDestroy(a);
    cudaStreamDestroy(b);
}
