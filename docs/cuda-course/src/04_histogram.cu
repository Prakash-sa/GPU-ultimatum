// CUDA crash course, module 4: histogram, two steps.
//   1. global atomics   every element does atomicAdd on the global histogram
//   2. shared atomics   each block builds a private histogram in shared memory, then adds
//                       it to the global one: nbins global atomics per block, not n
// Values are ints in [0, nbins), nbins <= 256.
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxBins = 256;

// --8<-- [start:global]
__global__ void hist_global(const int* __restrict__ x, int n, int* __restrict__ hist) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&hist[x[i]], 1);                       // contended: few bins, many threads
}
// --8<-- [end:global]

// --8<-- [start:shared]
__global__ void hist_shared(const int* __restrict__ x, int n, int* __restrict__ hist, int nbins) {
    __shared__ int local[kMaxBins];
    for (int b = threadIdx.x; b < nbins; b += blockDim.x) local[b] = 0;
    __syncthreads();                                     // the private histogram is zeroed
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        atomicAdd(&local[x[i]], 1);                      // shared-memory atomics: on the SM
    __syncthreads();                                     // the block's counts are complete
    for (int b = threadIdx.x; b < nbins; b += blockDim.x)
        if (local[b]) atomicAdd(&hist[b], local[b]);     // one global atomic per bin per block
}
// --8<-- [end:shared]

static void round_trip(const int* h_x, int n, int* h_hist, int nbins, bool shared) {
    int *d_x, *d_hist;
    cudaMalloc(&d_x, size_t(n > 0 ? n : 1) * sizeof(int));
    cudaMalloc(&d_hist, size_t(nbins) * sizeof(int));
    cudaMemcpy(d_x, h_x, size_t(n) * sizeof(int), cudaMemcpyHostToDevice);
    cudaMemset(d_hist, 0, size_t(nbins) * sizeof(int));
    int sms = 0, dev = 0;
    cudaGetDevice(&dev);
    cudaDeviceGetAttribute(&sms, cudaDevAttrMultiProcessorCount, dev);
    int blocks = (n + kThreads - 1) / kThreads;
    if (blocks > 8 * sms) blocks = 8 * sms;              // a fixed grid; grid-stride loops cover n
    if (blocks < 1) blocks = 1;
    if (shared) hist_shared<<<blocks, kThreads>>>(d_x, n, d_hist, nbins);
    else hist_global<<<blocks, kThreads>>>(d_x, n, d_hist);
    cudaMemcpy(h_hist, d_hist, size_t(nbins) * sizeof(int), cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_hist);
}

extern "C" void histogram_global(const int* x, int n, int* hist, int nbins) { round_trip(x, n, hist, nbins, false); }
extern "C" void histogram_shared(const int* x, int n, int* hist, int nbins) { round_trip(x, n, hist, nbins, true); }
