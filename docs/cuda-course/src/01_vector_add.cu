// CUDA crash course, module 1: vector addition, four ways to move the data.
// Same kernel every time; what changes is how the bytes get to the GPU:
//   1. baseline         pageable host memory + cudaMemcpy
//   2. pinned           page-locked host memory: the copy engine reads it directly (DMA)
//   3. unified          cudaMallocManaged: one pointer, pages migrate on first touch
//   4. unified+prefetch cudaMemPrefetchAsync moves the pages in bulk before the kernel
// Each entry point takes ordinary host arrays and does the whole round trip.
#include <cuda_runtime.h>
#include <cstring>

constexpr int kThreads = 256;

// --8<-- [start:kernel]
__global__ void vector_add(const float* __restrict__ a, const float* __restrict__ b,
                           float* __restrict__ c, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;      // one thread per element
    if (i < n) c[i] = a[i] + b[i];                       // the last block may overhang n
}
// --8<-- [end:kernel]

static int blocks_for(int n) { return n > 0 ? (n + kThreads - 1) / kThreads : 1; }

// --8<-- [start:baseline]
extern "C" void vadd_baseline(const float* h_a, const float* h_b, float* h_c, int n) {
    size_t bytes = size_t(n) * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice);   // pageable: staged through a
    cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice);   // hidden pinned buffer
    vector_add<<<blocks_for(n), kThreads>>>(d_a, d_b, d_c, n);
    cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost);   // waits for the kernel
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}
// --8<-- [end:baseline]

// --8<-- [start:pinned]
extern "C" void vadd_pinned(const float* h_a, const float* h_b, float* h_c, int n) {
    size_t bytes = size_t(n) * sizeof(float);
    float *p_a, *p_b, *p_c;                              // page-locked host buffers. In a real
    cudaMallocHost(&p_a, bytes);                         // program the data is created in them;
    cudaMallocHost(&p_b, bytes);                         // here we fill them from the caller's
    cudaMallocHost(&p_c, bytes);                         // arrays
    std::memcpy(p_a, h_a, bytes);
    std::memcpy(p_b, h_b, bytes);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, bytes);
    cudaMalloc(&d_b, bytes);
    cudaMalloc(&d_c, bytes);
    cudaMemcpy(d_a, p_a, bytes, cudaMemcpyHostToDevice); // DMA straight from p_a: no staging
    cudaMemcpy(d_b, p_b, bytes, cudaMemcpyHostToDevice);
    vector_add<<<blocks_for(n), kThreads>>>(d_a, d_b, d_c, n);
    cudaMemcpy(p_c, d_c, bytes, cudaMemcpyDeviceToHost);
    std::memcpy(h_c, p_c, bytes);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFreeHost(p_a);
    cudaFreeHost(p_b);
    cudaFreeHost(p_c);
}
// --8<-- [end:pinned]

// --8<-- [start:unified]
extern "C" void vadd_unified(const float* h_a, const float* h_b, float* h_c, int n) {
    size_t bytes = size_t(n) * sizeof(float);
    float *a, *b, *c;                                    // one pointer, valid on CPU and GPU
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    std::memcpy(a, h_a, bytes);                          // CPU touches: pages live on the CPU
    std::memcpy(b, h_b, bytes);
    vector_add<<<blocks_for(n), kThreads>>>(a, b, c, n); // GPU touches: page faults migrate them
    cudaDeviceSynchronize();                             // required before the CPU reads c
    std::memcpy(h_c, c, bytes);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
// --8<-- [end:unified]

// --8<-- [start:prefetch]
extern "C" void vadd_unified_prefetch(const float* h_a, const float* h_b, float* h_c, int n) {
    size_t bytes = size_t(n) * sizeof(float);
    int dev = 0;
    cudaGetDevice(&dev);
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);
    std::memcpy(a, h_a, bytes);
    std::memcpy(b, h_b, bytes);
    cudaMemPrefetchAsync(a, bytes, dev);                 // bulk-migrate before the kernel:
    cudaMemPrefetchAsync(b, bytes, dev);                 // no page faults inside it
    cudaMemPrefetchAsync(c, bytes, dev);
    vector_add<<<blocks_for(n), kThreads>>>(a, b, c, n);
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);     // and bring the result back in bulk
    cudaDeviceSynchronize();
    std::memcpy(h_c, c, bytes);
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);
}
// --8<-- [end:prefetch]
