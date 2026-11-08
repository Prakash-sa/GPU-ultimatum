// CUDA crash course, module 5: convolution, five steps.
//   1. 1-D naive             y[i] = sum_j x[i - r + j] * mask[j], mask in global memory
//   2. 1-D constant memory   the mask in __constant__ memory: one broadcast read per warp
//   3. 1-D tiled             the block's inputs + halo staged in shared memory
//   4. 1-D cache             only the block's own inputs in shared memory; the halo comes
//                            from global memory, where L1/L2 already hold it
//   5. 2-D constant memory   an image and an m x m mask in constant memory
// Zero padding outside the input. Mask length m is odd, m <= 31 (r = m / 2).
#include <cuda_runtime.h>

constexpr int kThreads = 256;
constexpr int kMaxMask = 31;

// --8<-- [start:naive]
__global__ void conv1d_naive(const float* x, const float* mask, float* y, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = m / 2;
    float acc = 0.0f;
    for (int j = 0; j < m; ++j) {
        int k = i - r + j;
        if (k >= 0 && k < n) acc += x[k] * mask[j];      // the mask is re-read by every thread
    }
    y[i] = acc;
}
// --8<-- [end:naive]

// --8<-- [start:constant]
__constant__ float c_mask[kMaxMask * kMaxMask];          // 64 KB of constant memory per GPU

__global__ void conv1d_constant(const float* x, float* y, int n, int m) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    int r = m / 2;
    float acc = 0.0f;
    for (int j = 0; j < m; ++j) {
        int k = i - r + j;
        if (k >= 0 && k < n) acc += x[k] * c_mask[j];    // all lanes read c_mask[j]: one broadcast
    }
    y[i] = acc;
}
// --8<-- [end:constant]

// --8<-- [start:tiled]
__global__ void conv1d_tiled(const float* x, float* y, int n, int m) {
    __shared__ float tile[kThreads + kMaxMask - 1];      // the block's inputs plus 2r halo
    int r = m / 2;
    int base = blockIdx.x * blockDim.x - r;              // global index of tile[0]
    for (int t = threadIdx.x; t < blockDim.x + 2 * r; t += blockDim.x) {
        int k = base + t;
        tile[t] = (k >= 0 && k < n) ? x[k] : 0.0f;       // zero padding at the edges
    }
    __syncthreads();
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float acc = 0.0f;
    for (int j = 0; j < m; ++j) acc += tile[threadIdx.x + j] * c_mask[j];   // all from shared
    y[i] = acc;
}
// --8<-- [end:tiled]

// --8<-- [start:cache]
__global__ void conv1d_cache(const float* x, float* y, int n, int m) {
    __shared__ float tile[kThreads];                     // only the block's own inputs
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    tile[threadIdx.x] = i < n ? x[i] : 0.0f;
    __syncthreads();
    if (i >= n) return;
    int r = m / 2, start = blockIdx.x * blockDim.x;
    float acc = 0.0f;
    for (int j = 0; j < m; ++j) {
        int k = i - r + j;
        if (k < 0 || k >= n) continue;
        int t = k - start;                               // inside the tile? shared memory;
        acc += (t >= 0 && t < int(blockDim.x) ? tile[t] : x[k]) * c_mask[j];   // halo: L1/L2
    }
    y[i] = acc;
}
// --8<-- [end:cache]

// --8<-- [start:conv2d]
__global__ void conv2d_constant(const float* img, float* out, int N, int m) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;
    int r = m / 2;
    float acc = 0.0f;
    for (int u = 0; u < m; ++u)
        for (int v = 0; v < m; ++v) {
            int rr = row - r + u, cc = col - r + v;
            if (rr >= 0 && rr < N && cc >= 0 && cc < N) acc += img[rr * N + cc] * c_mask[u * m + v];
        }
    out[row * N + col] = acc;
}
// --8<-- [end:conv2d]

// ---------------------------------------------------------------- host side

static void conv1d(const float* h_x, int n, const float* h_mask, int m, float* h_y, int step) {
    size_t bytes = size_t(n) * sizeof(float);
    float *d_x, *d_y, *d_mask;
    cudaMalloc(&d_x, bytes);
    cudaMalloc(&d_y, bytes);
    cudaMalloc(&d_mask, size_t(m) * sizeof(float));
    cudaMemcpy(d_x, h_x, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mask, h_mask, size_t(m) * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, h_mask, size_t(m) * sizeof(float));   // for steps 2-4
    int blocks = (n + kThreads - 1) / kThreads;
    if (step == 1) conv1d_naive<<<blocks, kThreads>>>(d_x, d_mask, d_y, n, m);
    if (step == 2) conv1d_constant<<<blocks, kThreads>>>(d_x, d_y, n, m);
    if (step == 3) conv1d_tiled<<<blocks, kThreads>>>(d_x, d_y, n, m);
    if (step == 4) conv1d_cache<<<blocks, kThreads>>>(d_x, d_y, n, m);
    cudaMemcpy(h_y, d_y, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x);
    cudaFree(d_y);
    cudaFree(d_mask);
}

extern "C" void conv1d_step(const float* x, int n, const float* mask, int m, float* y, int step) {
    conv1d(x, n, mask, m, y, step);
}

extern "C" void conv2d(const float* h_img, int N, const float* h_mask, int m, float* h_out) {
    size_t bytes = size_t(N) * N * sizeof(float);
    float *d_img, *d_out;
    cudaMalloc(&d_img, bytes);
    cudaMalloc(&d_out, bytes);
    cudaMemcpy(d_img, h_img, bytes, cudaMemcpyHostToDevice);
    cudaMemcpyToSymbol(c_mask, h_mask, size_t(m) * m * sizeof(float));
    dim3 block(16, 16), grid((N + 15) / 16, (N + 15) / 16);
    conv2d_constant<<<grid, block>>>(d_img, d_out, N, m);
    cudaMemcpy(h_out, d_out, bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_img);
    cudaFree(d_out);
}
