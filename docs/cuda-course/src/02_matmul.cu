// CUDA crash course, module 2: matrix multiplication, five steps.
//   1. baseline       square N x N, N a multiple of the block size, one thread per C element
//   2. no aliasing    __restrict__ + a register accumulator instead of += into C
//   3. non-multiple   any N: bounds checks for the threads that overhang the matrix
//   4. rectangular    A is M x K, B is K x N, C is M x N
//   5. tiled          16 x 16 tiles of A and B staged in shared memory
// Row-major throughout. Each entry point takes host arrays and does the round trip.
#include <cuda_runtime.h>

constexpr int T = 16;                                    // block is T x T threads

// --8<-- [start:baseline]
__global__ void mm_baseline_kernel(const float* a, const float* b, float* c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    c[row * N + col] = 0.0f;
    for (int k = 0; k < N; ++k)
        c[row * N + col] += a[row * N + k] * b[k * N + col];   // C is read and written every k
}
// --8<-- [end:baseline]

// --8<-- [start:no_alias]
__global__ void mm_no_alias_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                   float* __restrict__ c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    float acc = 0.0f;                                    // a register, not memory
    for (int k = 0; k < N; ++k) acc += a[row * N + k] * b[k * N + col];
    c[row * N + col] = acc;                              // one store at the end
}
// --8<-- [end:no_alias]

// --8<-- [start:non_multiple]
__global__ void mm_non_multiple_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                       float* __restrict__ c, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N || col >= N) return;                    // threads past the edge do nothing
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) acc += a[row * N + k] * b[k * N + col];
    c[row * N + col] = acc;
}
// --8<-- [end:non_multiple]

// --8<-- [start:rectangular]
__global__ void mm_rect_kernel(const float* __restrict__ a, const float* __restrict__ b,
                               float* __restrict__ c, int M, int K, int N) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;     // 0 .. M-1
    int col = blockIdx.x * blockDim.x + threadIdx.x;     // 0 .. N-1
    if (row >= M || col >= N) return;
    float acc = 0.0f;
    for (int k = 0; k < K; ++k) acc += a[row * K + k] * b[k * N + col];
    c[row * N + col] = acc;
}
// --8<-- [end:rectangular]

// --8<-- [start:tiled]
__global__ void mm_tiled_kernel(const float* __restrict__ a, const float* __restrict__ b,
                                float* __restrict__ c, int M, int K, int N) {
    __shared__ float As[T][T], Bs[T][T];                 // one tile of A and one of B
    int ty = threadIdx.y, tx = threadIdx.x;
    int row = blockIdx.y * T + ty, col = blockIdx.x * T + tx;
    float acc = 0.0f;
    for (int k0 = 0; k0 < K; k0 += T) {
        As[ty][tx] = (row < M && k0 + tx < K) ? a[row * K + k0 + tx] : 0.0f;   // each thread
        Bs[ty][tx] = (k0 + ty < K && col < N) ? b[(k0 + ty) * N + col] : 0.0f; // loads one value
        __syncthreads();                                 // the tiles are complete
        for (int k = 0; k < T; ++k) acc += As[ty][k] * Bs[k][tx];   // T FMAs from shared memory
        __syncthreads();                                 // done before the next load overwrites
    }
    if (row < M && col < N) c[row * N + col] = acc;      // no early return: all threads load
}
// --8<-- [end:tiled]

// ---------------------------------------------------------------- host side

template <typename Launch>
static void round_trip(const float* h_a, const float* h_b, float* h_c, int M, int K, int N, Launch launch) {
    size_t ba = size_t(M) * K * sizeof(float), bb = size_t(K) * N * sizeof(float), bc = size_t(M) * N * sizeof(float);
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, ba);
    cudaMalloc(&d_b, bb);
    cudaMalloc(&d_c, bc);
    cudaMemcpy(d_a, h_a, ba, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, bb, cudaMemcpyHostToDevice);
    launch(d_a, d_b, d_c);
    cudaMemcpy(h_c, d_c, bc, cudaMemcpyDeviceToHost);
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
}

static dim3 grid_for(int rows, int cols) { return dim3((cols + T - 1) / T, (rows + T - 1) / T); }

// N must be a multiple of 16 for steps 1 and 2.
extern "C" void mm_baseline(const float* a, const float* b, float* c, int N) {
    round_trip(a, b, c, N, N, N, [&](const float* x, const float* y, float* z) {
        mm_baseline_kernel<<<grid_for(N, N), dim3(T, T)>>>(x, y, z, N);
    });
}
extern "C" void mm_no_alias(const float* a, const float* b, float* c, int N) {
    round_trip(a, b, c, N, N, N, [&](const float* x, const float* y, float* z) {
        mm_no_alias_kernel<<<grid_for(N, N), dim3(T, T)>>>(x, y, z, N);
    });
}
extern "C" void mm_non_multiple(const float* a, const float* b, float* c, int N) {
    round_trip(a, b, c, N, N, N, [&](const float* x, const float* y, float* z) {
        mm_non_multiple_kernel<<<grid_for(N, N), dim3(T, T)>>>(x, y, z, N);
    });
}
extern "C" void mm_rectangular(const float* a, const float* b, float* c, int M, int K, int N) {
    round_trip(a, b, c, M, K, N, [&](const float* x, const float* y, float* z) {
        mm_rect_kernel<<<grid_for(M, N), dim3(T, T)>>>(x, y, z, M, K, N);
    });
}
extern "C" void mm_tiled(const float* a, const float* b, float* c, int M, int K, int N) {
    round_trip(a, b, c, M, K, N, [&](const float* x, const float* y, float* z) {
        mm_tiled_kernel<<<grid_for(M, N), dim3(T, T)>>>(x, y, z, M, K, N);
    });
}
