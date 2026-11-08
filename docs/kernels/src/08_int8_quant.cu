// Classic kernel 8: symmetric per-row int8 quantization and dequantization.
//   scale[r] = max_j |x[r, j]| / 127
//   q[r, j]  = clamp(round_half_even(x[r, j] * (127 / max)), -127, 127)
//   x'[r, j] = q[r, j] * scale[r]
// One warp per row: a warp-shuffle max, then one pass to quantize. The rounding is
// bit-exact with torch.round (round half to even) because both compute the same single
// multiply, x * (127 / max), in FP32.
#include <cuda_runtime.h>
#include <stdint.h>
#include <math.h>

constexpr int kWarps = 8;

// --8<-- [start:quant]
__global__ void quantize_rows(const float* __restrict__ x, int8_t* __restrict__ q, float* __restrict__ scale,
                              int M, int N) {
    int row = blockIdx.x * kWarps + (threadIdx.x >> 5), lane = threadIdx.x & 31;
    if (row >= M) return;                                // the whole warp leaves together
    const float* xr = x + (long long)row * N;
    float amax = 0.0f;
    for (int j = lane; j < N; j += 32) amax = fmaxf(amax, fabsf(xr[j]));
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, o));
    float inv = amax > 0.0f ? 127.0f / amax : 0.0f;      // all-zero row: q = 0, scale = 0
    for (int j = lane; j < N; j += 32) {
        float v = rintf(xr[j] * inv);                    // round half to even
        q[(long long)row * N + j] = int8_t(fminf(fmaxf(v, -127.0f), 127.0f));
    }
    if (lane == 0) scale[row] = amax / 127.0f;
}
// --8<-- [end:quant]

// --8<-- [start:dequant]
__global__ void dequantize_rows(const int8_t* __restrict__ q, const float* __restrict__ scale,
                                float* __restrict__ y, int M, int N) {
    long long total = (long long)M * N;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x)
        y[i] = float(q[i]) * scale[i / N];
}
// --8<-- [end:dequant]

// x, q, scale, y are device pointers.
extern "C" void quantize_int8(const float* x, int8_t* q, float* scale, int M, int N) {
    quantize_rows<<<(M + kWarps - 1) / kWarps, 32 * kWarps>>>(x, q, scale, M, N);
    cudaDeviceSynchronize();
}
extern "C" void dequantize_int8(const int8_t* q, const float* scale, float* y, int M, int N) {
    long long blocks = ((long long)M * N + 255) / 256;
    dequantize_rows<<<int(blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks)), 256>>>(q, scale, y, M, N);
    cudaDeviceSynchronize();
}
