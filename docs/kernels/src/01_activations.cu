// Classic kernel 1: fused activations.
//   bias_gelu:  y = GELU(x + bias)            x: M x N, bias: N   (exact erf GELU)
//   swiglu:     y = SiLU(gate) * up           gate_up: M x 2H (gate = first H, up = last H)
// Unfused, each is 2-3 kernels with a full read+write of an intermediate per step. Fused,
// every element is read once and written once: the kernel runs at memory bandwidth.
#include <cuda_runtime.h>
#include <math.h>

constexpr int kThreads = 256;

__device__ __forceinline__ float gelu(float v) { return 0.5f * v * (1.0f + erff(v * 0.70710678f)); }
__device__ __forceinline__ float silu(float v) { return v / (1.0f + expf(-v)); }

// --8<-- [start:bias_gelu]
__global__ void bias_gelu_vec4(const float4* __restrict__ x, const float4* __restrict__ bias,
                               float4* __restrict__ y, long long total4, int n4) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total4;
         i += (long long)gridDim.x * blockDim.x) {
        float4 v = x[i], b = bias[i % n4];               // 16-byte loads; bias stays in L1/L2
        y[i] = make_float4(gelu(v.x + b.x), gelu(v.y + b.y), gelu(v.z + b.z), gelu(v.w + b.w));
    }
}
// --8<-- [end:bias_gelu]

__global__ void bias_gelu_scalar(const float* __restrict__ x, const float* __restrict__ bias,
                                 float* __restrict__ y, long long total, int n) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x)
        y[i] = gelu(x[i] + bias[i % n]);
}

// --8<-- [start:swiglu]
__global__ void swiglu_kernel(const float* __restrict__ gate_up, float* __restrict__ y, int M, int H) {
    long long total = (long long)M * H;
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < total;
         i += (long long)gridDim.x * blockDim.x) {
        long long row = i / H, col = i % H;
        const float* r = gate_up + row * 2 * H;          // [gate | up] for this row
        y[i] = silu(r[col]) * r[H + col];
    }
}
// --8<-- [end:swiglu]

static int grid_for(long long work) {
    long long b = (work + kThreads - 1) / kThreads;
    return int(b < 1 ? 1 : (b > 65535 ? 65535 : b));
}

// x, bias, y, gate_up are device pointers
extern "C" void bias_gelu(const float* x, const float* bias, float* y, int M, int N) {
    long long total = (long long)M * N;
    if (N % 4 == 0)
        bias_gelu_vec4<<<grid_for(total / 4), kThreads>>>(reinterpret_cast<const float4*>(x),
                                                          reinterpret_cast<const float4*>(bias),
                                                          reinterpret_cast<float4*>(y), total / 4, N / 4);
    else
        bias_gelu_scalar<<<grid_for(total), kThreads>>>(x, bias, y, total, N);
    cudaDeviceSynchronize();
}

extern "C" void swiglu(const float* gate_up, float* y, int M, int H) {
    swiglu_kernel<<<grid_for((long long)M * H), kThreads>>>(gate_up, y, M, H);
    cudaDeviceSynchronize();
}
