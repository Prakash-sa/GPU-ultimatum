// Classic kernel 2: rotary position embedding (RoPE), in place.
// x: T x H x D (tokens, heads, head dim), D even. "Rotate half" layout (GPT-NeoX/LLaMA):
// element i pairs with element i + D/2, and the pair is rotated by angle pos * theta_i,
//   theta_i = base^(-2i / D),  i = 0 .. D/2 - 1.
//   x1' = x1 cos - x2 sin,   x2' = x2 cos + x1 sin
// One thread per pair. The angles are computed on the fly (no cos/sin tables to read).
#include <cuda_runtime.h>
#include <math.h>

// --8<-- [start:rope]
__global__ void rope_kernel(float* __restrict__ x, int T, int H, int D, float log_base, int pos0) {
    int half = D / 2;
    long long pairs = (long long)T * H * half;
    for (long long p = blockIdx.x * (long long)blockDim.x + threadIdx.x; p < pairs;
         p += (long long)gridDim.x * blockDim.x) {
        int i = int(p % half);                           // which frequency
        long long th = p / half;                         // (token, head)
        int t = int(th / H);
        float theta = expf(-log_base * (2.0f * i) / D);  // base^(-2i/D)
        float ang = float(t + pos0) * theta;
        float c = cosf(ang), s = sinf(ang);
        float* row = x + th * D;
        float a = row[i], b = row[i + half];             // the pair (i, i + D/2)
        row[i] = a * c - b * s;
        row[i + half] = b * c + a * s;
    }
}
// --8<-- [end:rope]

// x is a device pointer; rotated in place. pos0 = position of token 0 (for KV-cache decoding).
extern "C" void rope(float* x, int T, int H, int D, float base, int pos0) {
    long long pairs = (long long)T * H * (D / 2);
    long long blocks = (pairs + 255) / 256;
    rope_kernel<<<int(blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks)), 256>>>(x, T, H, D, logf(base), pos0);
    cudaDeviceSynchronize();
}
