// Classic kernel 10: 2-D max pooling with argmax indices (NCHW, no padding).
//   out[n, c, oh, ow] = max over the k x k window starting at (oh * s, ow * s)
//   idx[n, c, oh, ow] = flat index (h * W + w) of that max, for the backward pass
// One thread per output. Neighbouring threads (along ow) read neighbouring windows, so
// the loads are close together and mostly served by L1. The backward pass scatters each
// gradient to idx: with overlapping windows (s < k) that needs atomics.
#include <cuda_runtime.h>
#include <math.h>

// --8<-- [start:pool]
__global__ void maxpool2d_kernel(const float* __restrict__ x, float* __restrict__ y, int* __restrict__ idx,
                                 long long planes, int H, int W, int k, int s, int OH, int OW) {
    long long total = planes * OH * OW;
    for (long long o = blockIdx.x * (long long)blockDim.x + threadIdx.x; o < total;
         o += (long long)gridDim.x * blockDim.x) {
        int ow = int(o % OW), oh = int((o / OW) % OH);
        long long plane = o / ((long long)OH * OW);      // (n, c)
        const float* xp = x + plane * H * W;
        float best = -INFINITY;
        int arg = (oh * s) * W + ow * s;
        for (int u = 0; u < k; ++u)
            for (int v = 0; v < k; ++v) {
                int h = oh * s + u, w = ow * s + v;
                float val = xp[h * W + w];
                if (val > best || isnan(val)) { best = val; arg = h * W + w; }   // first max wins; NaN propagates
            }
        y[o] = best;
        idx[o] = arg;
    }
}
// --8<-- [end:pool]

// x, y, idx are device pointers. x: N x C x H x W; y, idx: N x C x OH x OW.
extern "C" void maxpool2d(const float* x, float* y, int* idx, int N, int C, int H, int W, int k, int s) {
    int OH = (H - k) / s + 1, OW = (W - k) / s + 1;
    long long total = (long long)N * C * OH * OW, blocks = (total + 255) / 256;
    maxpool2d_kernel<<<int(blocks < 1 ? 1 : (blocks > 65535 ? 65535 : blocks)), 256>>>(
        x, y, idx, (long long)N * C, H, W, k, s, OH, OW);
    cudaDeviceSynchronize();
}
