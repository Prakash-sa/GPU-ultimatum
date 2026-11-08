// S1. Pipelined copy + compute with streams (practice problem in LeetGPU's format; not on
// LeetGPU). Page: s1-stream-pipeline.md
//
// Given host arrays x and y of length n (pinned memory), compute y[i] = x[i] * x[i] + 1 on
// the GPU. The data starts and ends on the host, so the copies are part of the problem:
// overlap them with the kernel by splitting the work into chunks over num_streams streams.
#include <cuda_runtime.h>

__global__ void square_plus_one(const float* x, float* y, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = x[i] * x[i] + 1.0f;
}

extern "C" void solve(const float* h_x, float* h_y, int n, int num_streams) {
    const int CHUNKS_PER_STREAM = 4;              // more chunks: more overlap, more launches
    const int chunks = num_streams * CHUNKS_PER_STREAM;
    const int chunk = (n + chunks - 1) / chunks;

    float *d_x, *d_y;
    cudaMalloc(&d_x, sizeof(float) * n);
    cudaMalloc(&d_y, sizeof(float) * n);
    cudaStream_t s[8];
    for (int k = 0; k < num_streams; ++k) cudaStreamCreateWithFlags(&s[k], cudaStreamNonBlocking);

    for (int c = 0; c < chunks; ++c) {
        int off = c * chunk, len = n - off < chunk ? n - off : chunk;
        if (len <= 0) break;
        cudaStream_t st = s[c % num_streams];     // round-robin: chunk c+1 copies while c computes
        cudaMemcpyAsync(d_x + off, h_x + off, sizeof(float) * len, cudaMemcpyHostToDevice, st);
        square_plus_one<<<(len + 255) / 256, 256, 0, st>>>(d_x + off, d_y + off, len);
        cudaMemcpyAsync(h_y + off, d_y + off, sizeof(float) * len, cudaMemcpyDeviceToHost, st);
    }

    for (int k = 0; k < num_streams; ++k) {
        cudaStreamSynchronize(s[k]);              // h_y is only complete after this
        cudaStreamDestroy(s[k]);
    }
    cudaFree(d_x);
    cudaFree(d_y);
}
