// S4. Capture a time step in a CUDA Graph and replay it (practice problem in LeetGPU's
// format; not on LeetGPU). Page: s4-cuda-graph.md
//
// Same update as S3: `steps` explicit steps of the 1-D heat equation on u0 (n points, end
// points fixed), result in u. Each step is a tiny kernel, so launch overhead dominates.
// Capture one step once, replay it `steps` times. Kernel arguments are frozen at capture,
// so the ping-pong between the two buffers is driven by a parity flag in device memory.
#include <cuda_runtime.h>

__global__ void heat_step(float* buf0, float* buf1, const int* parity, int n, float alpha) {
    const float* src = *parity ? buf1 : buf0;     // read the flag on the device, every replay
    float* dst = *parity ? buf0 : buf1;
    int i = 1 + blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n - 1) dst[i] = src[i] + alpha * (src[i - 1] - 2.0f * src[i] + src[i + 1]);
}

__global__ void flip(int* parity) { *parity ^= 1; }

extern "C" void solve(const float* u0, float* u, int n, int steps, float alpha) {
    const int T = 256;
    float *buf0, *buf1;
    int* parity;
    cudaMalloc(&buf0, sizeof(float) * n);
    cudaMalloc(&buf1, sizeof(float) * n);
    cudaMalloc(&parity, sizeof(int));
    cudaMemcpy(buf0, u0, sizeof(float) * n, cudaMemcpyDeviceToDevice);
    cudaMemcpy(buf1, u0, sizeof(float) * n, cudaMemcpyDeviceToDevice);   // both hold the fixed ends
    cudaMemset(parity, 0, sizeof(int));

    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    cudaStreamBeginCapture(s, cudaStreamCaptureModeGlobal);   // record, don't run
    if (n > 2) heat_step<<<(n - 2 + T - 1) / T, T, 0, s>>>(buf0, buf1, parity, n, alpha);
    flip<<<1, 1, 0, s>>>(parity);
    cudaStreamEndCapture(s, &graph);
    cudaGraphInstantiate(&exec, graph, 0);

    for (int t = 0; t < steps; ++t) cudaGraphLaunch(exec, s);  // one launch per step
    cudaStreamSynchronize(s);
    cudaMemcpy(u, steps % 2 ? buf1 : buf0, sizeof(float) * n, cudaMemcpyDeviceToDevice);

    cudaGraphExecDestroy(exec);
    cudaGraphDestroy(graph);
    cudaStreamDestroy(s);
    cudaFree(buf0);
    cudaFree(buf1);
    cudaFree(parity);
}
