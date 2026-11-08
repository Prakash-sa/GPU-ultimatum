// S3. Hide a halo exchange behind interior work (practice problem in LeetGPU's format; not
// on LeetGPU). Page: s3-halo-overlap.md
//
// Run `steps` explicit steps of the 1-D heat equation on u0 (n points, the two end points
// fixed) and write the result to u:  u'[i] = u[i] + alpha * (u[i-1] - 2 u[i] + u[i+1]).
// The domain is split into two halves with ghost cells, as if each half lived on its own
// GPU. Each step: edge cells + halo exchange on stream `comm`, interior cells on stream
// `compute`, and events so neither stream runs ahead of the other.
#include <cuda_runtime.h>

struct Half {
    float* cur;   // [0] ghost, [1..m] owned cells, [m+1] ghost
    float* nxt;
    int m;
};

__global__ void heat(const float* cur, float* nxt, int lo, int hi, float alpha) {
    int i = lo + blockIdx.x * blockDim.x + threadIdx.x;
    if (i <= hi) nxt[i] = cur[i] + alpha * (cur[i - 1] - 2.0f * cur[i] + cur[i + 1]);
}

__global__ void heat_edges(const float* cur, float* nxt, int m, float alpha) {
    int i = threadIdx.x == 0 ? 1 : m;            // the two cells next to the ghosts
    nxt[i] = cur[i] + alpha * (cur[i - 1] - 2.0f * cur[i] + cur[i + 1]);
}

extern "C" void solve(const float* u0, float* u, int n, int steps, float alpha) {
    const int T = 256;
    Half h[2];
    h[0].m = (n - 2) / 2;                         // owns global points 1 .. m0
    h[1].m = n - 2 - h[0].m;                      // owns global points m0+1 .. n-2
    const int start[2] = {0, h[0].m};             // global index of each half's ghost [0]
    for (int k = 0; k < 2; ++k) {
        cudaMalloc(&h[k].cur, sizeof(float) * (h[k].m + 2));
        cudaMalloc(&h[k].nxt, sizeof(float) * (h[k].m + 2));
        cudaMemcpy(h[k].cur, u0 + start[k], sizeof(float) * (h[k].m + 2), cudaMemcpyDeviceToDevice);
        cudaMemcpy(h[k].nxt, u0 + start[k], sizeof(float) * (h[k].m + 2), cudaMemcpyDeviceToDevice);
    }
    cudaStream_t comm, compute;
    cudaStreamCreateWithFlags(&comm, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&compute, cudaStreamNonBlocking);
    cudaEvent_t comm_done, compute_done;
    cudaEventCreateWithFlags(&comm_done, cudaEventDisableTiming);
    cudaEventCreateWithFlags(&compute_done, cudaEventDisableTiming);

    for (int t = 0; t < steps; ++t) {
        // Each stream waits for the other's previous step: the edges read interior cells
        // (and overwrite buffers the interior read), and the other way round.
        cudaStreamWaitEvent(comm, compute_done, 0);
        cudaStreamWaitEvent(compute, comm_done, 0);

        for (int k = 0; k < 2; ++k) heat_edges<<<1, 2, 0, comm>>>(h[k].cur, h[k].nxt, h[k].m, alpha);
        // halo exchange: on two GPUs this would be cudaMemcpyPeerAsync or an NCCL send/recv
        cudaMemcpyAsync(h[0].nxt + h[0].m + 1, h[1].nxt + 1, sizeof(float), cudaMemcpyDeviceToDevice, comm);
        cudaMemcpyAsync(h[1].nxt, h[0].nxt + h[0].m, sizeof(float), cudaMemcpyDeviceToDevice, comm);
        cudaEventRecord(comm_done, comm);

        for (int k = 0; k < 2; ++k)
            if (h[k].m > 2)
                heat<<<(h[k].m - 2 + T - 1) / T, T, 0, compute>>>(h[k].cur, h[k].nxt, 2, h[k].m - 1, alpha);
        cudaEventRecord(compute_done, compute);

        for (int k = 0; k < 2; ++k) {
            float* tmp = h[k].cur; h[k].cur = h[k].nxt; h[k].nxt = tmp;
        }
    }

    cudaStreamSynchronize(comm);
    cudaStreamSynchronize(compute);
    cudaMemcpy(u, u0, sizeof(float) * n, cudaMemcpyDeviceToDevice);          // fixed ends
    for (int k = 0; k < 2; ++k)
        cudaMemcpy(u + start[k] + 1, h[k].cur + 1, sizeof(float) * h[k].m, cudaMemcpyDeviceToDevice);

    for (int k = 0; k < 2; ++k) {
        cudaFree(h[k].cur);
        cudaFree(h[k].nxt);
    }
    cudaEventDestroy(comm_done);
    cudaEventDestroy(compute_done);
    cudaStreamDestroy(comm);
    cudaStreamDestroy(compute);
}
