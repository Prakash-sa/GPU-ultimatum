// Classic kernel 5: bitonic sort (ascending floats, any n).
// A sorting network: log2(n) * (log2(n) + 1) / 2 stages of compare-exchange, each stage
// fully parallel (n/2 independent pairs, partner = i XOR j). The input is padded to the
// next power of two with +inf.
//   * stages with partner distance j < 2*B run inside one block, in shared memory
//     (many stages, one launch, one DRAM read and write);
//   * stages with larger j run as one global kernel each.
// O(n log^2 n) work: more than radix sort, but branch-free, data-independent and simple.
#include <cuda_runtime.h>
#include <math.h>

constexpr int B = 256;                                   // threads per block; a block sorts 2B elements

__device__ __forceinline__ void compare_exchange(float& a, float& b, bool ascending) {
    if ((a > b) == ascending) { float t = a; a = b; b = t; }
}

// --8<-- [start:shared]
// All stages (k, j) with j < 2B, for one 2B-element chunk, in shared memory.
// If k_max == 2B this sorts each chunk from scratch; otherwise it finishes merge step k.
__global__ void bitonic_shared(float* data, int k_first, int k_last) {
    __shared__ float s[2 * B];
    long long base = (long long)blockIdx.x * 2 * B;
    s[threadIdx.x] = data[base + threadIdx.x];
    s[threadIdx.x + B] = data[base + threadIdx.x + B];
    __syncthreads();
    for (int k = k_first; k <= k_last; k <<= 1) {
        for (int j = (k == k_first && k_first > 2 * B ? B : k / 2); j > 0; j >>= 1) {
            int t = threadIdx.x;
            int i = 2 * t - (t & (j - 1));               // first index of this thread's pair
            long long gi = base + i;
            bool asc = (gi & k) == 0;                    // direction from the global index
            compare_exchange(s[i], s[i + j], asc);
            __syncthreads();
        }
    }
    data[base + threadIdx.x] = s[threadIdx.x];
    data[base + threadIdx.x + B] = s[threadIdx.x + B];
}
// --8<-- [end:shared]

// --8<-- [start:global]
// One stage (k, j) with j >= 2B: every thread compare-exchanges one pair in global memory.
__global__ void bitonic_global(float* data, long long n, int k, int j) {
    long long t = blockIdx.x * (long long)blockDim.x + threadIdx.x;
    if (t >= n / 2) return;
    long long i = 2 * t - (t & (j - 1));
    bool asc = (i & k) == 0;
    float a = data[i], b = data[i + j];
    if ((a > b) == asc) { data[i] = b; data[i + j] = a; }
}
// --8<-- [end:global]

__global__ void pad_copy(const float* in, float* out, long long n, long long np) {
    for (long long i = blockIdx.x * (long long)blockDim.x + threadIdx.x; i < np; i += (long long)gridDim.x * blockDim.x)
        out[i] = i < n ? in[i] : INFINITY;
}

// x is a device pointer, sorted in place (ascending).
extern "C" void bitonic_sort(float* x, int n) {
    if (n <= 1) return;
    long long np = 2 * B;
    while (np < n) np <<= 1;                             // power of two, at least 2B
    float* buf;
    cudaMalloc(&buf, np * sizeof(float));
    int pblocks = int((np + 255) / 256 < 65535 ? (np + 255) / 256 : 65535);
    pad_copy<<<pblocks, 256>>>(x, buf, n, np);
    int chunks = int(np / (2 * B));
    bitonic_shared<<<chunks, B>>>(buf, 2, 2 * B);        // sort every 2B chunk
    for (long long k = 4 * B; k <= np; k <<= 1) {
        for (long long j = k / 2; j >= 2 * B; j >>= 1) {
            long long pairs = np / 2;
            bitonic_global<<<int((pairs + 255) / 256), 256>>>(buf, np, int(k), int(j));
        }
        bitonic_shared<<<chunks, B>>>(buf, int(k), int(k)); // finish j < 2B in shared memory
    }
    cudaMemcpy(x, buf, n * sizeof(float), cudaMemcpyDeviceToDevice);
    cudaFree(buf);
}
