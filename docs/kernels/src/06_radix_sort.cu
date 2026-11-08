// Classic kernel 6: LSD radix sort of (uint32 key, uint32 value) pairs, 4 bits per pass.
// Eight passes; each pass is stable, so the order from earlier (lower) digits survives.
// Per pass, three kernels:
//   1. count:   each block counts how many of its keys have each digit (16 bins)
//   2. scan:    exclusive scan over counts laid out digit-major [digit][block]: gives
//               every (digit, block) its first output slot
//   3. scatter: each element goes to  slot(digit, block) + its rank among equal digits
//               earlier in the block. The rank uses __match_any_sync inside a warp and
//               per-warp counts across warps, so equal keys keep their order (stable).
// O(8 n) work, the classic GPU sort (CUB's DeviceRadixSort is a tuned version of this).
#include <cuda_runtime.h>

constexpr int B = 256;                                   // threads (= elements) per block
constexpr int W = B / 32;
constexpr int RADIX = 16;
constexpr unsigned FULL = 0xffffffffu;

// --8<-- [start:count]
__global__ void radix_count(const unsigned* keys, int n, int shift, unsigned* counts, int nblocks) {
    __shared__ unsigned hist[RADIX];
    if (threadIdx.x < RADIX) hist[threadIdx.x] = 0;
    __syncthreads();
    int i = blockIdx.x * B + threadIdx.x;
    if (i < n) atomicAdd(&hist[(keys[i] >> shift) & 15u], 1u);
    __syncthreads();
    if (threadIdx.x < RADIX) counts[threadIdx.x * nblocks + blockIdx.x] = hist[threadIdx.x];   // digit-major
}
// --8<-- [end:count]

// Exclusive scan of `len` counts in place, one block of 1024 threads with a running carry.
__global__ void scan_exclusive(unsigned* a, int len) {
    __shared__ unsigned warp_sum[32];
    __shared__ unsigned carry;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    if (threadIdx.x == 0) carry = 0;
    __syncthreads();
    for (int base = 0; base < len; base += blockDim.x) {
        int i = base + threadIdx.x;
        unsigned v = i < len ? a[i] : 0u, x = v;
        for (int o = 1; o < 32; o <<= 1) {               // inclusive warp scan
            unsigned y = __shfl_up_sync(FULL, x, o);
            if (lane >= o) x += y;
        }
        if (lane == 31) warp_sum[w] = x;
        __syncthreads();
        if (w == 0) {
            unsigned t = lane < int(blockDim.x >> 5) ? warp_sum[lane] : 0u, u = t;
            for (int o = 1; o < 32; o <<= 1) {
                unsigned y = __shfl_up_sync(FULL, u, o);
                if (lane >= o) u += y;
            }
            warp_sum[lane] = u - t;                      // exclusive offsets of the warps
        }
        __syncthreads();
        unsigned incl = x + warp_sum[w] + carry;
        if (i < len) a[i] = incl - v;                    // exclusive
        __syncthreads();
        if (threadIdx.x == blockDim.x - 1) carry = incl;
        __syncthreads();
    }
}

// --8<-- [start:scatter]
__global__ void radix_scatter(const unsigned* keys, const unsigned* vals, unsigned* keys_out,
                              unsigned* vals_out, int n, int shift, const unsigned* offsets, int nblocks) {
    __shared__ unsigned warp_count[W][RADIX + 1];
    int i = blockIdx.x * B + threadIdx.x;
    int lane = threadIdx.x & 31, w = threadIdx.x >> 5;
    unsigned key = i < n ? keys[i] : 0u;
    unsigned d = i < n ? (key >> shift) & 15u : RADIX;   // tail lanes: their own group
    unsigned peers = __match_any_sync(FULL, d);          // lanes of this warp with my digit
    unsigned rank_in_warp = __popc(peers & ((1u << lane) - 1u));
    for (int b = threadIdx.x; b < W * (RADIX + 1); b += B) (&warp_count[0][0])[b] = 0;
    __syncthreads();
    if (rank_in_warp == 0) warp_count[w][d] = __popc(peers);   // the group leader writes
    __syncthreads();
    if (i < n) {
        unsigned before = 0;
        for (int v = 0; v < w; ++v) before += warp_count[v][d];  // equal digits in earlier warps
        unsigned dst = offsets[d * nblocks + blockIdx.x] + before + rank_in_warp;
        keys_out[dst] = key;
        vals_out[dst] = vals[i];
    }
}
// --8<-- [end:scatter]

// keys, vals are device pointers; sorted in place by key (stable), vals permuted alongside.
extern "C" void radix_sort_pairs(unsigned* keys, unsigned* vals, int n) {
    if (n <= 1) return;
    int nblocks = (n + B - 1) / B;
    unsigned *k2, *v2, *counts;
    cudaMalloc(&k2, size_t(n) * sizeof(unsigned));
    cudaMalloc(&v2, size_t(n) * sizeof(unsigned));
    cudaMalloc(&counts, size_t(RADIX) * nblocks * sizeof(unsigned));
    unsigned *kin = keys, *vin = vals, *kout = k2, *vout = v2;
    for (int shift = 0; shift < 32; shift += 4) {
        radix_count<<<nblocks, B>>>(kin, n, shift, counts, nblocks);
        scan_exclusive<<<1, 1024>>>(counts, RADIX * nblocks);
        radix_scatter<<<nblocks, B>>>(kin, vin, kout, vout, n, shift, counts, nblocks);
        unsigned* t = kin; kin = kout; kout = t;
        t = vin; vin = vout; vout = t;
    }
    // 8 passes: an even number of swaps, so the result is back in keys/vals.
    cudaDeviceSynchronize();
    cudaFree(k2);
    cudaFree(v2);
    cudaFree(counts);
}
