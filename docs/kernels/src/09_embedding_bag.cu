// Classic kernel 9: embedding bag (gather + reduce), the core of recommender models.
//   out[b, :] = sum (or mean) of table[indices[j], :] for j in [offsets[b], offsets[b+1])
// The bags are stored like CSR rows: `offsets` has B + 1 entries. One block per bag, its
// threads spread across the embedding dimension D, so every gathered row is read with
// coalesced loads. A plain embedding lookup is the special case of one index per bag.
#include <cuda_runtime.h>

// --8<-- [start:bag]
__global__ void embedding_bag_kernel(const float* __restrict__ table, const int* __restrict__ indices,
                                     const int* __restrict__ offsets, float* __restrict__ out,
                                     int D, int mean) {
    int b = blockIdx.x;
    int start = offsets[b], end = offsets[b + 1];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {  // threads across the row: coalesced
        float acc = 0.0f;
        for (int j = start; j < end; ++j)
            acc += table[(long long)indices[j] * D + d]; // gather: a random row, read contiguously
        if (mean && end > start) acc /= float(end - start);
        out[(long long)b * D + d] = acc;                 // empty bag: zeros
    }
}
// --8<-- [end:bag]

// table, indices, offsets (B + 1 entries), out are device pointers. mean: 0 = sum, 1 = mean.
extern "C" void embedding_bag(const float* table, const int* indices, const int* offsets, float* out,
                              int B, int D, int mean) {
    int threads = D >= 256 ? 256 : ((D + 31) / 32) * 32;
    embedding_bag_kernel<<<B, threads>>>(table, indices, offsets, out, D, mean);
    cudaDeviceSynchronize();
}
