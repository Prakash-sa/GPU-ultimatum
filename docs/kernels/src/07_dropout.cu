// Classic kernel 7: dropout with a counter-based RNG (Philox4x32-10).
//   y[i] = keep(i) ? x[i] / (1 - p) : 0,    keep(i) = uniform(seed, i) >= p
// A counter-based generator has no state: the random number for element i is a pure
// function of (seed, i). So every thread computes its own numbers independently, the
// forward and backward passes regenerate the same mask without storing it, and the
// result does not depend on the launch configuration. Philox4x32-10 is the generator
// cuRAND and PyTorch use; one call gives 4 numbers, so each thread handles 4 elements.
#include <cuda_runtime.h>
#include <stdint.h>

// --8<-- [start:philox]
struct U4 { uint32_t x, y, z, w; };

__host__ __device__ inline void mulhilo(uint32_t a, uint32_t b, uint32_t& hi, uint32_t& lo) {
    uint64_t p = uint64_t(a) * b;
    hi = uint32_t(p >> 32);
    lo = uint32_t(p);
}

__host__ __device__ inline U4 philox4x32_10(U4 c, uint32_t k0, uint32_t k1) {
    for (int r = 0; r < 10; ++r) {
        uint32_t hi0, lo0, hi1, lo1;
        mulhilo(0xD2511F53u, c.x, hi0, lo0);
        mulhilo(0xCD9E8D57u, c.z, hi1, lo1);
        c = U4{hi1 ^ c.y ^ k0, lo1, hi0 ^ c.w ^ k1, lo0};
        k0 += 0x9E3779B9u;                               // Weyl sequence for the key
        k1 += 0xBB67AE85u;
    }
    return c;
}

__host__ __device__ inline float to_unit(uint32_t r) { return float(r >> 8) * (1.0f / 16777216.0f); }   // [0, 1)
// --8<-- [end:philox]

// --8<-- [start:dropout]
__global__ void dropout_kernel(const float* __restrict__ x, float* __restrict__ y, unsigned char* __restrict__ mask,
                               long long n, float p, uint64_t seed) {
    float scale = 1.0f / (1.0f - p);
    long long groups = (n + 3) / 4;
    for (long long g = blockIdx.x * (long long)blockDim.x + threadIdx.x; g < groups;
         g += (long long)gridDim.x * blockDim.x) {
        U4 r = philox4x32_10(U4{uint32_t(g), uint32_t(uint64_t(g) >> 32), 0u, 0u},
                             uint32_t(seed), uint32_t(seed >> 32));   // counter = group index
        uint32_t rs[4] = {r.x, r.y, r.z, r.w};
        for (int q = 0; q < 4; ++q) {
            long long i = 4 * g + q;
            if (i >= n) break;
            bool keep = to_unit(rs[q]) >= p;
            y[i] = keep ? x[i] * scale : 0.0f;
            mask[i] = keep;
        }
    }
}
// --8<-- [end:dropout]

// x, y, mask are device pointers.
extern "C" void dropout(const float* x, float* y, unsigned char* mask, long long n, float p, unsigned long long seed) {
    long long groups = (n + 3) / 4, blocks = (groups + 255) / 256;
    dropout_kernel<<<int(blocks < 1 ? 1 : (blocks > 4096 ? 4096 : blocks)), 256>>>(x, y, mask, n, p, seed);
    cudaDeviceSynchronize();
}

// Host copy of the generator, for tests: the 4 outputs for counter `g`.
extern "C" void philox_host(unsigned long long g, unsigned long long seed, unsigned* out4) {
    U4 r = philox4x32_10(U4{uint32_t(g), uint32_t(g >> 32), 0u, 0u}, uint32_t(seed), uint32_t(seed >> 32));
    out4[0] = r.x; out4[1] = r.y; out4[2] = r.z; out4[3] = r.w;
}

// Raw generator, for known-answer tests against the Random123 reference vectors.
extern "C" void philox_raw(const unsigned* ctr4, const unsigned* key2, unsigned* out4) {
    U4 r = philox4x32_10(U4{ctr4[0], ctr4[1], ctr4[2], ctr4[3]}, key2[0], key2[1]);
    out4[0] = r.x; out4[1] = r.y; out4[2] = r.z; out4[3] = r.w;
}
