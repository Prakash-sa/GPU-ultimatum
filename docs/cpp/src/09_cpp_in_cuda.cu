// C++ for GPU programmers, part 9: the same C++ inside CUDA.
// Which of parts 1-8 carry into device code, shown on one small, tested library:
//   * RAII: a unique_ptr with a cudaFree deleter owns device memory (parts 4, 5);
//   * a span-like view, passed to kernels by value (part 4);
//   * a kernel template taking a functor or a __device__ lambda (parts 2, 6);
//   * constexpr launch parameters and static_assert (part 8);
//   * templates + if constexpr choosing the accumulator type (part 2).
//
//   nvcc -std=c++20 --extended-lambda -o cpp_in_cuda 09_cpp_in_cuda.cu
// Tested without a GPU by tests/cuda/run_tests.py (CPU emulation) and compiled by nvcc.
#include <cuda_runtime.h>
#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string>
#include <type_traits>

// --8<-- [start:check]
// Host code may throw; device code may not (no exceptions, no RTTI on the GPU).
inline void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + " failed");
}
// --8<-- [end:check]

// --8<-- [start:raii]
// Device memory owned like any other resource: freed exactly once, on every path.
struct CudaFree {
    void operator()(void* p) const { cudaFree(p); }
};
template <typename T>
using device_ptr = std::unique_ptr<T[], CudaFree>;

template <typename T>
device_ptr<T> device_alloc(std::size_t n) {
    void* p = nullptr;
    check(cudaMalloc(&p, n * sizeof(T)), "cudaMalloc");
    return device_ptr<T>(static_cast<T*>(p));
}
// --8<-- [end:raii]

// --8<-- [start:view]
// A (pointer, size) view: trivially copyable, so it can be a kernel argument. std::span
// itself works in device code with libcu++'s cuda::std::span; this is the idea in 6 lines.
template <typename T>
struct View {
    T* data;
    int size;
    __host__ __device__ T& operator[](int i) const { return data[i]; }
};
static_assert(std::is_trivially_copyable_v<View<float>>);
// --8<-- [end:view]

// --8<-- [start:kernel_template]
constexpr int kThreads = 256;                       // compile-time launch parameter
static_assert(kThreads % 32 == 0, "block size must be a whole number of warps");

// One kernel, any element-wise operation: the functor's type is a template parameter, so
// the call inlines. This is how Thrust's transform and CUB's device-wide algorithms work.
template <typename Op>
__global__ void transform_kernel(View<const float> x, View<float> y, Op op) {
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < x.size; i += gridDim.x * blockDim.x)
        y[i] = op(x[i], y[i]);
}

template <typename Op>
void transform(View<const float> x, View<float> y, Op op) {
    int blocks = (x.size + kThreads - 1) / kThreads;
    transform_kernel<Op><<<blocks < 1024 ? (blocks < 1 ? 1 : blocks) : 1024, kThreads>>>(x, y, op);
    check(cudaGetLastError(), "launch");
}

struct Saxpy {                                      // a functor usable on the device
    float a;
    __host__ __device__ float operator()(float x, float y) const { return a * x + y; }
};
// --8<-- [end:kernel_template]

// --8<-- [start:accumulate]
// if constexpr picks the accumulator per type at compile time, inside device code.
template <typename T>
__device__ auto widen(T v) {
    if constexpr (std::is_same_v<T, float>) return double(v);
    else return v;
}

template <typename T>
__global__ void sum_kernel(View<const T> x, double* out) {
    // one block, deterministic: each thread sums a strided slice, then thread 0 adds
    __shared__ double part[kThreads];
    double s = 0.0;
    for (int i = threadIdx.x; i < x.size; i += blockDim.x) s += widen(x[i]);
    part[threadIdx.x] = s;
    __syncthreads();
    if (threadIdx.x == 0) {
        double t = 0.0;
        for (int k = 0; k < kThreads; ++k) t += part[k];
        *out = t;
    }
}
// --8<-- [end:accumulate]

// --8<-- [start:api]
// Host API used by the tests: device pointers in, as in the LeetGPU solutions.
extern "C" void saxpy_functor(const float* x, float* y, int n, float a) {
    transform(View<const float>{x, n}, View<float>{y, n}, Saxpy{a});
    check(cudaDeviceSynchronize(), "sync");
}

extern "C" void saxpy_lambda(const float* x, float* y, int n, float a) {
    // A __device__ lambda: needs nvcc --extended-lambda. Captures `a` by value.
    transform(View<const float>{x, n}, View<float>{y, n},
              [a] __device__(float xi, float yi) { return a * xi + yi; });
    check(cudaDeviceSynchronize(), "sync");
}

extern "C" double sum_raii(const float* host_x, int n) {
    auto d_x = device_alloc<float>(n);              // freed automatically on return or throw
    auto d_out = device_alloc<double>(1);
    check(cudaMemcpy(d_x.get(), host_x, n * sizeof(float), cudaMemcpyHostToDevice), "copy in");
    sum_kernel<float><<<1, kThreads>>>(View<const float>{d_x.get(), n}, d_out.get());
    double out = 0.0;
    check(cudaMemcpy(&out, d_out.get(), sizeof(double), cudaMemcpyDeviceToHost), "copy out");
    return out;
}
// --8<-- [end:api]
