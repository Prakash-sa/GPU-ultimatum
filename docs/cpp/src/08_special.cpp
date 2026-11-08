// C++ for GPU programmers, part 8: special topics.
// Topics 43-47: namespaces, pseudo-random numbers, file I/O, constexpr, concepts.
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 08_special.cpp && ./a.out
#include <algorithm>
#include <array>
#include <cassert>
#include <cmath>
#include <concepts>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <random>
#include <string>
#include <vector>

// --8<-- [start:namespaces]
// Namespaces keep names from colliding. CUDA's own libraries live in namespaces too:
// cub::, thrust::, cuda::std::, nvcuda::wmma.
namespace solver {
namespace detail {                           // by convention: internal, not part of the API
inline int halo_width(int order) { return order / 2; }
}  // namespace detail
inline int padded(int n, int order) { return n + 2 * detail::halo_width(order); }
}  // namespace solver

namespace sv = solver;                       // an alias for a long name

void namespaces() {
    assert(solver::padded(64, 2) == 66);
    assert(sv::padded(64, 4) == 68);
    using solver::padded;                    // bring one name in (never `using namespace` in a header)
    assert(padded(10, 2) == 12);
}
// --8<-- [end:namespaces]

// --8<-- [start:random]
// <random> separates the engine (the bit generator) from the distribution (the shape).
// A fixed seed gives the same sequence on every run: essential for reproducible tests.
void random_numbers() {
    std::mt19937 gen(42);                                // Mersenne Twister, seeded
    std::uniform_real_distribution<float> uni(-1.0f, 1.0f);
    std::normal_distribution<double> gauss(0.0, 1.0);
    std::vector<float> x(1000);
    for (auto& v : x) v = uni(gen);
    assert(*std::min_element(x.begin(), x.end()) >= -1.0f);

    std::mt19937 again(42);
    assert(uni(again) == x[0]);                          // same seed, same numbers

    double mean = 0;
    for (int i = 0; i < 100000; ++i) mean += gauss(gen);
    assert(std::abs(mean / 100000) < 0.02);
    // On the GPU each thread needs its own independent stream: cuRAND's Philox generator
    // takes (seed, subsequence = thread id, offset) for exactly that reason. A shared
    // engine like mt19937 can't be used from many threads at once.
}
// --8<-- [end:random]

// --8<-- [start:file_io]
// Text for humans, binary for data: a binary write of a float array is one call and
// round-trips exactly; text loses precision unless you print 9 significant digits.
void file_io() {
    namespace fs = std::filesystem;
    const fs::path dir = fs::temp_directory_path();
    std::vector<float> field{1.5f, 2.25f, -3.0f, 0.1f};

    {   // binary
        std::ofstream out(dir / "field.bin", std::ios::binary);
        out.write(reinterpret_cast<const char*>(field.data()), field.size() * sizeof(float));
    }   // the destructor closes (and flushes) the file
    std::vector<float> back(field.size());
    {
        std::ifstream in(dir / "field.bin", std::ios::binary);
        in.read(reinterpret_cast<char*>(back.data()), back.size() * sizeof(float));
        assert(in.good());
    }
    assert(back == field);                               // bit-exact

    {   // text: a CSV of results
        std::ofstream csv(dir / "timings.csv");
        csv << "kernel,us\n" << "saxpy," << 12.5 << "\n" << "reduce," << 7.25 << "\n";
    }
    std::ifstream csv(dir / "timings.csv");
    std::string line;
    int rows = 0;
    while (std::getline(csv, line)) ++rows;
    assert(rows == 3);
    fs::remove(dir / "field.bin");
    fs::remove(dir / "timings.csv");
}
// --8<-- [end:file_io]

// --8<-- [start:constexpr]
// constexpr functions can run at compile time; consteval ones must. Use them for tile
// sizes, shared-memory layouts and lookup tables, so the numbers are checked by the
// compiler and cost nothing at run time.
constexpr int next_pow2(int n) {
    int p = 1;
    while (p < n) p <<= 1;
    return p;
}

constexpr int smem_bytes(int bm, int bn, int bk, int stages) {
    return (bm * bk + bk * bn) * int(sizeof(float)) * stages;
}

template <int N>
constexpr auto make_weights() {              // a table computed by the compiler
    std::array<float, N> w{};
    for (int i = 0; i < N; ++i) w[i] = 1.0f / float(i + 1);
    return w;
}

void constexpr_demo() {
    static_assert(next_pow2(1000) == 1024);
    static_assert(smem_bytes(128, 128, 32, 3) == 98304);      // 96 KB: fits an H100, not a T4
    static_assert(smem_bytes(128, 128, 32, 3) <= 227 * 1024);
    constexpr auto w = make_weights<4>();
    static_assert(w[3] == 0.25f);
    int runtime_n = 37;
    assert(next_pow2(runtime_n) == 64);      // the same function at run time
}
// --8<-- [end:constexpr]

// --8<-- [start:concepts]
// Concepts name the requirements a template has, so a wrong type gives a one-line error
// at the call, not pages of errors from inside the template.
template <typename T>
concept Scalar = std::floating_point<T> || std::integral<T>;

template <typename T>
concept Reducible = requires(T a, T b) {
    { a + b } -> std::convertible_to<T>;
    T{};                                     // has a zero
};

template <Reducible T>
T total(const std::vector<T>& v) {
    T s{};
    for (const auto& x : v) s += x;
    return s;
}

template <Scalar T>
T fma_like(T a, T b, T c) { return a * b + c; }

void concepts() {
    assert(total(std::vector<int>{1, 2, 3}) == 6);
    assert(total(std::vector<double>{0.5, 0.25}) == 0.75);
    assert(fma_like(2.0f, 3.0f, 1.0f) == 7.0f);
    static_assert(Scalar<float> && !Scalar<std::string>);
    static_assert(Reducible<std::string>);   // + concatenates, and "" is the zero
    // fma_like(std::string("a"), ...) would fail with "constraints not satisfied: Scalar".
}
// --8<-- [end:concepts]

int main() {
    namespaces();
    random_numbers();
    file_io();
    constexpr_demo();
    concepts();
    std::cout << "08_special: all checks passed\n";
}
