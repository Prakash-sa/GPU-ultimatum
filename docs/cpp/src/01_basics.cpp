// C++ for GPU programmers, part 1: the basics.
// Topics 0-6 of the "C++ From Scratch" outline: main, printing, variables, conditionals,
// std::array, for loops, while loops. Every example is original and runs as a test:
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 01_basics.cpp && ./a.out
#include <array>
#include <cassert>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <limits>
#include <string_view>

// --8<-- [start:variables]
void variables() {
    int n = 1 << 20;                    // 32-bit signed: element counts up to ~2.1e9
    std::int64_t bytes = 4LL * n;       // byte counts overflow int fast: use 64 bits
    float x = 0.1f;                     // the GPU's native type: 'f' suffix, or it's a double
    double y = 0.1;                     // 8 bytes; 1/2 to 1/64 of FP32 speed on most GPUs
    auto z = x * 2;                     // auto: the compiler deduces float here
    const int threads = 256;            // const: can't change after initialization
    unsigned mask = 0xffffffffu;        // unsigned for bit patterns (warp masks)

    static_assert(sizeof(float) == 4 && sizeof(double) == 8);
    assert(bytes == 4'194'304);         // ' separates digits (C++14)
    assert(std::numeric_limits<int>::max() == 2'147'483'647);
    std::cout << "n=" << n << " bytes=" << bytes << " x=" << x << " y=" << y
              << " z=" << z << " threads=" << threads << " mask=" << std::hex << mask
              << std::dec << '\n';
}
// --8<-- [end:variables]

// --8<-- [start:conditionals]
const char* classify(double intensity_flop_per_byte, double ridge) {
    if (intensity_flop_per_byte < ridge / 4) {
        return "memory bound";
    } else if (intensity_flop_per_byte < ridge) {
        return "near the ridge";
    } else {
        return "compute bound";
    }
}

int round_up(int n, int multiple) {
    // the ternary operator: a value-producing if/else
    return n % multiple == 0 ? n : (n / multiple + 1) * multiple;
}

void conditionals() {
    double ridge = 67e12 / 3.35e12;     // H100: FP32 FLOP/s over bytes/s, ~20 FLOP/byte
    assert(classify(0.25, ridge) == std::string_view("memory bound"));     // vector add
    assert(classify(945, ridge) == std::string_view("compute bound"));     // big SGEMM
    assert(round_up(1000, 32) == 1024 && round_up(1024, 32) == 1024);

    switch (int cc = 90; cc / 10) {     // if/switch with an initializer (C++17)
        case 7: std::puts("Volta/Turing"); break;
        case 8: std::puts("Ampere/Ada"); break;
        case 9: std::puts("Hopper"); break;
        default: std::puts("other");
    }
}
// --8<-- [end:conditionals]

// --8<-- [start:array]
void arrays() {
    std::array<float, 4> v{1.0f, 2.0f, 3.0f, 4.0f};   // size fixed at compile time, on the stack
    std::array<int, 3> zeros{};                        // {} value-initializes: all 0
    assert(v.size() == 4 && zeros[2] == 0);
    assert(v.at(3) == 4.0f);                           // .at() checks bounds (throws), [] doesn't

    // A fixed-size array is a plain value: it copies, compares and returns like an int.
    auto w = v;
    w[0] = 10.0f;
    assert(v[0] == 1.0f && w != v);

    // Multi-dimensional: an array of arrays, row-major like a CUDA tile.
    std::array<std::array<float, 3>, 2> tile{};
    tile[1][2] = 5.0f;
    assert(tile[1][2] == 5.0f && sizeof(tile) == 6 * sizeof(float));   // no overhead
}
// --8<-- [end:array]

// --8<-- [start:for]
void for_loops() {
    std::array<float, 8> a{}, b{}, c{};
    for (int i = 0; i < 8; ++i) {       // index loop: the shape of every CUDA kernel body
        a[i] = float(i);
        b[i] = 2.0f * i;
    }
    for (std::size_t i = 0; i < c.size(); ++i) c[i] = a[i] + b[i];

    float sum = 0.0f;
    for (float value : c) sum += value; // range-for: every element, no index
    assert(sum == 3.0f * 28);

    for (auto& value : c) value = 0.0f; // auto&: modify in place (without &, a copy)
    assert(c[7] == 0.0f);

    // The grid-stride loop, written on the CPU: thread t of T handles t, t+T, t+2T, ...
    const int n = 10, T = 4;
    int visited = 0;
    for (int t = 0; t < T; ++t)
        for (int i = t; i < n; i += T) ++visited;
    assert(visited == n);
}
// --8<-- [end:for]

// --8<-- [start:while]
void while_loops() {
    // while: test first. Halving loop, the shape of a tree reduction's offsets.
    int offset = 16, steps = 0;
    while (offset > 0) {
        offset /= 2;
        ++steps;
    }
    assert(steps == 5);                 // log2(32) shuffle steps per warp reduction

    // do-while: run the body at least once, test after. A retry loop.
    int attempts = 0;
    bool ok = false;
    do {
        ++attempts;
        ok = attempts >= 3;             // stand-in for "the allocation succeeded"
    } while (!ok);
    assert(attempts == 3);

    // break and continue work in every loop.
    int first_big = -1;
    for (int i = 0; i < 100; ++i) {
        if (i % 2) continue;
        if (i * i > 200) { first_big = i; break; }
    }
    assert(first_big == 16);
}
// --8<-- [end:while]

// --8<-- [start:main]
int main() {                            // execution starts here; returning 0 means success
    std::cout << "Hello from the host\n";
    std::printf("printf works too: %d threads, %.2f ms\n", 256, 1.25);
    variables();
    conditionals();
    arrays();
    for_loops();
    while_loops();
    std::cout << "01_basics: all checks passed\n";
    return 0;
}
// --8<-- [end:main]
