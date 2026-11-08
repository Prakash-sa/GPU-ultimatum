// C++ for GPU programmers, part 2: functions and templates.
// Topics 7-10: functions, overloading, function templates, template specialization.
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 02_functions.cpp && ./a.out
#include <cassert>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <type_traits>

// --8<-- [start:functions]
// A declaration says what a function takes and returns; the definition gives the body.
int ceil_div(int a, int b);

int ceil_div(int a, int b) { return (a + b - 1) / b; }

// Default arguments fill in from the right.
int blocks_for(int n, int threads_per_block = 256) { return ceil_div(n, threads_per_block); }

// Several results: return a struct (structured bindings unpack it at the call site).
struct LaunchConfig {
    int blocks;
    int threads;
};
LaunchConfig launch_config(int n) { return {blocks_for(n), 256}; }

void functions() {
    assert(ceil_div(1000, 256) == 4);
    assert(blocks_for(1000, 128) == 8);
    auto [blocks, threads] = launch_config(1'000'000);
    assert(blocks == 3907 && threads == 256);
}
// --8<-- [end:functions]

// --8<-- [start:overloading]
// Same name, different parameter types: the compiler picks by the argument types.
// CUDA's math library does exactly this: sqrt(float) vs sqrt(double), atomicAdd(int*) vs
// atomicAdd(float*).
std::string describe(int) { return "int"; }
std::string describe(float) { return "float"; }
std::string describe(double) { return "double"; }
std::string describe(const float*, std::size_t n) { return "array of " + std::to_string(n) + " floats"; }

void overloading() {
    assert(describe(1) == "int");
    assert(describe(1.0f) == "float");
    assert(describe(1.0) == "double");         // a literal without 'f' is a double
    float buf[4]{};
    assert(describe(buf, 4) == "array of 4 floats");
    // describe(1u) would be ambiguous: unsigned converts equally well to int, float, double.
}
// --8<-- [end:overloading]

// --8<-- [start:templates]
// One definition, instantiated per type at compile time: no runtime cost. Every CUDA
// kernel library (CUB, Thrust, CUTLASS) is built from templates like this.
template <typename T>
T axpy(T a, T x, T y) { return a * x + y; }

// Non-type template parameters: values known at compile time, e.g. a tile size. The
// compiler can unroll loops over them completely.
template <typename T, int N>
T sum_fixed(const T (&v)[N]) {
    T s{};
    for (int i = 0; i < N; ++i) s += v[i];
    return s;
}

void templates() {
    assert(axpy(2, 3, 4) == 10);               // T deduced as int
    assert(axpy(2.0f, 3.0f, 4.0f) == 10.0f);   // T = float
    assert(axpy<double>(2, 3.5, 1) == 8.0);    // explicit T: the int args convert
    int tile[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    assert(sum_fixed(tile) == 36);             // T = int, N = 8, both deduced
}
// --8<-- [end:templates]

// --8<-- [start:specialization]
// The primary template handles the general case...
template <typename T>
struct Accumulator {
    using type = T;                            // accumulate in the input type
};
// ...and a full specialization replaces it for one type. Accumulating half precision in
// half loses everything past ~2048 terms, so the specialization widens it.
struct half_t { std::uint16_t bits; };
template <>
struct Accumulator<half_t> {
    using type = float;
};
template <>
struct Accumulator<float> {
    using type = double;                       // e.g. for a reproducible reference sum
};

// Function templates can be specialized too (overloading is usually the cleaner tool).
template <typename T> const char* type_name() { return "unknown"; }
template <> const char* type_name<float>() { return "float"; }
template <> const char* type_name<double>() { return "double"; }

// if constexpr: pick a code path per type at compile time, inside one template.
template <typename T>
double machine_epsilon() {
    if constexpr (std::is_same_v<T, float>) return 1.0 / (1 << 23);
    else if constexpr (std::is_same_v<T, double>) return std::ldexp(1.0, -52);
    else return 0.0;
}

void specialization() {
    static_assert(std::is_same_v<Accumulator<int>::type, int>);
    static_assert(std::is_same_v<Accumulator<half_t>::type, float>);
    static_assert(std::is_same_v<Accumulator<float>::type, double>);
    assert(std::string(type_name<float>()) == "float");
    assert(std::string(type_name<int>()) == "unknown");
    assert(machine_epsilon<float>() == 1.1920928955078125e-07);
}
// --8<-- [end:specialization]

int main() {
    functions();
    overloading();
    templates();
    specialization();
    std::cout << "02_functions: all checks passed\n";
}
