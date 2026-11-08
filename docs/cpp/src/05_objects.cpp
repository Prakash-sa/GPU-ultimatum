// C++ for GPU programmers, part 5: objects and resource management.
// Topics 21-29: structs, classes, constructors, destructors, copy constructors, operator
// overloading, std::move, move constructors, push_back vs emplace_back.
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 05_objects.cpp && ./a.out
#include <cassert>
#include <cstddef>
#include <cstring>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

// --8<-- [start:structs]
// A struct groups data. Members are public by default. A plain struct of floats is what
// CUDA's float2/float3/float4 are.
struct Vec3 {
    float x = 0.0f, y = 0.0f, z = 0.0f;     // default member initializers
};

// Array of structures (AoS) vs structure of arrays (SoA): the layout question of every
// GPU kernel. With SoA, thread i reading x[i] makes a warp read 128 contiguous bytes.
struct ParticlesAoS { std::vector<Vec3> p; };
struct ParticlesSoA { std::vector<float> x, y, z; };

void structs() {
    Vec3 a{1.0f, 2.0f, 3.0f};               // aggregate initialization, in member order
    Vec3 b{.x = 1.0f, .z = 3.0f};           // designated initializers (C++20)
    assert(a.y == 2.0f && b.y == 0.0f);
    static_assert(sizeof(Vec3) == 12);      // three floats, no padding

    ParticlesAoS aos{std::vector<Vec3>(4)};
    ParticlesSoA soa{std::vector<float>(4), std::vector<float>(4), std::vector<float>(4)};
    // AoS: consecutive x values are 12 bytes apart. SoA: 4 bytes apart (coalesced).
    assert(reinterpret_cast<char*>(&aos.p[1].x) - reinterpret_cast<char*>(&aos.p[0].x) == 12);
    assert(reinterpret_cast<char*>(&soa.x[1]) - reinterpret_cast<char*>(&soa.x[0]) == 4);
}
// --8<-- [end:structs]

// --8<-- [start:classes]
// A class bundles data with the functions that keep it valid. Members are private by
// default; the public interface is the only way in, so the invariant (here: the
// accumulated count matches the data) can't be broken from outside.
class RunningMean {
public:
    void add(double v) {
        ++n_;
        mean_ += (v - mean_) / double(n_);   // Welford's update: no overflow, little cancellation
    }
    double mean() const { return mean_; }     // const: doesn't modify the object
    long count() const { return n_; }

private:
    long n_ = 0;
    double mean_ = 0.0;
};

void classes() {
    RunningMean m;
    for (int i = 1; i <= 100; ++i) m.add(i);
    assert(m.count() == 100 && m.mean() == 50.5);
    // m.n_ = 0;                             // error: n_ is private
}
// --8<-- [end:classes]

// --8<-- [start:lifetime]
// A buffer that owns memory: constructors acquire it, the destructor releases it. This is
// RAII ("resource acquisition is initialization"), and it is how to wrap cudaMalloc,
// streams, events and cuBLAS handles: the CUDA page has the device version.
class Buffer {
public:
    Buffer() = default;                                       // empty buffer
    explicit Buffer(std::size_t n) : n_(n), data_(new float[n]()) { ++live; }   // constructor
    ~Buffer() {                                               // destructor: always runs
        if (data_) --live;
        delete[] data_;
    }

    Buffer(const Buffer& other) : n_(other.n_), data_(new float[other.n_]) {   // copy: deep
        std::memcpy(data_, other.data_, n_ * sizeof(float));
        ++live;
        ++copies;
    }
    Buffer& operator=(const Buffer& other) {                  // copy assignment
        if (this != &other) {
            Buffer tmp(other);                                // copy, then swap: exception safe
            swap(tmp);
        }
        return *this;
    }
    Buffer(Buffer&& other) noexcept                           // move: steal the pointer
        : n_(std::exchange(other.n_, 0)), data_(std::exchange(other.data_, nullptr)) {}
    Buffer& operator=(Buffer&& other) noexcept {
        swap(other);                                          // other's destructor frees ours
        return *this;
    }

    void swap(Buffer& o) noexcept {
        std::swap(n_, o.n_);
        std::swap(data_, o.data_);
    }
    float& operator[](std::size_t i) { return data_[i]; }
    std::size_t size() const { return n_; }
    const float* data() const { return data_; }

    static inline int live = 0;                               // buffers currently allocated
    static inline int copies = 0;

private:
    std::size_t n_ = 0;
    float* data_ = nullptr;
};

void lifetime() {
    {
        Buffer a(1000);
        a[0] = 1.0f;
        assert(Buffer::live == 1);
        Buffer b = a;                       // copy constructor: a second allocation
        b[0] = 2.0f;
        assert(Buffer::live == 2 && a[0] == 1.0f);
        Buffer c = std::move(a);            // move constructor: no allocation, a is emptied
        assert(Buffer::live == 2 && a.size() == 0 && c[0] == 1.0f);
    }                                       // destructors run here, in reverse order
    assert(Buffer::live == 0);              // nothing leaked
}
// --8<-- [end:lifetime]

// --8<-- [start:operators]
// Operator overloading: give user types natural syntax. CUDA code defines these for
// float2/float4 (the CUDA samples ship them in helper_math.h).
struct Complex {
    float re, im;
};
Complex operator+(Complex a, Complex b) { return {a.re + b.re, a.im + b.im}; }
Complex operator*(Complex a, Complex b) {
    return {a.re * b.re - a.im * b.im, a.re * b.im + a.im * b.re};
}
bool operator==(Complex a, Complex b) { return a.re == b.re && a.im == b.im; }
std::ostream& operator<<(std::ostream& os, Complex c) {
    return os << c.re << (c.im < 0 ? " - " : " + ") << (c.im < 0 ? -c.im : c.im) << "i";
}

// operator() makes an object callable (a functor): see part 6.
struct Twiddle {
    int n;
    Complex operator()(int k) const { return k % 4 == 0 ? Complex{1, 0} : Complex{0, -1}; }
};

void operators() {
    Complex i{0, 1};
    assert(i * i == (Complex{-1, 0}));
    assert(i + i == (Complex{0, 2}));
    std::cout << "i * i = " << i * i << '\n';
    Twiddle w{4};
    assert(w(0) == (Complex{1, 0}));
}
// --8<-- [end:operators]

// --8<-- [start:move]
// std::move doesn't move anything: it casts to an rvalue reference, which says "I'm done
// with this, you may steal its insides". The move constructor does the stealing.
std::vector<float> make_field(std::size_t n) {
    std::vector<float> f(n, 1.0f);
    return f;                               // returned by move (or elided): no copy
}

void moves() {
    std::vector<float> a = make_field(1 << 20);
    const float* storage = a.data();
    std::vector<float> b = std::move(a);    // b takes a's buffer: O(1), no allocation
    assert(b.data() == storage);
    assert(a.empty());                      // moved-from: valid but unspecified; here empty

    std::string s = "a long string that does not fit the small-string buffer";
    std::string t = std::move(s);
    assert(t.size() > 40);
    // After a move, only assign to or destroy the moved-from object.
    s = "reused";
    assert(s == "reused");
}
// --8<-- [end:move]

// --8<-- [start:emplace]
struct Kernel {
    std::string name;
    int blocks, threads;
    Kernel(std::string n, int b, int t) : name(std::move(n)), blocks(b), threads(t) {}
};

void push_vs_emplace() {
    std::vector<Kernel> launches;
    launches.reserve(2);
    launches.push_back(Kernel("saxpy", 4096, 256));   // builds a temporary, then moves it in
    launches.emplace_back("reduce", 528, 256);        // builds in place from the arguments
    assert(launches[1].name == "reduce" && launches[1].blocks == 528);

    // Without reserve, growth moves every element to the new buffer, and only if the
    // move constructor is noexcept; otherwise it copies. Buffer's move is noexcept:
    Buffer::copies = 0;
    std::vector<Buffer> bufs;
    for (int i = 0; i < 5; ++i) bufs.emplace_back(16);
    assert(Buffer::copies == 0);                      // reallocations moved, never copied
}
// --8<-- [end:emplace]

int main() {
    structs();
    classes();
    lifetime();
    operators();
    moves();
    push_vs_emplace();
    std::cout << "05_objects: all checks passed\n";
}
