// C++ for GPU programmers, part 6: inheritance, polymorphism and generic code.
// Topics 30-38: inheritance, polymorphism, virtual functions, abstract classes, class
// templates, default comparisons, virtual inheritance, function objects, lambdas.
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 06_polymorphism.cpp && ./a.out
#include <algorithm>
#include <array>
#include <cassert>
#include <compare>
#include <functional>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

// --8<-- [start:inheritance]
// A derived class reuses a base class's members and adds its own.
struct Device {
    std::string name;
    int memory_gb;
    std::string describe() const { return name + " (" + std::to_string(memory_gb) + " GB)"; }
};

struct Gpu : Device {                       // public inheritance: a Gpu *is a* Device
    int sms;
    int cores() const { return sms * 128; }
};

void inheritance() {
    Gpu h100{{"H100", 80}, 132};
    assert(h100.describe() == "H100 (80 GB)");   // inherited member function
    assert(h100.cores() == 16896);
    const Device& d = h100;                      // a derived object binds to a base reference
    assert(d.memory_gb == 80);
}
// --8<-- [end:inheritance]

// --8<-- [start:virtual]
// Runtime polymorphism: call through a base pointer, run the derived version. The object
// carries a hidden pointer to a table of its class's virtual functions (the vtable), and
// each call is an indirect jump through it.
class BoundaryCondition {                    // abstract: has a pure virtual function
public:
    virtual ~BoundaryCondition() = default;  // virtual destructor: delete via base is safe
    virtual float ghost(float inner) const = 0;   // = 0: derived classes must implement it
    virtual std::string name() const { return "boundary"; }
};

class Dirichlet : public BoundaryCondition {
public:
    explicit Dirichlet(float value) : value_(value) {}
    float ghost(float inner) const override { return 2.0f * value_ - inner; }
    std::string name() const override { return "Dirichlet"; }
private:
    float value_;
};

class Neumann : public BoundaryCondition {   // zero gradient
public:
    float ghost(float inner) const override { return inner; }
    std::string name() const override { return "Neumann"; }
};

void polymorphism() {
    std::vector<std::unique_ptr<BoundaryCondition>> walls;
    walls.push_back(std::make_unique<Dirichlet>(1.0f));
    walls.push_back(std::make_unique<Neumann>());
    assert(walls[0]->ghost(0.5f) == 1.5f);   // dispatched to Dirichlet::ghost
    assert(walls[1]->ghost(0.5f) == 0.5f);   // dispatched to Neumann::ghost
    assert(walls[1]->name() == "Neumann");
    // BoundaryCondition b;                  // error: abstract classes can't be instantiated
    //
    // In GPU kernels, avoid this pattern: virtual calls work in device code only for
    // objects created on the device, they block inlining, and divergent targets serialize
    // a warp. Use templates (static polymorphism) or a switch on a small enum instead.
}
// --8<-- [end:virtual]

// --8<-- [start:virtual_inheritance]
// The "diamond": two bases share a base. Without `virtual`, the derived class would hold
// two copies of the shared base; with it, exactly one.
struct Resource {
    int id = 0;
};
struct Readable : virtual Resource {};
struct Writable : virtual Resource {};
struct File : Readable, Writable {};

void virtual_inheritance() {
    File f;
    f.id = 7;                                // unambiguous: only one Resource inside
    assert(static_cast<Readable&>(f).id == 7 && static_cast<Writable&>(f).id == 7);
}
// --8<-- [end:virtual_inheritance]

// --8<-- [start:class_templates]
// A class template: one definition, a separate class per set of parameters. Fixed-size
// small vectors like this are what CUDA kernels keep in registers: N is known at compile
// time, so loops over it unroll and the array never touches memory.
template <typename T, int N>
class SmallVec {
public:
    T& operator[](int i) { return v_[i]; }
    const T& operator[](int i) const { return v_[i]; }
    static constexpr int size() { return N; }
    T dot(const SmallVec& o) const {
        T s{};
        for (int i = 0; i < N; ++i) s += v_[i] * o.v_[i];
        return s;
    }
private:
    std::array<T, N> v_{};
};

// Class template argument deduction (CTAD) works for constructors too.
template <typename T>
struct Pair {
    T first, second;
};
template <typename T> Pair(T, T) -> Pair<T>;

void class_templates() {
    SmallVec<float, 4> a, b;
    for (int i = 0; i < 4; ++i) { a[i] = float(i); b[i] = 1.0f; }
    assert(a.dot(b) == 6.0f);
    static_assert(SmallVec<double, 8>::size() == 8);
    static_assert(sizeof(SmallVec<float, 4>) == 16);     // no overhead over float[4]
    Pair p{2, 3};                                        // Pair<int> deduced
    assert(p.first + p.second == 5);
}
// --8<-- [end:class_templates]

// --8<-- [start:comparisons]
// C++20: one defaulted <=> gives all six comparisons, member by member, in order.
struct Version {
    int major, minor;
    auto operator<=>(const Version&) const = default;
    bool operator==(const Version&) const = default;
};

void default_comparisons() {
    Version cuda11{11, 8}, cuda12{12, 0}, cuda12_6{12, 6};
    assert(cuda11 < cuda12 && cuda12 < cuda12_6 && cuda12_6 >= cuda12);
    assert(cuda12 != cuda12_6);
    std::vector<Version> vs{cuda12_6, cuda11, cuda12};
    std::sort(vs.begin(), vs.end());                     // works: < is defined
    assert(vs.front() == cuda11);
}
// --8<-- [end:comparisons]

// --8<-- [start:functors]
// A function object (functor): a class with operator(). Unlike a function pointer, its
// type names the operation, so a template taking it can inline the call. This is how
// CUB and Thrust pass reduction operators, and how a CUDA kernel template takes "what to
// do per element".
struct Saxpy {
    float a;
    float operator()(float x, float y) const { return a * x + y; }
};

template <typename Op>
void elementwise(const std::vector<float>& x, std::vector<float>& y, Op op) {
    for (std::size_t i = 0; i < x.size(); ++i) y[i] = op(x[i], y[i]);   // inlined
}

void function_objects() {
    std::vector<float> x(4, 1.0f), y(4, 2.0f);
    elementwise(x, y, Saxpy{3.0f});
    assert(y[0] == 5.0f);
    elementwise(x, y, std::plus<>());                    // standard functors exist too
    assert(y[0] == 6.0f);
    std::function<float(float, float)> erased = Saxpy{2.0f};   // type-erased: flexible, slower
    assert(erased(1.0f, 1.0f) == 3.0f);
}
// --8<-- [end:functors]

// --8<-- [start:lambdas]
// A lambda is a functor the compiler writes for you. The capture list says which outer
// variables it copies [=] or references [&].
void lambdas() {
    float a = 3.0f;
    std::vector<float> x(4, 1.0f), y(4, 2.0f);
    elementwise(x, y, [a](float xi, float yi) { return a * xi + yi; });   // captures a by value
    assert(y[0] == 5.0f);

    int calls = 0;
    auto counted = [&calls](float v) { ++calls; return v * 2.0f; };      // by reference
    std::transform(x.begin(), x.end(), x.begin(), counted);
    assert(calls == 4 && x[0] == 2.0f);

    auto generic = [](auto u, auto v) { return u < v ? u : v; };         // a template lambda
    assert(generic(3, 5) == 3 && generic(2.5, 1.5) == 1.5);

    auto clamp01 = [](float v) constexpr { return v < 0 ? 0.0f : (v > 1 ? 1.0f : v); };
    static_assert(clamp01(2.0f) == 1.0f);                 // usable at compile time
    // In CUDA, the same lambda runs on the GPU when marked __device__ (and compiled with
    // nvcc --extended-lambda): see the CUDA page of this course.
}
// --8<-- [end:lambdas]

int main() {
    inheritance();
    polymorphism();
    virtual_inheritance();
    class_templates();
    default_comparisons();
    function_objects();
    lambdas();
    std::cout << "06_polymorphism: all checks passed\n";
}
