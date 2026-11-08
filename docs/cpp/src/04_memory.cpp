// C++ for GPU programmers, part 4: memory.
// Topics 13-20: std::vector, references, pass by reference, pointers, dynamic allocation,
// std::unique_ptr, std::shared_ptr, std::span.
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 04_memory.cpp && ./a.out
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <memory>
#include <numeric>
#include <span>
#include <vector>

// --8<-- [start:vector]
void vectors() {
    std::vector<float> v(1000, 0.0f);       // 1000 floats, contiguous on the heap
    assert(v.size() == 1000);
    v.push_back(1.0f);                      // grows; may reallocate and move everything
    assert(v.size() == 1001 && v.capacity() >= 1001);

    std::vector<float> w;
    w.reserve(1 << 20);                     // allocate once up front: no reallocations
    float* before = w.data();
    for (int i = 0; i < (1 << 20); ++i) w.push_back(float(i));
    assert(w.data() == before);             // reserve kept the buffer in place

    // .data() is the raw pointer you hand to cudaMemcpy: vector storage is contiguous.
    std::vector<float> copy(w.size());
    std::memcpy(copy.data(), w.data(), w.size() * sizeof(float));
    assert(copy[12345] == 12345.0f);

    // 2-D data: one flat vector indexed row * cols + col (what GPU kernels expect), not
    // vector<vector<float>>, whose rows are separate allocations scattered in memory.
    const int rows = 3, cols = 4;
    std::vector<float> m(rows * cols);
    m[2 * cols + 1] = 7.0f;
    assert(m[9] == 7.0f);
}
// --8<-- [end:vector]

// --8<-- [start:references]
void references() {
    int x = 1;
    int& r = x;                             // r is another name for x: no copy, never null
    r = 5;
    assert(x == 5);

    std::vector<float> big(1 << 20, 1.0f);
    const std::vector<float>& view = big;   // const&: read without copying 4 MB
    assert(&view[0] == &big[0]);

    for (auto& e : big) e *= 2.0f;         // & in range-for: modify the elements
    assert(big[0] == 2.0f);
}
// --8<-- [end:references]

// --8<-- [start:pass_by_ref]
// By value: the callee gets a copy (fine for small things: ints, floats, small structs).
float sum_by_value(std::vector<float> v) { return std::accumulate(v.begin(), v.end(), 0.0f); }
// By const reference: no copy, and the callee promises not to modify. The default for
// anything bigger than a couple of words.
float sum_by_cref(const std::vector<float>& v) { return std::accumulate(v.begin(), v.end(), 0.0f); }
// By non-const reference: an output (or in/out) parameter.
void scale_in_place(std::vector<float>& v, float a) {
    for (auto& e : v) e *= a;
}

void pass_by_ref() {
    std::vector<float> v(1000, 1.0f);
    assert(sum_by_value(v) == 1000.0f);     // copied 4 KB on the way in
    assert(sum_by_cref(v) == 1000.0f);      // copied nothing
    scale_in_place(v, 3.0f);
    assert(v[999] == 3.0f);
    // CUDA kernels take every argument by value. Passing a std::vector to a kernel copies
    // the vector *object* (a host pointer + sizes), never the data: pass .data() of a
    // device buffer instead.
}
// --8<-- [end:pass_by_ref]

// --8<-- [start:pointers]
void pointers() {
    float a[4] = {1, 2, 3, 4};
    float* p = a;                           // an array decays to a pointer to its first element
    assert(*p == 1.0f && p[2] == 3.0f && *(p + 3) == 4.0f);
    ++p;                                    // pointer arithmetic moves by sizeof(float)
    assert(*p == 2.0f);
    assert(reinterpret_cast<char*>(p) - reinterpret_cast<char*>(a) == 4);

    float* none = nullptr;                  // "points nowhere": check before dereferencing
    assert(none == nullptr);

    const float* read_only = a;             // can't write through it (a kernel input)
    float* const fixed = a;                 // the pointer itself can't move
    fixed[0] = 10.0f;
    assert(read_only[0] == 10.0f);

    // A pointer to a pointer: how cudaMalloc returns the address it allocated.
    float* buffer = nullptr;
    float** out = &buffer;
    *out = a;
    assert(buffer == a);
}
// --8<-- [end:pointers]

// --8<-- [start:dynamic]
// cudaMalloc has the same shape as this function: it writes the new address through
// the pointer-to-pointer it is given.
int fake_malloc(void** ptr, std::size_t bytes) {
    *ptr = std::malloc(bytes);
    return *ptr ? 0 : 1;
}

void dynamic_alloc() {
    float* x = new float[1000];             // heap allocation, uninitialized
    x[999] = 1.0f;
    delete[] x;                             // new[] pairs with delete[]; forgetting it leaks

    float* y = nullptr;
    int err = fake_malloc(reinterpret_cast<void**>(&y), 1000 * sizeof(float));
    assert(err == 0 && y != nullptr);
    y[0] = 2.0f;
    std::free(y);
    // Manual new/delete (and cudaMalloc/cudaFree) leak on every early return or
    // exception between the two calls. The next two types fix that.
}
// --8<-- [end:dynamic]

// --8<-- [start:unique_ptr]
// unique_ptr: exactly one owner; the memory is freed when the owner goes out of scope.
// A custom deleter makes it manage anything with a free function: here std::free, and
// cudaFree in the CUDA page.
struct FreeDeleter {
    void operator()(void* p) const { std::free(p); }
};
using HostBuffer = std::unique_ptr<float[], FreeDeleter>;

HostBuffer make_buffer(std::size_t n) {
    return HostBuffer(static_cast<float*>(std::malloc(n * sizeof(float))));
}

void unique_ptrs() {
    auto a = std::make_unique<float[]>(1000);    // zero-initialized array, freed automatically
    a[10] = 1.0f;
    auto b = std::move(a);                       // ownership moves; a is now empty
    assert(a == nullptr && b[10] == 1.0f);
    // auto c = b;                               // error: a unique_ptr can't be copied

    HostBuffer buf = make_buffer(256);
    buf[0] = 3.0f;
    assert(buf.get() != nullptr);
}   // b and buf are freed here, in reverse order of creation
// --8<-- [end:unique_ptr]

// --8<-- [start:shared_ptr]
// shared_ptr: many owners, a reference count, freed when the last one goes away. It costs
// an atomic increment per copy, so prefer unique_ptr unless ownership is truly shared
// (e.g. a buffer cached by several objects).
struct Mesh {
    std::vector<float> coords;
};

void shared_ptrs() {
    auto mesh = std::make_shared<Mesh>();
    mesh->coords.assign(300, 0.0f);
    assert(mesh.use_count() == 1);
    {
        auto solver_view = mesh;                 // a second owner
        auto output_view = mesh;                 // a third
        assert(mesh.use_count() == 3);
    }
    assert(mesh.use_count() == 1);               // the copies were released
    std::weak_ptr<Mesh> observer = mesh;         // watches without owning
    mesh.reset();
    assert(observer.expired());                  // the Mesh is gone
}
// --8<-- [end:shared_ptr]

// --8<-- [start:span]
// std::span: a non-owning (pointer, size) view. One function accepts a vector, an array
// or a raw buffer, and the size travels with the pointer. It is the shape of a device
// array view: a pointer and a length, cheap to copy, passed by value.
float dot(std::span<const float> a, std::span<const float> b) {
    assert(a.size() == b.size());
    float s = 0.0f;
    for (std::size_t i = 0; i < a.size(); ++i) s += a[i] * b[i];
    return s;
}

void spans() {
    std::vector<float> v(8, 1.0f);
    float raw[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    assert(dot(v, raw) == 8.0f);
    std::span<float> all(v);
    auto tail = all.subspan(4);                  // a view of elements 4..7, no copy
    tail[0] = 5.0f;
    assert(v[4] == 5.0f && tail.size() == 4);
    std::span<float, 4> fixed(raw, 4);           // size in the type: checked at compile time
    static_assert(sizeof(fixed) == sizeof(float*));   // a static extent stores only the pointer
}
// --8<-- [end:span]

int main() {
    vectors();
    references();
    pass_by_ref();
    pointers();
    dynamic_alloc();
    unique_ptrs();
    shared_ptrs();
    spans();
    std::cout << "04_memory: all checks passed\n";
}
