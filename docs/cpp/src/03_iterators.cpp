// C++ for GPU programmers, part 3: iterators and algorithms.
// Topics 11-12: iterators and std::sort (plus the other algorithms GPU code mirrors).
//
//   g++ -std=c++20 -Wall -Wextra -Werror -fsanitize=address,undefined 03_iterators.cpp && ./a.out
#include <algorithm>
#include <array>
#include <cassert>
#include <functional>
#include <iostream>
#include <iterator>
#include <numeric>
#include <vector>

// --8<-- [start:iterators]
// An iterator is a generalized pointer: *it reads, ++it advances, and begin()/end() mark a
// half-open range [begin, end). Algorithms take ranges, so they work on any container.
template <typename It>
int count_negative(It first, It last) {
    int n = 0;
    for (It it = first; it != last; ++it)
        if (*it < 0) ++n;
    return n;
}

void iterators() {
    std::vector<float> v{3.0f, -1.0f, 2.0f, -5.0f};
    std::array<int, 3> a{-1, -2, 7};
    float raw[3] = {-1.0f, 0.0f, 1.0f};

    assert(count_negative(v.begin(), v.end()) == 2);
    assert(count_negative(a.begin(), a.end()) == 2);
    assert(count_negative(raw, raw + 3) == 1);           // a pointer is an iterator

    auto it = std::find(v.begin(), v.end(), 2.0f);
    assert(it != v.end() && std::distance(v.begin(), it) == 2);

    // Random-access iterators support arithmetic, like pointers: it + k, it[k], it2 - it1.
    auto mid = v.begin() + v.size() / 2;
    assert(*mid == 2.0f);

    // Reverse iterators walk backwards.
    std::vector<float> back(v.rbegin(), v.rend());
    assert(back.front() == -5.0f);
}
// --8<-- [end:iterators]

// --8<-- [start:sort]
struct Particle {
    int cell;           // which grid cell the particle is in
    float mass;
};

void sorting() {
    std::vector<int> v{5, 3, 9, 1, 3};
    std::sort(v.begin(), v.end());                       // ascending, O(n log n)
    assert((v == std::vector<int>{1, 3, 3, 5, 9}));

    std::sort(v.begin(), v.end(), std::greater<>());     // a comparator changes the order
    assert(v.front() == 9);

    // Sort structs by a key with a lambda. Sorting particles by cell is how GPU codes make
    // neighbour lookups and scatter-adds coalesced (counting sort or radix sort there).
    std::vector<Particle> p{{2, 1.0f}, {0, 2.0f}, {1, 3.0f}, {0, 4.0f}};
    std::stable_sort(p.begin(), p.end(),
                     [](const Particle& a, const Particle& b) { return a.cell < b.cell; });
    assert(p[0].cell == 0 && p[0].mass == 2.0f && p[1].mass == 4.0f);   // stable: order kept

    // The algorithms GPU libraries provide in parallel (Thrust, CUB) start life here:
    std::vector<int> x(8);
    std::iota(x.begin(), x.end(), 1);                    // 1, 2, ..., 8
    assert(std::accumulate(x.begin(), x.end(), 0) == 36);              // reduce
    std::vector<int> scan(8);
    std::inclusive_scan(x.begin(), x.end(), scan.begin());              // prefix sum
    assert(scan.back() == 36 && scan[3] == 10);
    std::vector<int> evens;
    std::copy_if(x.begin(), x.end(), std::back_inserter(evens),
                 [](int i) { return i % 2 == 0; });                     // stream compaction
    assert(evens.size() == 4);
    std::transform(x.begin(), x.end(), x.begin(), [](int i) { return i * i; });   // map
    assert(x[7] == 64);
    auto [lo, hi] = std::minmax_element(x.begin(), x.end());
    assert(*lo == 1 && *hi == 64);
    // Binary search needs sorted input: lower_bound finds the first element >= value.
    std::vector<int> offsets{0, 3, 3, 7, 10};            // CSR row pointers
    auto row = std::upper_bound(offsets.begin(), offsets.end(), 5) - offsets.begin() - 1;
    assert(row == 2);                                    // element 5 belongs to row 2
}
// --8<-- [end:sort]

int main() {
    iterators();
    sorting();
    std::cout << "03_iterators: all checks passed\n";
}
