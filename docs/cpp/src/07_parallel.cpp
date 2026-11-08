// C++ for GPU programmers, part 7: parallel programming on the CPU.
// Topics 39-42: parallel STL algorithms, threads, std::mutex, std::atomic.
//
//   g++ -std=c++20 -O2 -Wall -Wextra -Werror -pthread -fsanitize=thread 07_parallel.cpp && ./a.out
#include <algorithm>
#include <atomic>
#include <cassert>
#include <execution>
#include <iostream>
#include <mutex>
#include <numeric>
#include <thread>
#include <vector>

// --8<-- [start:parallel_stl]
// Execution policies ask the library to run an algorithm in parallel (par) and/or with
// vector instructions (par_unseq). With GCC, the parallel backend is Intel TBB: link -ltbb,
// otherwise the calls still compile and run sequentially. nvc++ -stdpar runs the same code
// on an NVIDIA GPU.
void parallel_stl() {
    std::vector<double> v(1 << 20);
    std::iota(v.begin(), v.end(), 0.0);
    double s = std::reduce(std::execution::par, v.begin(), v.end(), 0.0);  // order unspecified
    assert(s == double(v.size()) * (v.size() - 1) / 2);
    std::transform(std::execution::par_unseq, v.begin(), v.end(), v.begin(),
                   [](double x) { return 2.0 * x; });
    assert(v[10] == 20.0);
    // std::reduce (unlike std::accumulate) may add in any order: for floats the result can
    // change between runs and machines. The same trade-off as GPU atomics.
}
// --8<-- [end:parallel_stl]

// --8<-- [start:threads]
// std::thread runs a function on a new OS thread; join() waits for it. std::jthread
// (C++20) joins automatically in its destructor. One host thread per GPU is a common way
// to drive several GPUs from one process.
void threads() {
    const int T = 4;
    const int n = 1000;
    std::vector<long> partial(T, 0);
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < T; ++t)
            pool.emplace_back([t, &partial] {
                for (int i = t; i < n; i += T) partial[t] += i;   // each writes its own slot
            });
    }                                        // jthreads join here
    assert(std::accumulate(partial.begin(), partial.end(), 0L) == 499500);
    // Writing neighbouring slots from different threads is correct but slow: they share a
    // cache line ("false sharing"). Pad per-thread data to 64 bytes in hot loops.
}
// --8<-- [end:threads]

// --8<-- [start:mutex]
// A data race (two threads, same memory, at least one writing, no synchronization) is
// undefined behaviour. A mutex makes a block of code run one thread at a time.
void mutexes() {
    std::mutex m;
    std::vector<int> log;
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < 4; ++t)
            pool.emplace_back([&, t] {
                for (int i = 0; i < 100; ++i) {
                    std::lock_guard<std::mutex> lock(m);     // locked until end of scope
                    log.push_back(t);                        // vector isn't thread-safe
                }
            });
    }
    assert(log.size() == 400);
    // Keep critical sections short: everything inside runs serially. The GPU analogue is
    // a lock built from atomicCAS, which is rarely a good idea there: prefer atomics or
    // a per-thread/per-block result plus a reduction.
}
// --8<-- [end:mutex]

// --8<-- [start:atomic]
// std::atomic: indivisible read-modify-write on one variable, no lock. The CPU twin of
// CUDA's atomicAdd (and of cuda::atomic in libcu++).
void atomics() {
    std::atomic<long> counter{0};
    std::atomic<int> max_seen{0};
    {
        std::vector<std::jthread> pool;
        for (int t = 0; t < 4; ++t)
            pool.emplace_back([&, t] {
                for (int i = 0; i < 10000; ++i) {
                    counter.fetch_add(1, std::memory_order_relaxed);   // just a count: relaxed
                    int v = t * 10000 + i, cur = max_seen.load();
                    while (v > cur && !max_seen.compare_exchange_weak(cur, v)) {}   // CAS loop
                }
            });
    }
    assert(counter.load() == 40000);
    assert(max_seen.load() == 39999);
    // compare_exchange is the universal building block: atomicMax for floats on a GPU is
    // written with exactly this loop around atomicCAS.
}
// --8<-- [end:atomic]

int main() {
    parallel_stl();
    threads();
    mutexes();
    atomics();
    std::cout << "07_parallel: all checks passed\n";
}
