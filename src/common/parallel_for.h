#pragma once
// f(i) for every i in [0, n), each index once, on up to `threads` threads (0: half the hardware threads, one a core
// on a machine with two a core). The calls write disjoint outputs, so the result does not depend on the thread count.
// The first exception is rethrown once every thread has stopped.
#include <algorithm>
#include <atomic>
#include <cstddef>
#include <exception>
#include <mutex>
#include <thread>
#include <vector>

namespace common {

template <class F>
void parallel_for(size_t n, unsigned threads, F&& f) {
  if (threads == 0) threads = std::max(1u, std::thread::hardware_concurrency() / 2);
  threads = unsigned(std::min<size_t>(threads, n));
  std::atomic<size_t> next{0};
  std::exception_ptr err;
  std::mutex err_mu;
  const auto run = [&] {
    for (size_t i; (i = next++) < n;) {
      try {
        f(i);
      } catch (...) {
        const std::lock_guard<std::mutex> g(err_mu);
        if (!err) err = std::current_exception();
        next = n;
      }
    }
  };
  std::vector<std::thread> pool;
  for (unsigned t = 1; t < threads; ++t) pool.emplace_back(run);
  run();
  for (std::thread& t : pool) t.join();
  if (err) std::rethrow_exception(err);
}

}  // namespace common
