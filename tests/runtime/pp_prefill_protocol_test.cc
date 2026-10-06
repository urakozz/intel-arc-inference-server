// Spec 16c: the two-card prefill pipeline's order (runtime/pipeline_prefill_plan.h) run by
// its ONE executor (pp_prefill_run - the engine's too) over two host threads standing in
// for the two devices: each thread drains an in-order queue (an immediate command list),
// events are signalled by one "device" and waited on by the other (unbounded, as a list
// waits) or by the host (bounded). Per-chunk resources are plain memory indexed by chunk % 2
// - each device's Control, device 0's ids, the landing slot, the shadows - and under
// ThreadSanitizer (tools/mac/pp_prefill_tsan.sh) the shipped order is data-race free: every
// one of them is handed between the host and the "devices" through an event or a queue.
// That a resource is not REUSED too early is a different property - a slot overwritten
// after device 1 waited for it is still ordered by the event - and the stamps, the per-chunk
// Control check and the reference states catch it (shown below with an order that drops the
// waits: it corrupts, and the test sees it).
//
// What it proves: the rows of every chunk reach device 1 intact and in order, in both
// hand-offs (copy: plain words; peer: runtime/pipeline_protocol.h's send / recv per slot, the
// sequence numbers advancing per slot over 101 chunks); device 0 never starts chunk j before
// device 1 has finished chunk j - 2 (the back-pressure rule), measured with a slowed device
// 1; spec 7's hooks run in order, after both devices finished their chunk, read the shadows
// while the next chunk runs, and a throwing hook leaves the state at its end; a lost hand-off
// is a bounded throw, the waiting device is released and a fresh run works; and (without
// ThreadSanitizer) an order that drops the waits corrupts a slot, which the stamps catch -
// the test can see what it guards against. What it does not: Level Zero's events on two
// cards, the copy engine, PCIe (the box: queue row 23).
#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdio>
#include <deque>
#include <functional>
#include <memory>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

#include "check.h"
#include "runtime/pipeline_prefill_plan.h"
#include "runtime/pipeline_protocol.h"

#if defined(__has_feature)
#if __has_feature(thread_sanitizer)
#define B70_TSAN 1
#endif
#endif
#if defined(__SANITIZE_THREAD__)
#define B70_TSAN 1
#endif

namespace {
using runtime::PpChunk;
using Clock = std::chrono::steady_clock;
namespace proto = runtime::pp_protocol;

constexpr uint32_t kWords = 64;   // a chunk's "rows": kWords words per chunk
constexpr uint32_t kSums = 8;

struct Timeout : std::runtime_error {
  using std::runtime_error::runtime_error;
};

// An event: a device signals it, a device waits on it without a bound (a list's wait), the
// host waits with one, the host resets it before reuse and host-signals it to release.
class Event {
 public:
  void signal() {
    std::lock_guard<std::mutex> l(m_);
    on_ = true;
    cv_.notify_all();
  }
  void reset() {
    std::lock_guard<std::mutex> l(m_);
    on_ = false;
  }
  void wait() {
    std::unique_lock<std::mutex> l(m_);
    cv_.wait(l, [&] { return on_; });
  }
  bool wait_for(std::chrono::milliseconds t) {
    std::unique_lock<std::mutex> l(m_);
    return cv_.wait_for(l, t, [&] { return on_; });
  }

 private:
  std::mutex m_;
  std::condition_variable cv_;
  bool on_ = false;
};

// One device: an in-order queue drained by its own thread.
class Device {
 public:
  Device() : t_([this] { loop(); }) {}
  ~Device() {
    {
      std::lock_guard<std::mutex> l(m_);
      stop_ = true;
      cv_.notify_all();
    }
    t_.join();
  }
  void append(std::function<void()> f) {
    std::lock_guard<std::mutex> l(m_);
    q_.push_back(std::move(f));
    cv_.notify_all();
  }
  // Bounded: true when everything appended has run.
  bool idle_within(std::chrono::milliseconds t) {
    std::unique_lock<std::mutex> l(m_);
    return idle_cv_.wait_for(l, t, [&] { return q_.empty() && !busy_; });
  }

 private:
  void loop() {
    for (;;) {
      std::function<void()> f;
      {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [&] { return stop_ || !q_.empty(); });
        if (q_.empty()) return;
        f = std::move(q_.front());
        q_.pop_front();
        busy_ = true;
      }
      f();
      std::lock_guard<std::mutex> l(m_);
      busy_ = false;
      if (q_.empty()) idle_cv_.notify_all();
    }
  }
  std::mutex m_;
  std::condition_variable cv_, idle_cv_;
  std::deque<std::function<void()>> q_;
  bool stop_ = false, busy_ = false;
  std::thread t_;
};

uint64_t mix(uint64_t h, uint64_t v) { return (h ^ v) * 0x100000001b3ull + 0x9e3779b97f4a7c15ull; }

// What one card computes: device 0's state after each chunk, the rows it hands off, device
// 1's state after each chunk.
struct Reference {
  std::vector<uint64_t> s0, s1;
  std::vector<std::vector<uint32_t>> rows;
};
std::vector<uint32_t> rows_of(uint64_t s0, uint32_t j) {
  std::vector<uint32_t> r(kWords + kSums);
  for (uint32_t i = 0; i < r.size(); ++i) r[i] = uint32_t(mix(s0, i * 31 + j) >> 13);
  return r;
}
Reference reference(const std::vector<PpChunk>& chunks, const std::vector<uint32_t>& ids) {
  Reference r;
  uint64_t a = 1, b = 2;
  for (uint32_t j = 0; j < chunks.size(); ++j) {
    for (uint32_t p = chunks[j].pos; p < chunks[j].end(); ++p) a = mix(a, ids[p] + uint64_t(p) * 7);
    r.s0.push_back(a);
    r.rows.push_back(rows_of(a, j));
    for (uint32_t w : r.rows.back()) b = mix(b, w);
    r.s1.push_back(b);
  }
  return r;
}

struct Options {
  bool peer = false;
  int jitter_us = 0;            // random per-chunk work on each device
  int slow1_us = 0;             // device 1's extra work per chunk (back-pressure)
  int64_t drop = -1;            // device 0 never hands off this chunk
  int64_t throw_hook = -1;      // the hook at this chunk throws
  std::chrono::milliseconds bound{2000};
};

// The test's driver: pp_prefill_run's interface over the two thread devices.
class Driver {
 public:
  Driver(const std::vector<PpChunk>& chunks, const std::vector<uint32_t>& ids, const Reference& ref,
         const Options& o)
      : ids_(ids), ref_(ref), o_(o) {
    (void)chunks;
    for (auto& l : land_) l = std::make_unique<proto::Landing>(kWords, kSums);
  }

  // Run0 / Run1. Host-side writes go to resources of parity j % 2 only.
  void run(uint32_t dev, const PpChunk& c, uint32_t j) {
    const uint32_t p = j % runtime::kPpPfDepth;
    if (dev == 0) {
      ctl_[0][p] = c;
      ids0_[p].assign(ids_.begin() + c.pos, ids_.begin() + c.end());
      ready_[p].reset();
      done_[0][p].reset();
      ++runs0_;
      dev0_.append([this, c, j, p] { device0(c, j, p); });
    } else {
      ctl_[1][p] = c;
      done_[1][p].reset();
      dev1_.append([this, c, j, p] { device1(c, j, p); });
    }
  }
  void wait(uint32_t dev, uint32_t j) {
    const uint32_t p = j % runtime::kPpPfDepth;
    const Clock::time_point t0 = Clock::now();
    if (!done_[dev][p].wait_for(o_.bound)) {
      release();
      throw Timeout("device " + std::to_string(dev) + "'s chunk " + std::to_string(j) +
                    " did not finish within the bound");
    }
    if (dev == 1) waited1_ms_ += std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
  }
  void hook(const PpChunk& c, uint32_t j) {
    const uint32_t p = j % runtime::kPpPfDepth;
    hooks_.push_back(c.end());
    // Both devices finished chunk j: the shadows hold its end (the live state may be past it).
    CHECK_EQ(shadow_[0][p], ref_.s0[j]);
    CHECK_EQ(shadow_[1][p], ref_.s1[j]);
    if (int64_t(j) == o_.throw_hook) {
      // The engine's rule: drain, copy the shadows back, rethrow - the state at the hook's end.
      if (!drain_within()) throw Timeout("drain after a throwing hook");
      live_[0] = shadow_[0][p];
      live_[1] = shadow_[1][p];
      throw std::runtime_error("hook");
    }
  }
  void head(const PpChunk&, uint32_t) {
    dev1_.append([this] { token_ = uint32_t(live_[1] % 248320); });
  }
  void drain() {
    if (!drain_within()) {
      release();
      throw Timeout("drain");
    }
  }
  // After a failure: release device 1 from a hand-off that will not come, then let both drain.
  void release() {
    for (Event& e : ready_) e.signal();
    drain_within();
  }

  uint64_t live(uint32_t d) const { return live_[d]; }
  uint32_t token() const { return token_; }
  uint32_t violations() const { return violations_.load(); }
  uint32_t corrupt() const { return corrupt_.load(); }
  uint32_t max_ahead() const { return max_ahead_.load(); }
  const std::vector<uint32_t>& hooks() const { return hooks_; }
  double waited1_ms() const { return waited1_ms_; }
  uint32_t seq(uint32_t slot) const { return send_seq_[slot]; }

 private:
  bool drain_within() { return dev0_.idle_within(o_.bound) && dev1_.idle_within(o_.bound); }
  void work(int us) {
    if (us > 0) std::this_thread::sleep_for(std::chrono::microseconds(us));
  }
  int jitter() {
    if (o_.jitter_us == 0) return 0;
    std::lock_guard<std::mutex> l(rng_m_);
    return int(rng_() % uint32_t(o_.jitter_us));
  }

  void device0(PpChunk c, uint32_t j, uint32_t p) {
    // The back-pressure rule as device 0 sees it: device 1 has finished chunk j - 2.
    const uint32_t done1 = finished1_.load();
    const uint32_t ahead = j + 1 - std::min(j + 1, done1);
    if (j >= 2 && done1 < j - 1) ++violations_;
    max_ahead_.store(std::max(max_ahead_.load(), ahead));
    // The Control and the ids are this chunk's (the host wrote them before appending).
    if (ctl_[0][p].pos != c.pos || ctl_[0][p].rows != c.rows) ++corrupt_;
    for (uint32_t i = 0; i < c.rows; ++i) live_[0] = mix(live_[0], ids0_[p][i] + uint64_t(c.pos + i) * 7);
    work(jitter());
    const std::vector<uint32_t> rows = rows_of(live_[0], j);
    if (c.hook) shadow_[0][p] = live_[0];
    if (int64_t(j) != o_.drop) {
      if (o_.peer) {
        proto::send(rows.data(), rows.data() + kWords, *land_[p], send_seq_[p]);
      } else {
        slot_[p].stamp = j;
        std::copy(rows.begin(), rows.end(), slot_[p].words.begin());
      }
      ready_[p].signal();
    }
    finished0_.fetch_add(1);
    done_[0][p].signal();
  }

  void device1(PpChunk c, uint32_t j, uint32_t p) {
    work(o_.slow1_us);
    ready_[p].wait();   // a list's wait: no bound - the host's wait is the bound
    std::vector<uint32_t> rows(kWords + kSums);
    if (o_.peer) {
      const uint32_t st = proto::recv(*land_[p], recv_state_[p], rows.data(), rows.data() + kWords, 1u << 20);
      if (st != proto::kOk) ++corrupt_;
    } else {
      if (slot_[p].stamp != j) ++corrupt_;
      std::copy(slot_[p].words.begin(), slot_[p].words.end(), rows.begin());
    }
    if (int64_t(j) != o_.drop && rows != ref_.rows[j]) ++corrupt_;
    if (ctl_[1][p].pos != c.pos || ctl_[1][p].rows != c.rows) ++corrupt_;
    work(jitter());
    for (uint32_t w : rows) live_[1] = mix(live_[1], w);
    if (c.hook) shadow_[1][p] = live_[1];
    finished1_.fetch_add(1);
    done_[1][p].signal();
  }

  const std::vector<uint32_t>& ids_;
  const Reference& ref_;
  Options o_;
  // Per parity: each device's Control, device 0's ids, the landing slot, the shadows.
  PpChunk ctl_[2][runtime::kPpPfDepth];
  std::vector<uint32_t> ids0_[runtime::kPpPfDepth];
  struct Slot {   // fixed size: a slot is overwritten in place, never reallocated
    uint32_t stamp = ~0u;
    std::array<uint32_t, kWords + kSums> words{};
  } slot_[runtime::kPpPfDepth];
  std::unique_ptr<proto::Landing> land_[runtime::kPpPfDepth];
  uint32_t send_seq_[runtime::kPpPfDepth] = {0, 0};                 // device 0's, per slot
  uint32_t recv_state_[runtime::kPpPfDepth][proto::kStateWords] = {};   // device 1's, per slot
  uint64_t shadow_[2][runtime::kPpPfDepth] = {};
  uint64_t live_[2] = {1, 2};   // each device's "persistent state"
  uint32_t token_ = 0;
  Event ready_[runtime::kPpPfDepth], done_[2][runtime::kPpPfDepth];
  std::atomic<uint32_t> finished0_{0}, finished1_{0}, violations_{0}, corrupt_{0}, max_ahead_{0};
  uint32_t runs0_ = 0;
  std::vector<uint32_t> hooks_;
  double waited1_ms_ = 0;
  std::mutex rng_m_;
  std::mt19937 rng_{1234};
  // Declared last: the device threads stop (and join) before anything they touch goes.
  Device dev0_, dev1_;
};

std::vector<uint32_t> prompt(size_t n) {
  std::vector<uint32_t> v(n);
  for (size_t i = 0; i < n; ++i) v[i] = uint32_t((i * 2654435761u) % 248320u);
  return v;
}

struct Run {
  uint64_t s0, s1;
  uint32_t token, violations, corrupt, max_ahead;
  std::vector<uint32_t> hooks;
  double waited1_ms;
};

Run run_case(const char* what, size_t n_ids, uint32_t chunk, bool hooked, const Options& o,
             const std::vector<runtime::PpPfStep>* steps = nullptr) {
  const std::vector<uint32_t> ids = prompt(n_ids);
  const std::vector<PpChunk> chunks = runtime::pp_prefill_chunks(0, n_ids, chunk, hooked, 64);
  const Reference ref = reference(chunks, ids);
  Driver d(chunks, ids, ref, o);
  const std::vector<runtime::PpPfStep> s = steps ? *steps : runtime::pp_prefill_schedule(chunks);
  runtime::pp_prefill_run(chunks, s, d);
  Run r{d.live(0), d.live(1), d.token(), d.violations(), d.corrupt(), d.max_ahead(), d.hooks(),
        d.waited1_ms()};
  if (!steps) {
    CHECK_EQ(r.s0, ref.s0.back());
    CHECK_EQ(r.s1, ref.s1.back());
    CHECK_EQ(r.token, uint32_t(ref.s1.back() % 248320));
    CHECK_EQ(r.corrupt, 0u);
    CHECK_EQ(r.violations, 0u);
    CHECK(r.max_ahead <= 2);
  }
  std::printf("%-44s %3zu chunks: %s, device 0 at most %u chunks ahead, %zu hooks, host waited"
              " %.1f ms on device 1\n",
              what, chunks.size(), steps ? "(a mutated order: not graded here)" : "states and token = one card",
              r.max_ahead, r.hooks.size(), r.waited1_ms);
  return r;
}
}  // namespace

int main() {
  Options base;
  base.jitter_us = 300;
  // Copy and peer over many chunks; the peer path's per-slot sequence numbers advance 51 / 50.
  run_case("copy, jitter", 64 * 37 + 5, 64, false, base);
  Options peer = base;
  peer.peer = true;
  run_case("peer, jitter (per-slot sequence numbers)", 64 * 101, 64, false, peer);
  // One chunk, two chunks: the schedule's edges.
  run_case("copy, one chunk", 10, 64, false, base);
  run_case("copy, two chunks", 100, 64, false, base);
  // Back-pressure: device 1 slowed; device 0 waits, never more than two chunks ahead.
  Options slow = base;
  slow.slow1_us = 3000;
  const Run s = run_case("copy, device 1 slowed 3 ms a chunk", 64 * 24, 64, false, slow);
  CHECK(s.waited1_ms > 24 * 2.0);   // the host really waited on device 1
  CHECK_EQ(s.max_ahead, 2u);        // and device 0 really ran ahead by the two slots
  Options slow_peer = slow;
  slow_peer.peer = true;
  run_case("peer, device 1 slowed", 64 * 24, 64, false, slow_peer);
  // Spec 7: a hook at every block end (block 64 here, chunk 64: every chunk), in order, each
  // from the shadows while the next chunk runs.
  const Run h = run_case("copy, hooks at every block end", 64 * 20 + 7, 64, true, base);
  CHECK_EQ(h.hooks.size(), size_t{20});
  for (uint32_t i = 0; i < h.hooks.size(); ++i) CHECK_EQ(h.hooks[i], 64 * (i + 1));
  run_case("copy, hooks, device 1 slowed", 64 * 12 + 1, 64, true, slow);

  // A throwing hook at chunk 2's end: the state is chunk 2's end on both devices.
  {
    const std::vector<uint32_t> ids = prompt(64 * 8);
    const std::vector<PpChunk> chunks = runtime::pp_prefill_chunks(0, ids.size(), 64, true, 64);
    const Reference ref = reference(chunks, ids);
    Options o = base;
    o.throw_hook = 2;
    Driver d(chunks, ids, ref, o);
    bool threw = false;
    try {
      runtime::pp_prefill_run(chunks, runtime::pp_prefill_schedule(chunks), d);
    } catch (const std::runtime_error& e) {
      threw = std::string(e.what()) == "hook";
    }
    CHECK(threw);
    CHECK_EQ(d.live(0), ref.s0[2]);
    CHECK_EQ(d.live(1), ref.s1[2]);
    std::printf("a throwing hook at chunk 2: both devices' state is chunk 2's end\n");
  }

  // A lost hand-off: device 0 never hands chunk 3 over. Device 1 waits without a bound (a
  // list's wait); the host's wait gives up within its bound, releases device 1, and throws.
  for (bool peer_mode : {false, true}) {
    const std::vector<uint32_t> ids = prompt(64 * 10);
    const std::vector<PpChunk> chunks = runtime::pp_prefill_chunks(0, ids.size(), 64, false, 64);
    const Reference ref = reference(chunks, ids);
    Options o = base;
    o.peer = peer_mode;
    o.drop = 3;
    o.bound = std::chrono::milliseconds(300);
    const Clock::time_point t0 = Clock::now();
    std::string err;
    {
      Driver d(chunks, ids, ref, o);
      try {
        runtime::pp_prefill_run(chunks, runtime::pp_prefill_schedule(chunks), d);
      } catch (const Timeout& e) {
        err = e.what();
      }
    }   // the driver (and its threads) go: nothing is left waiting
    const double s = std::chrono::duration<double>(Clock::now() - t0).count();
    CHECK(err.find("device 1's chunk 3") != std::string::npos);
    CHECK(s < 3.0);
    // A fresh run (the engine's reset()) is whole again.
    run_case(peer_mode ? "peer, after a lost hand-off" : "copy, after a lost hand-off", ids.size(),
             64, false, base);
    std::printf("%s: a lost hand-off threw after %.2f s: %s\n", peer_mode ? "peer" : "copy", s, err.c_str());
  }

#ifndef B70_TSAN
  // The test sees what it guards against: an order without the waits lets device 0 run ahead
  // of a slowed device 1 and overwrite a slot it has not read (the stamps catch it). Not
  // under ThreadSanitizer: without the waits the host also rewrites a Control a device may
  // still be reading - a real race, which is the point, but not one to hand the sanitizer.
  {
    const std::vector<PpChunk> chunks = runtime::pp_prefill_chunks(0, 64 * 12, 64, false, 64);
    std::vector<runtime::PpPfStep> broken;
    for (uint32_t j = 0; j < chunks.size(); ++j) broken.push_back({runtime::PpPfOp::Run0, j});
    for (uint32_t j = 0; j < chunks.size(); ++j) broken.push_back({runtime::PpPfOp::Run1, j});
    broken.push_back({runtime::PpPfOp::Head, uint32_t(chunks.size() - 1)});
    broken.push_back({runtime::PpPfOp::Drain, uint32_t(chunks.size())});
    CHECK(!runtime::pp_prefill_check(broken, chunks).empty());
    Options o = slow;
    o.jitter_us = 0;
    const Run b = run_case("an order without the waits (must corrupt)", 64 * 12, 64, false, o, &broken);
    CHECK(b.corrupt > 0);
    CHECK(b.violations > 0);
    std::printf("  ... and it did: %u corrupt reads, %u back-pressure violations\n", b.corrupt, b.violations);
  }
#endif
  std::printf("pp_prefill_protocol_test: OK\n");
  return 0;
}
