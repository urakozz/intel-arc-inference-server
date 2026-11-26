#include "runtime/qwen4exp/qwen4exp_cpu_experts.h"

#include <immintrin.h>

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <future>
#include <stdexcept>
#include <string>

#include "cpu/cpu_pick.h"
#include "cpu/expert_q4.h"
#include "l0/cmdlist.h"
#include "l0/fence.h"
#include "l0/queue.h"

namespace runtime::qwen4exp {
namespace {
using Clock = std::chrono::steady_clock;
double us_since(Clock::time_point t0) { return std::chrono::duration<double, std::micro>(Clock::now() - t0).count(); }
double median(std::vector<double> v) {
  std::sort(v.begin(), v.end());
  return v[v.size() / 2];
}
constexpr uint32_t kRw = 32, kSlots = 11, kE = 512, kTopK = 10;
constexpr uint32_t kHid = cpu::kQ4Hidden;
constexpr double kDrift = 0.25;   // re-decide when an observed cost moves this far from the one the route was set with
constexpr uint32_t kMinObs = 32;  // hand-offs observed before a re-decision
}  // namespace

bool CpuExperts::has_host_experts(const loader::Q4LoadedModel& m) {
  for (const loader::Q4DevicePart& p : m.parts)
    for (const loader::Q4Layer& L : p.layers)
      if (L.mirror && L.expert_row) return true;
  return false;
}

CpuExperts::CpuExperts(const std::vector<l0::Context*>& ctxs, const loader::Q4LoadedModel& m, Mode mode)
    : mode_(mode), inter_(m.placement.tp ? cpu::kQ4TpInter : cpu::kQ4Inter), share_(m.placement.tp ? 2 : 1) {
  if (m.desc.experts != kE || m.desc.top_k != kTopK || m.desc.hidden != kHid || m.desc.moe_inter != cpu::kQ4Inter)
    throw std::runtime_error("CpuExperts: built for 512 experts, top 10, hidden 2560, inter 640");
  for (uint32_t i = 0; i < m.parts.size(); ++i) {
    const loader::Q4DevicePart& p = m.parts[i];
    l0::Context& ctx = *ctxs.at(i);
    Card c;
    c.first = p.first;
    c.n = p.end - p.first;
    c.has.assign(c.n, 0);
    c.mlo.assign(c.n, 0);
    c.msz.assign(c.n, 0);
    for (uint32_t li = 0; li < c.n; ++li) {
      const loader::Q4Layer& L = p.layers[li];
      if (!L.mirror || !L.expert_row) continue;
      c.has[li] = 1;
      c.host_layers.push_back(li);
      c.mlo[li] = reinterpret_cast<uint64_t>(L.mirror->ptr());
      c.msz[li] = L.mirror->size();
    }
    if (p.expert_table) {
      c.rows.resize(size_t(c.n) * 2 * kE);
      l0::CmdList::immediate(ctx).copy(c.rows.data(), p.expert_table->ptr(), c.rows.size() * 8);
    }
    c.ev = std::make_unique<l0::SyncEventPool>(ctx, std::vector<const l0::Context*>{&ctx}, kVerifyRows * c.n * 2);
    c.x = std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, size_t(kVerifyRows) * kHid * 2);
    c.route = std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, size_t(kVerifyRows) * kRw * 4);
    c.y = std::make_unique<l0::Mem>(ctx, l0::MemKind::Host, size_t(kVerifyRows) * kSlots * kHid * 4);
    std::memset(c.x->ptr(), 0, c.x->size());
    std::memset(c.route->ptr(), 0, c.route->size());
    cards_.push_back(std::move(c));
  }
  dec_ = std::make_unique<l0::Mem>(*ctxs[0], l0::MemKind::Host, size_t(kVerifyRows + 1) * (kMaxN + 1) * 4);
  std::memset(dec_->ptr(), 0, dec_->size());

  // The host thread owns the pool (its workers and itself pinned to the idlest physical cores, never the caller's
  // thread) and measures the CPU's cost before it serves anything.
  std::promise<void> measured;
  std::future<void> done = measured.get_future();
  th_ = std::thread([this, &measured] {
    try {
      cores_ = cpu::physical_cores();
      const char* avx2 = std::getenv("B70_CPU_EXPERTS_AVX2");
      avx2_forced_ = avx2 && !std::strcmp(avx2, "1") && cpu::cpu_has_avx512();
      isa_ = avx2_forced_ ? cpu::Isa::Avx2 : cpu::best_isa();
      pool_ = std::make_unique<cpu::Pool>(cores_, cpu::pick_cpus(100));
      const Card* c = nullptr;
      for (const Card& k : cards_)
        if (!k.host_layers.empty()) c = &k;
      const uint32_t li = c->host_layers[0];
      const uint64_t* row = c->rows.data() + size_t(li) * 2 * kE;
      std::vector<uint32_t> ex;   // host-RAM experts of that layer, a different one each rep (cold in RAM)
      for (uint32_t e = 0; e < kE && ex.size() < 16; ++e)
        if (row[e] - c->mlo[li] < c->msz[li]) ex.push_back(e);
      std::vector<uint16_t> x(size_t(kVerifyRows) * kHid, 0x3C00);
      std::vector<float> y(size_t(kVerifyRows) * kHid);
      const uint16_t* xp[kVerifyRows];
      float* yp[kVerifyRows];
      for (uint32_t r = 0; r < kVerifyRows; ++r) {
        xp[r] = x.data() + size_t(r) * kHid;
        yp[r] = y.data() + size_t(r) * kHid;
      }
      // an expert's cost as the host thread runs it: under tp both cards' halves in one batch (share_ experts at
      // once, the time a half)
      std::vector<float> y2(size_t(kVerifyRows) * kHid);
      const auto time_rows = [&](uint32_t n) {
        std::vector<double> t;
        for (uint32_t rep = 0; rep < 8; ++rep) {
          cpu::Q4Job jobs[2];
          for (uint32_t b = 0; b < share_; ++b) {
            const uint32_t e = ex[(rep * share_ + b) % ex.size()];
            jobs[b].gu = reinterpret_cast<const uint32_t*>(row[e]);
            jobs[b].dn = reinterpret_cast<const uint32_t*>(row[kE + e]);
            jobs[b].n = n;
            for (uint32_t r = 0; r < n; ++r) {
              jobs[b].x[r] = xp[r];
              jobs[b].out[r] = b ? y2.data() + size_t(r) * kHid : yp[r];
            }
          }
          const auto t0 = Clock::now();
          cpu::q4_experts(jobs, share_, pool_.get(), isa_, inter_);
          if (rep > 0) t.push_back(us_since(t0) / share_);
        }
        return median(t);
      };
      cpu_us_[1] = time_rows(1);
      cpu_us_[kVerifyRows] = time_rows(kVerifyRows);
      for (uint32_t M = 2; M < kVerifyRows; ++M)
        cpu_us_[M] = cpu_us_[1] + (cpu_us_[kVerifyRows] - cpu_us_[1]) * (M - 1) / (kVerifyRows - 1);
      measured.set_value();
    } catch (...) {
      measured.set_exception(std::current_exception());
      return;
    }
    loop();
  });
  try {
    done.get();
    measure(ctxs);
  } catch (...) {
    quit_ = true;
    wake_.fetch_add(1);
    wake_.notify_all();
    th_.join();
    throw;
  }
  decide(true);
}

CpuExperts::~CpuExperts() {
  quit_.store(true, std::memory_order_release);
  wake_.fetch_add(1, std::memory_order_release);
  wake_.notify_all();
  if (th_.joinable()) th_.join();
}

// The link and the hand-off, on the first card with a host-RAM layer: one expert's two blocks copied from its mirror
// into a scratch on the card (the card reads a host expert at the link's rate either way), and a list of 1 against 8
// hand-offs on that layer's own events, this thread answering each at once.
void CpuExperts::measure(const std::vector<l0::Context*>& ctxs) {
  uint32_t ci = 0;
  while (cards_[ci].host_layers.empty()) ++ci;
  Card& c = cards_[ci];
  l0::Context& ctx = *ctxs[ci];
  const uint32_t li = c.host_layers[0];
  const uint64_t* row = c.rows.data() + size_t(li) * 2 * kE;
  uint32_t e = 0;
  while (row[e] - c.mlo[li] >= c.msz[li]) ++e;
  const size_t gu = cpu::kQ4Hidden / 64 * (2 * inter_ / 16) * 136 * 4, dn = inter_ / 64 * (cpu::kQ4Hidden / 16) * 136 * 4;
  {
    l0::Mem tmp(ctx, l0::MemKind::Device, gu + dn);
    l0::CmdList im = l0::CmdList::immediate(ctx);
    std::vector<double> t;
    for (int rep = 0; rep < 6; ++rep) {
      const auto t0 = Clock::now();
      im.copy(tmp.ptr(), reinterpret_cast<const void*>(row[e]), gu);
      im.copy(static_cast<uint8_t*>(tmp.ptr()) + gu, reinterpret_cast<const void*>(row[kE + e]), dn);
      if (rep > 0) t.push_back(us_since(t0));
    }
    pcie_us_ = median(t);
  }
  l0::Queue q(ctx);
  l0::Fence f(q);
  l0::Mem dx(ctx, l0::MemKind::Device, size_t(kVerifyRows) * (kHid * 2 + kRw * 4));
  for (uint32_t M : {1u, kVerifyRows}) {
    const uint32_t ready = ev_index(c, M, li, 0), done = ev_index(c, M, li, 1);
    const auto run = [&](uint32_t hand_offs) {
      l0::CmdList l = l0::CmdList::regular(ctx);
      for (uint32_t h = 0; h < hand_offs; ++h) {
        l.reset_event(c.ev->handle(done));
        l.copy(c.x->ptr(), dx.ptr(), size_t(M) * kHid * 2);
        l.copy(c.route->ptr(), static_cast<uint8_t*>(dx.ptr()) + size_t(kVerifyRows) * kHid * 2, size_t(M) * kRw * 4);
        l.barrier_signal(c.ev->handle(ready));
        l.wait_event(c.ev->handle(done));
      }
      l.close();
      std::vector<double> t;
      for (int rep = 0; rep < 6; ++rep) {
        c.ev->host_reset(ready);
        const auto t0 = Clock::now();
        q.execute(l, &f);
        for (uint32_t h = 0; h < hand_offs; ++h) {
          const auto w0 = Clock::now();
          while (!c.ev->signalled(ready))
            if (us_since(w0) > 1e6) throw std::runtime_error("CpuExperts: a hand-off never arrived (1 s)");
          c.ev->host_reset(ready);
          c.ev->host_signal(done);
        }
        f.wait();
        if (rep > 0) t.push_back(us_since(t0));
      }
      return median(t);
    };
    hand_us_[M] = std::max(0.0, (run(8) - run(1)) / 7.0);
    c.ev->host_reset(ready);
    c.ev->host_reset(done);
  }
  for (uint32_t M = 2; M < kVerifyRows; ++M)
    hand_us_[M] = hand_us_[1] + (hand_us_[kVerifyRows] - hand_us_[1]) * (M - 1) / (kVerifyRows - 1);
}

// dec[M][n]: of a layer's n host-RAM experts, how many the CPU takes - the k that finishes the layer soonest, the CPU's
// k beside the card's reads of the rest's gate||up (2/3 of an expert's bytes), the rest's down after the wait:
// max(hand-off + k cpu, (n - k) pcie 2/3) + (n - k) pcie / 3. The CPU-route list runs at M when an expert on the CPU
// plus a hand-off costs less than one over the link. Under tp the costs are a card's half expert's: the CPU's in the
// batch of both cards' halves (measured and observed that way), each card's link its own (2026-11-26, served with MTP:
// counting the CPU twice for the two cards sent auto back to PCIe at 1-4 rows, tool calls 98.5 t/s geomean against
// 112.5 with every host half expert on the CPU).
bool CpuExperts::route() {
  const std::array<bool, kVerifyRows + 1> was = on_;
  uint32_t* dec = dec_->as<uint32_t>();
  for (uint32_t M = 1; M <= kVerifyRows; ++M) {
    const double cc = cpu_us_[M], p = pcie_us_, h = hand_us_[M];
    for (uint32_t n = 0; n <= kMaxN; ++n) {
      uint32_t best = 0;
      double bt = 1e300;
      for (uint32_t k = 0; k <= n; ++k) {
        const double t = std::max(h + k * cc, (n - k) * p * 2.0 / 3.0) + (n - k) * p / 3.0;
        if (t < bt) {
          bt = t;
          best = k;
        }
      }
      dec[size_t(M) * (kMaxN + 1) + n] = mode_ == Mode::All ? n : best;
    }
    on_[M] = mode_ == Mode::All || (mode_ == Mode::Adaptive && cc + h < p);
  }
  return was != on_;
}

void CpuExperts::set_mode(Mode mode) {
  mode_ = mode;
  (void)route();
}

void CpuExperts::decide(bool log) {
  if (!route() && !log) return;
  uint32_t lo = 0, hi = 0;
  for (uint32_t M = 1; M <= kVerifyRows; ++M)
    if (on_[M]) {
      if (!lo) lo = M;
      hi = M;
    }
  const std::string costs = "CPU " + std::to_string(int(cpu_us_[1] + 0.5)) + " / " +
                            std::to_string(int(cpu_us_[kVerifyRows] + 0.5)) +
                            (share_ > 1 ? " us a card's half expert at 1 / " : " us an expert at 1 / ") +
                            std::to_string(kVerifyRows) + " rows, PCIe " + std::to_string(int(pcie_us_ + 0.5)) +
                            " us, hand-off " + std::to_string(int(hand_us_[1] + 0.5)) + " us";
  const char* isa = isa_ == cpu::Isa::Avx512 ? "AVX-512" : avx2_forced_ ? "AVX2 (forced)" : "AVX2";
  if (!lo)
    std::fprintf(stderr, "experts: host-RAM experts read over PCIe (the CPU is slower on this machine: %s)\n",
                 costs.c_str());
  else
    std::fprintf(stderr, "experts: host-RAM experts computed on the CPU (%s, %u cores) at %u%s rows%s (%s)\n", isa,
                 cores_, lo, lo == hi ? "" : ("-" + std::to_string(hi)).c_str(),
                 hi == kVerifyRows ? "" : ", read over PCIe above", costs.c_str());
  if (lo && isa_ == cpu::Isa::Avx512)
    std::fprintf(stderr,
                 "experts: warning: host-RAM experts use the AVX-512 kernel, which has not yet been tested; if output "
                 "looks wrong (garbage, wrong tokens), please report it and set B70_CPU_EXPERTS_AVX2=1 to use the AVX2 "
                 "kernel\n");
}

bool CpuExperts::has(uint32_t card, uint32_t li) const {
  return card < cards_.size() && li < cards_[card].n && cards_[card].has[li];
}

CpuExperts::Channel CpuExperts::channel(uint32_t card, uint32_t M, uint32_t li) const {
  if (!has(card, li) || M < 1 || M > kVerifyRows) throw std::logic_error("CpuExperts::channel: no hand-off there");
  const Card& c = cards_[card];
  Channel ch;
  ch.ready = c.ev->handle(ev_index(c, M, li, 0));
  ch.done = c.ev->handle(ev_index(c, M, li, 1));
  ch.x = c.x->ptr();
  ch.route = c.route->ptr();
  ch.cpu_y = c.y->as<float>();
  ch.dec = dec_->as<uint32_t>() + size_t(M) * (kMaxN + 1);
  ch.mlo = c.mlo[li];
  ch.msz = c.msz[li];
  return ch;
}

void CpuExperts::before_step(uint32_t M) {
  if (mode_ == Mode::Adaptive && M >= 1 && M <= kVerifyRows && obs_n_[M] >= kMinObs) {
    bool redo = false;
    if (obs_cpu_[M] > 0.0 && std::abs(obs_cpu_[M] - cpu_us_[M]) > kDrift * cpu_us_[M]) {
      cpu_us_[M] = obs_cpu_[M];
      redo = true;
    }
    obs_n_[M] = 0;
    if (redo) decide(false);
  }
  const uint32_t pm = M >= 1 && M <= kVerifyRows && on_[M] ? M : 0;
  uint32_t n = 0;
  for (const Card& c : cards_) n += pm ? uint32_t(c.host_layers.size()) : 0;
  pending_.store(n, std::memory_order_release);
  poll_m_.store(pm, std::memory_order_release);
  wake_.fetch_add(1, std::memory_order_release);
  wake_.notify_all();
}

void CpuExperts::set_rows(uint32_t card, const std::vector<uint64_t>& rows) {
  Card& c = cards_.at(card);
  if (rows.size() != c.rows.size()) throw std::logic_error("CpuExperts::set_rows: not the card's table");
  c.rows = rows;
}

void CpuExperts::reset() {
  for (Card& c : cards_) {
    for (uint32_t i = 0; i < c.ev->size(); ++i) c.ev->host_reset(i);
    c.cursor.fill(0);
  }
  pending_.store(0, std::memory_order_release);
}

// One hand-off: the layer's distinct experts in first-occurrence order (q4_moe_union.cl's rank order; slot order at
// M 1), those in its mirror, the first dec[n] of them for the CPU, into cpu_y at each of their (row, slot) places.
uint32_t CpuExperts::collect(const Card& c, uint32_t M, uint32_t li, std::vector<cpu::Q4Job>& jobs) {
  const uint32_t* rt = c.route->as<uint32_t>();
  const uint16_t* x = c.x->as<uint16_t>();
  float* y = c.y->as<float>();
  const uint64_t* row = c.rows.data() + size_t(li) * 2 * kE;
  uint32_t ids[kMaxN], nd = 0, n = 0;
  bool host[kMaxN];
  for (uint32_t r = 0; r < M; ++r)
    for (uint32_t k = 0; k < kTopK; ++k) {
      const uint32_t e = std::min(rt[r * kRw + k], kE - 1);
      if (std::find(ids, ids + nd, e) != ids + nd) continue;
      host[nd] = row[e] - c.mlo[li] < c.msz[li];
      n += host[nd];
      ids[nd++] = e;
    }
  const uint32_t lim = dec_->as<uint32_t>()[size_t(M) * (kMaxN + 1) + n];
  uint32_t taken = 0;
  for (uint32_t d = 0; d < nd && taken < lim; ++d) {
    if (!host[d]) continue;
    ++taken;
    cpu::Q4Job j;
    j.gu = reinterpret_cast<const uint32_t*>(row[ids[d]]);
    j.dn = reinterpret_cast<const uint32_t*>(row[kE + ids[d]]);
    for (uint32_t r = 0; r < M; ++r)
      for (uint32_t k = 0; k < kTopK; ++k)
        if (std::min(rt[r * kRw + k], kE - 1) == ids[d]) {
          j.x[j.n] = x + size_t(r) * kHid;
          j.out[j.n++] = y + (size_t(r) * kSlots + k) * kHid;
        }
    jobs.push_back(j);
  }
  return taken;
}

void CpuExperts::loop() {
  uint64_t seen = wake_.load(std::memory_order_acquire);
  auto last = Clock::now();
  std::vector<cpu::Q4Job> jobs;
  std::vector<uint32_t> batch;   // the cards served together
  while (!quit_.load(std::memory_order_acquire)) {
    const uint32_t M = poll_m_.load(std::memory_order_acquire);
    batch.clear();
    if (M)
      for (uint32_t i = 0; i < cards_.size(); ++i) {
        const Card& c = cards_[i];
        if (!c.host_layers.empty() && c.ev->signalled(ev_index(c, M, c.host_layers[c.cursor[M]], 0))) batch.push_back(i);
      }
    if (!batch.empty() && share_ > 1) {
      // tp: the other card hands off the same layer microseconds later - wait for it (bounded: a card that never comes
      // is served alone) so both cards' half experts run in one pool run a phase
      const uint32_t li = cards_[batch[0]].host_layers[cards_[batch[0]].cursor[M]];
      const auto w0 = Clock::now();
      for (uint32_t i = 0; i < cards_.size(); ++i) {
        const Card& c = cards_[i];
        if (std::find(batch.begin(), batch.end(), i) != batch.end() || c.host_layers.empty() ||
            c.host_layers[c.cursor[M]] != li)
          continue;
        const uint32_t ev = ev_index(c, M, li, 0);
        while (!c.ev->signalled(ev) && Clock::now() - w0 < std::chrono::milliseconds(2)) _mm_pause();
        if (c.ev->signalled(ev)) batch.push_back(i);
      }
    }
    if (!batch.empty()) {
      jobs.clear();
      for (uint32_t i : batch) {
        const Card& c = cards_[i];
        c.ev->host_reset(ev_index(c, M, c.host_layers[c.cursor[M]], 0));
      }
      std::atomic_thread_fence(std::memory_order_acquire);
      for (uint32_t i : batch) collect(cards_[i], M, cards_[i].host_layers[cards_[i].cursor[M]], jobs);
      const auto t0 = Clock::now();
      cpu::q4_experts(jobs.data(), uint32_t(jobs.size()), pool_.get(), isa_, inter_);
      if (!jobs.empty()) {
        const double us = us_since(t0) / double(jobs.size());
        obs_cpu_[M] = obs_cpu_[M] > 0.0 ? 0.9 * obs_cpu_[M] + 0.1 * us : us;
      }
      std::atomic_thread_fence(std::memory_order_release);
      for (uint32_t i : batch) {
        Card& c = cards_[i];
        c.ev->host_signal(ev_index(c, M, c.host_layers[c.cursor[M]], 1));
        c.cursor[M] = (c.cursor[M] + 1) % uint32_t(c.host_layers.size());
        ++obs_n_[M];
        pending_.fetch_sub(1, std::memory_order_acq_rel);
      }
      last = Clock::now();
      continue;
    }
    // Asleep only with no hand-off of the running step left, ~2 ms after the last one: a list waits on this thread.
    if (pending_.load(std::memory_order_acquire) == 0 && Clock::now() - last > std::chrono::milliseconds(2)) {
      wake_.wait(seen, std::memory_order_acquire);   // until the next step (or the destructor)
      seen = wake_.load(std::memory_order_acquire);
      last = Clock::now();
    } else {
      _mm_pause();
    }
  }
}

}  // namespace runtime::qwen4exp
