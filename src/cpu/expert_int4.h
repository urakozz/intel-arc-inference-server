#pragma once
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <vector>

// PERF-124: one routed SwiGLU expert (gate||up: hidden -> 2 x inter, h = silu(gate) * up, down: inter -> hidden) computed
// on the CPU from int4 g64 symmetric weights in host RAM, for M = 1..8 activation rows. The weights are streamed once for
// all M rows; the math is the card's (q4_moe.cl): w = (q - 8) * scale, a group's 64 products summed in fp32, times the
// group's scale, added to the column's fp32 sum; gate / up are rounded to bf16, h = bf16(bf16(silu(gate)) * up) - the
// down input is bf16 as on the card. The result is the fp32 down sum (the caller's combine rounds and weights it).
namespace cpu {

inline constexpr uint32_t kGroup = 64;
inline constexpr uint32_t kMaxRows = 8;
inline constexpr uint32_t kSliceGroups = 8;
inline constexpr uint32_t slice_groups(uint32_t G, uint32_t sl) { return G - sl * kSliceGroups < kSliceGroups ? G - sl * kSliceGroups : kSliceGroups; }

// A [K x N] int4 matrix in K slices of 8 groups (512 k; the last slice may be shorter), a slice's columns one after
// another: group g (slice sl = g / 8, gi = g % 8, gs groups in that slice) of column n is 32 bytes at
// q[((sl * 8 * N) + n * gs + gi) * 32], byte j = code(k = 64 g + j) | code(k = 64 g + 32 + j) << 4 (the unpack is a mask and
// a shift), and its f16 scale at s[(sl * 8 * N) + n * gs + gi]. Code c means (c - 8). A thread's columns of a slice are
// one sequential run, and the slice of x they all read stays in L1.
struct Int4Matrix {
  uint32_t K = 0, N = 0;
  std::vector<uint8_t> q;
  std::vector<uint16_t> s;
  size_t bytes() const { return q.size() + s.size() * 2; }
};
// codes [N][K] (0..15), scales [N][K/64] f16 words.
Int4Matrix pack_int4(const uint8_t* codes, const uint16_t* scales, uint32_t K, uint32_t N);

// gate_up: K = hidden, N = 2 x inter, columns [0, inter) the gate, [inter, 2 inter) the up; down: K = inter, N = hidden.
struct ExpertInt4 {
  Int4Matrix gate_up, down;
  uint32_t hidden() const { return gate_up.K; }
  uint32_t inter() const { return down.K; }
  size_t bytes() const { return gate_up.bytes() + down.bytes(); }
};

// A fixed set of spinning workers, asleep once idle for ~2 ms (a per-expert call is hundreds of microseconds; a condition variable's wake-up is not
// small beside that). run() executes fn(ctx, tid, n) on n = threads() threads, the caller being tid 0, and returns when
// all are done. Not re-entrant; one caller at a time.
class Pool {
 public:
  explicit Pool(unsigned threads, std::vector<int> cpus = {});   // cpus: thread i runs on cpus[i] (none: the scheduler's)
  ~Pool();
  Pool(const Pool&) = delete;
  Pool& operator=(const Pool&) = delete;
  unsigned threads() const { return n_; }
  void run(void (*fn)(void*, unsigned, unsigned), void* ctx);

 private:
  void worker(unsigned tid);
  unsigned n_;
  std::vector<std::thread> workers_;
  // The job (read by every worker) and the two counters (written by the caller / by each worker) on lines of their own.
  alignas(64) void (*fn_)(void*, unsigned, unsigned) = nullptr;
  void* ctx_ = nullptr;
  std::atomic<bool> stop_{false};
  alignas(64) std::atomic<uint64_t> gen_{0};
  alignas(64) std::atomic<unsigned> done_{0};
  std::atomic<unsigned> sleepers_{0};   // workers blocked in gen_.wait (run() notifies only then)
};

enum class Isa { Avx2, Avx512 };
const char* isa_name(Isa i);
bool cpu_has_avx512();   // avx512f + avx512bw at run time
inline Isa best_isa() { return cpu_has_avx512() ? Isa::Avx512 : Isa::Avx2; }

// Scratch for one call: x as fp32 [M][hidden], h [M][inter].
struct ExpertScratch {
  std::vector<float> x, h;
};
// x: bf16 [M][hidden]; out: fp32 [M][hidden], the down sums. M in 1..8. pool null: the calling thread alone.
void expert_forward(const ExpertInt4& e, const uint16_t* x, uint32_t M, float* out, ExpertScratch& sc, Pool* pool,
                    Isa isa);

}  // namespace cpu
