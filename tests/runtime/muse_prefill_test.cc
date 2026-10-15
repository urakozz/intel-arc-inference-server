// Spec 23d (MG2-MG4 on prefill, Review Focus 2 / 3 / 5): MuseEngine::prefill on the card with a checkpoint - box only
// (labels checkpoint;muse;prefill). 23b's synthetic checkpoint (tools/box_validate/muse_oracle.sh synth-ckpt: 4
// real-width layers [s, s, s, f]) carries it; the golden gate itself is muse_golden_test's `prefill` mode.
//
//   muse_prefill_test <checkpoint> <prompts dir> [all | split | pp] [bf16] [pp2[:split]]
//
// The prompts: <dir>/muse_short_prose.ids (42 ids: every window covers the prompt), muse_4k.ids and muse_8k.ids
// (the window and the ring active: a chunk of 2048 starts at 2048, 4096 and 6144, where the sliding layers' rings
// wrap) - 23a's golden prompts, Muse's tokenisation.
//
//   mode `all` (default; one card, or two with pp2), per prompt:
//     1. the walk: prefill_chunk_launches(desc, placement) per chunk (1 + 134 x layers) and the head's 4; the plan's
//        prefill term (prefill_sizes + the prefill link) equals the allocation, per device; pos == the prompt's length;
//     2. (4k) MG4, determinism and replay, bitwise: two immediate prefills (reset between) and two RECORDED ones
//        (set_prefill_replay: B70_PREFILL_REPLAY's path) leave the same KV of every full layer, the ring of every
//        sliding layer, the last logits and the first token;
//     3. (4k) chunks of 64 against the default chunk: KV, rings, logits and the first token bitwise (Review Focus 2:
//        every kernel is row-local or keyed to ABSOLUTE positions - the slab GEMMs, the 64-key flash tiles, RoPE, the
//        ring slots);
//     4. Review Focus 5, prefill against decode's M = 1 fill (ingest, one id per replay) of the same prompt: per layer,
//        per position the ring / the cache still holds, K and V cosine - gated (PROPOSED, Kolibri's bars): every row of
//        layer 0 >= 0.999, all rows' median >= 0.9998, 1st percentile >= 0.99; printed per layer; pos exact; the ring
//        slots p & 4095 (read_kv addresses them so); the last logits' cosine printed;
//     5. tokens: prefill + 32 greedy against ingest + 32 greedy - every row whose decode-side top-2 raw gap is at least
//        one bf16 ulp of the top logit agrees; teacher-forced on decode's token.
//   mode `split` (Review Focus 2 / 3): the 8k prompt in ONE prefill against TWO calls split at 64, 1000, 2048, 2049,
//     4097 - KV, rings, the last logits and the first token. Gated: the multiples of 64 bitwise, the others within
//     the bars (logits cosine >= 0.9995, layers 0-1 every row >= 0.999, median >= 0.9998, p01 >= 0.99) - each printed
//     bitwise or not. The full (NoPE) layer's later chunk reads the earlier rows from the cache, not recomputed: a
//     split continuation's full-layer KV of the FIRST part is bitwise the one-call run's.
//   mode `pp` (two GPUs): --pp 2 --pipeline-split <s> (pp2:<s>, default 2) against --pp 1 on the same checkpoint,
//     BITWISE: one prefill (KV, rings, logits, the first token, both Control blocks equal after the mirror), recorded +
//     replayed, a split continuation (1000 + rest), chunks of 64; each card's prefill launches are
//     prefill_device_launches'; P4 - device 0's half of a chunk dropped (drop_next_handoff): the prefill throws within
//     its 5 s bound naming device 1, the next prefill is refused until reset(), which recovers bitwise.
//
// B70_MUSE_ATTN=eager runs the eager attention in BOTH halves (prefill's mg_pf_attn EAGER and decode's mg_attn_eager;
// one variable, one parser) - comparison 4 is then between matched forms.
// Exit 77 (SKIP) when the checkpoint or the prompts are absent, or pp needs two GPUs.
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "check.h"
#include "common/bf16.h"
#include "runtime/control.h"
#include "runtime/muse/muse_sizes.h"
#include "runtime/muse_rig.h"

namespace {

namespace rm = runtime::muse;
using Ids = std::vector<uint32_t>;
constexpr uint32_t kGen = 32;
constexpr uint32_t kRing = model::MuseGlimmerDesc::kRing;

double cos_bf16(const uint16_t* a, const uint16_t* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    const double x = common::bf16_to_f32(a[i]), y = common::bf16_to_f32(b[i]);
    ab += x * y;
    aa += x * x;
    bb += y * y;
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}
double cos_f32(const float* a, const float* b, size_t n) {
  double ab = 0, aa = 0, bb = 0;
  for (size_t i = 0; i < n; ++i) {
    ab += double(a[i]) * b[i];
    aa += double(a[i]) * a[i];
    bb += double(b[i]) * b[i];
  }
  return (aa == 0 && bb == 0) ? 1.0 : ab / std::sqrt(aa * bb);
}

struct Snap {
  uint32_t first_token = 0, pos = 0;
  std::vector<uint32_t> lo;                // per layer: the first position read
  std::vector<std::vector<uint16_t>> kv;   // per layer: K rows then V rows, positions [lo, n)
  std::vector<float> logits;
  bool operator==(const Snap& o) const {
    return first_token == o.first_token && pos == o.pos && kv == o.kv && logits == o.logits;
  }
};

// Every layer's K / V over the positions it still holds after n: a full layer [0, n), a sliding layer the ring's last
// min(n, 4096) positions (read through the ring at p & 4095, MuseEngine::read_kv).
Snap snap(rm::MuseEngine& e, uint32_t n) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  Snap s;
  const runtime::Control c = e.control(e.devices() - 1);
  s.first_token = c.cur_token[0];
  s.pos = c.pos;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const uint32_t lo = !d.is_full(l) && n > kRing ? n - kRing : 0;
    std::vector<uint16_t> k = e.read_kv(l, lo, n - lo, false), v = e.read_kv(l, lo, n - lo, true);
    k.insert(k.end(), v.begin(), v.end());
    s.lo.push_back(lo);
    s.kv.push_back(std::move(k));
  }
  s.logits = e.read_logits();
  return s;
}

struct KvCmp {
  double layer0_worst = 1, median = 1, p01 = 1, worst = 1;
  uint32_t worst_layer = 0, worst_pos = 0, below99 = 0;
  std::vector<double> layer_worst, layer_median;
};
// Positions both snapshots hold of every layer (the later lo), K and V cosine per row.
KvCmp compare_kv(const model::MuseGlimmerDesc& d, const Snap& a, const Snap& b, uint32_t n) {
  KvCmp c;
  const size_t row = d.kv_n();
  std::vector<double> all;
  for (uint32_t l = 0; l < d.layers; ++l) {
    const uint32_t lo = std::max(a.lo[l], b.lo[l]);
    const size_t na = n - a.lo[l], nb = n - b.lo[l];
    std::vector<double> per;
    for (uint32_t p = lo; p < n; ++p) {
      const uint16_t* ka = a.kv[l].data() + size_t(p - a.lo[l]) * row;
      const uint16_t* kb = b.kv[l].data() + size_t(p - b.lo[l]) * row;
      const double v = std::min(cos_bf16(ka, kb, row), cos_bf16(ka + na * row, kb + nb * row, row));
      per.push_back(v);
      all.push_back(v);
      if (l == 0) c.layer0_worst = std::min(c.layer0_worst, v);
      if (v < c.worst) {
        c.worst = v;
        c.worst_layer = l;
        c.worst_pos = p;
      }
      c.below99 += v < 0.99;
    }
    std::sort(per.begin(), per.end());
    c.layer_worst.push_back(per.empty() ? 1.0 : per.front());
    c.layer_median.push_back(per.empty() ? 1.0 : per[per.size() / 2]);
  }
  std::sort(all.begin(), all.end());
  c.median = all[all.size() / 2];
  c.p01 = all[all.size() / 100];
  return c;
}

void print_layers(const KvCmp& c) {
  std::printf("   per layer (worst / median):");
  for (size_t l = 0; l < c.layer_worst.size(); ++l) std::printf(" L%zu %.6f/%.7f", l, c.layer_worst[l], c.layer_median[l]);
  std::printf("\n");
}

bool run_prompt(rm::MuseEngine& e, const std::string& name, const Ids& prompt, bool deep) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  const uint32_t n = uint32_t(prompt.size());
  bool ok = true;
  std::printf("\n== %s: %u ids\n", name.c_str(), n);

  // ---- 1. the walk and the plan -----------------------------------------------------------------------------------
  e.prepare_prefill();
  const size_t l0 = e.prefill_launches();
  e.reset();
  e.prefill(prompt);
  const size_t chunks = (n + rm::kPfC - 1) / rm::kPfC;
  const size_t per_chunk = rm::prefill_chunk_launches(d, e.weights().placement);
  CHECK_EQ(per_chunk, size_t(1 + rm::prefill_layer_launches(d) * d.layers));
  CHECK_EQ(e.prefill_launches() - l0, chunks * per_chunk + rm::kPrefillHeadLaunches);
  CHECK_EQ(e.pos(), n);
  for (uint32_t i = 0; i < e.devices(); ++i)
    CHECK_EQ(e.memory_use(i).prefill_scratch,
             rm::prefill_sizes(d).total() + (e.devices() > 1 ? rm::pf_link_bytes(d, i) : 0));
  std::printf("1. walk: %zu chunk(s) x %zu + %zu launches on %u device(s); prefill scratch %.3f GB a device = the plan's\n"
              "%s\n", chunks, per_chunk, rm::kPrefillHeadLaunches, e.devices(), rm::prefill_sizes(d).total() / 1e9,
              e.memory_line().c_str());
  const Snap a = snap(e, n);

  if (deep) {
    // ---- 2. determinism and replay ----------------------------------------------------------------------------------
    e.reset();
    e.prefill(prompt);
    const bool same = snap(e, n) == a;
    e.set_prefill_replay(true);
    e.reset();
    e.prefill(prompt);   // records
    const bool rec1 = snap(e, n) == a;
    e.reset();
    e.prefill(prompt);   // replays the recordings
    const bool rec2 = snap(e, n) == a;
    e.set_prefill_replay(false);
    std::printf("2. MG4: immediate twice %s; recorded %s, replayed %s\n", same ? "bitwise" : "DIFFERS",
                rec1 ? "bitwise" : "DIFFERS", rec2 ? "bitwise" : "DIFFERS");
    ok = ok && same && rec1 && rec2;
    // ---- 3. chunks of 64 ---------------------------------------------------------------------------------------------
    e.reset();
    e.prefill(prompt, 64);
    const bool c64 = snap(e, n) == a;
    std::printf("3. chunks of 64 vs %u: KV, rings, logits, first token %s\n", rm::kPfC, c64 ? "bitwise" : "DIFFER");
    ok = ok && c64;
  }

  // ---- 4. against decode's fill --------------------------------------------------------------------------------------
  e.reset();
  e.ingest(prompt);
  CHECK_EQ(e.pos(), n);
  const Snap dec = snap(e, n);
  const KvCmp kc = compare_kv(d, a, dec, n);
  const bool kv_ok = kc.layer0_worst >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
  const double lg = cos_f32(a.logits.data(), dec.logits.data(), d.vocab);
  std::printf("4. prefill KV / rings vs decode's fill (%u positions x %u layers): layer 0's worst %.6f, median %.7f, "
              "p01 %.6f, %u rows < 0.99, worst %.4f (L%u pos %u)%s; pos %u == %u; last logits cosine %.6f; first token "
              "%u vs %u\n", n, d.layers, kc.layer0_worst, kc.median, kc.p01, kc.below99, kc.worst, kc.worst_layer,
              kc.worst_pos, kv_ok ? "" : "  ** FAIL **", a.pos, dec.pos, lg, a.first_token, dec.first_token);
  print_layers(kc);
  ok = ok && kv_ok && a.pos == dec.pos;

  // ---- 5. tokens -------------------------------------------------------------------------------------------------------
  std::vector<uint32_t> dtok(kGen);
  std::vector<bool> determined(kGen);
  for (uint32_t j = 0; j < kGen; ++j) {   // decode side: the state after ingest(prompt) is still there
    const std::vector<float> l = e.read_logits();
    std::vector<float> s(l.begin(), l.end());
    std::partial_sort(s.begin(), s.begin() + 2, s.end(), std::greater<float>());
    const float ulp = std::fabs(common::bf16_to_f32(uint16_t(common::f32_to_bf16(s[0]) + 1)) - s[0]);
    determined[j] = s[0] - s[1] >= ulp;
    dtok[j] = e.control(e.devices() - 1).cur_token[0];
    e.ingest({dtok[j]});
  }
  uint32_t agree = 0, det = 0, det_agree = 0;
  e.reset();
  e.prefill(prompt);
  for (uint32_t j = 0; j < kGen; ++j) {
    const uint32_t got = e.control(e.devices() - 1).cur_token[0];
    agree += got == dtok[j];
    if (determined[j]) {
      ++det;
      det_agree += got == dtok[j];
    }
    e.ingest({dtok[j]});   // teacher-forced on decode's token
  }
  std::printf("5. tokens: %u / %u agree; %u / %u determined rows agree%s\n", agree, kGen, det_agree, det,
              det_agree == det ? "" : "  ** FAIL **");
  return ok && det_agree == det;
}

int run_all(rm::MuseEngine& e, const std::vector<std::pair<std::string, Ids>>& prompts) {
  bool ok = true;
  for (const auto& [name, ids] : prompts) ok = run_prompt(e, name, ids, name == "muse_4k") && ok;
  return ok ? 0 : 1;
}

int run_split(rm::MuseEngine& e, const Ids& ids) {
  const model::MuseGlimmerDesc& d = e.weights().desc;
  const uint32_t N = uint32_t(ids.size());
  e.reset();
  e.prefill(ids);
  const Snap ref = snap(e, N);
  bool ok = true;
  for (uint32_t s : {64u, 1000u, 2048u, 2049u, 4097u}) {
    if (s >= N) continue;
    e.reset();
    e.prefill(Ids(ids.begin(), ids.begin() + s));
    // Review Focus 3: the full layers' rows of the first part, written before the second call, are read from the
    // cache by the second part's rows - and the first part's rows are bitwise the one-call run's when the cut is at a
    // multiple of 64 (they are computed by the same chunks' kernels on the same rows)
    e.prefill(Ids(ids.begin() + s, ids.end()));
    CHECK_EQ(e.pos(), N);
    const Snap got = snap(e, N);
    const bool bitwise = got == ref;
    const double lg = cos_f32(got.logits.data(), ref.logits.data(), d.vocab);
    const KvCmp kc = compare_kv(d, ref, got, N);
    const double shallow = std::min(kc.layer_worst[0], d.layers > 1 ? kc.layer_worst[1] : 1.0);
    const bool bars = lg >= 0.9995 && shallow >= 0.999 && kc.median >= 0.9998 && kc.p01 >= 0.99;
    const bool pass = s % 64 == 0 ? bitwise : (bitwise || bars);
    std::printf("  split at %4u: %s | logits %.6f, layers 0-1 worst %.6f, median %.7f, p01 %.6f%s\n", s,
                bitwise ? "bitwise" : "not bitwise", lg, shallow, kc.median, kc.p01, pass ? "" : "  ** FAIL **");
    ok = ok && pass;
  }
  return ok ? 0 : 1;
}

// One card against two, bitwise. Both engines on the same checkpoint.
int run_pp(muse_rig::Rig& rig, const std::string& snapdir, rm::MuseOptions o, uint32_t split, const Ids& ids) {
  struct Run {
    Snap one, rep, cont, c64;
  };
  const auto runs = [&](rm::MuseEngine& e) {
    Run r;
    const uint32_t n = uint32_t(ids.size());
    e.reset();
    e.prefill(ids);
    r.one = snap(e, n);
    if (e.devices() == 2) {
      const runtime::Control c0 = e.control(0), c1 = e.control(1);
      CHECK(std::memcmp(&c0, &c1, sizeof c0) == 0);   // the mirror after the head
      for (uint32_t i = 0; i < 2; ++i) {
        std::printf("   device %u: %zu prefill launches\n", i, e.prefill_launches(i));
        CHECK_EQ(e.prefill_launches(i), ((n + rm::kPfC - 1) / rm::kPfC) *
                                            rm::prefill_device_launches(e.weights().desc, e.weights().placement, i) +
                                            (i == 1 ? rm::kPrefillHeadLaunches : 0));
      }
    }
    e.set_prefill_replay(true);
    e.reset();
    e.prefill(ids);
    e.reset();
    e.prefill(ids);
    r.rep = snap(e, n);
    e.set_prefill_replay(false);
    e.reset();
    e.prefill(Ids(ids.begin(), ids.begin() + 1000));
    e.prefill(Ids(ids.begin() + 1000, ids.end()));
    r.cont = snap(e, n);
    e.reset();
    e.prefill(ids, 64);
    r.c64 = snap(e, n);
    return r;
  };
  rm::MuseOptions o2 = o;
  o2.split = split ? split : 2;
  o2.prefill_timeout_ms = 5000;   // P4's bound below
  o.split = 0;
  muse_rig::build(rig, snapdir, 1, o);
  const Run one = runs(*rig.eng);
  muse_rig::build(rig, snapdir, 2, o2);
  const model::MuseGlimmerDesc& d = rig.eng->weights().desc;
  for (uint32_t i = 0; i < 2; ++i) CHECK(rig.eng->prefill_launches(i) == 0);
  const Run two = runs(*rig.eng);
  const bool a = two.one == one.one, b = two.rep == one.rep && two.rep == one.one, c = two.cont == one.cont,
             e = two.c64 == one.c64;
  bool p4 = false;
  {
    rm::MuseEngine& en = *rig.eng;
    en.reset();
    en.drop_next_handoff();
    const auto t0 = std::chrono::steady_clock::now();
    std::string what;
    try {
      en.prefill(ids);
    } catch (const std::exception& ex) {
      what = ex.what();
    }
    const double s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
    bool refused = false;
    try {
      en.prefill(ids);
    } catch (const std::exception&) {
      refused = true;
    }
    en.reset();
    en.prefill(ids);
    const bool back = snap(en, uint32_t(ids.size())) == one.one;
    p4 = !what.empty() && what.find("device 1") != std::string::npos && s < 30.0 && refused && back;
    std::printf("P4: a dropped prefill hand-off threw after %.1f s (\"%.120s\"); the next prefill %s; reset() recovers "
                "%s\n", s, what.c_str(), refused ? "refused" : "RAN", back ? "bitwise" : "DIFFERENTLY");
  }
  std::printf("pp: --pp 2 --pipeline-split %u vs --pp 1 (%u layers): one prefill %s, recorded + replayed %s, split "
              "continuation (1000 + rest) %s, chunks of 64 %s\n", rig.eng->split(), d.layers, a ? "bitwise" : "DIFFERS",
              b ? "bitwise" : "DIFFERS", c ? "bitwise" : "DIFFERS", e ? "bitwise" : "DIFFERS");
  return a && b && c && e && p4 ? 0 : 1;
}

}  // namespace

int main(int argc, char** argv) {
  const std::string snapdir = muse_rig::muse_snapshot(argc > 1 ? argv[1] : "");
  if (snapdir.empty()) {
    std::printf("SKIP: no Muse-Glimmer checkpoint at '%s' (muse_oracle.sh synth-ckpt; the real one: hf download)\n",
                argc > 1 ? argv[1] : "");
    return 77;
  }
  const std::string pdir = argc > 2 ? argv[2] : "tests/golden/prompts";
  const std::string mode = argc > 3 ? argv[3] : "all";
  CHECK(mode == "all" || mode == "split" || mode == "pp");
  rm::MuseOptions o;
  o.prefill = true;
  uint32_t devices = 1, split = 0;
  for (int i = 4; i < argc; ++i) {
    const std::string x = argv[i];
    if (x == "bf16") {
      o.int8_head = false;
    } else if (x == "pp2") {
      devices = 2;
    } else if (x.rfind("pp2:", 0) == 0) {
      devices = 2;
      split = uint32_t(std::strtoul(x.c_str() + 4, nullptr, 10));
    } else {
      std::fprintf(stderr, "muse_prefill_test: unknown flag %s (bf16, pp2[:split])\n", x.c_str());
      return 2;
    }
  }
  if ((mode == "pp" || devices == 2) && l0::Context::gpu_count() < 2) {
    std::printf("SKIP: two cards needed, Level Zero shows %u\n", l0::Context::gpu_count());
    return 77;
  }
  const model::MuseGlimmerDesc pre = loader::muse_checkpoint_desc(snapdir);
  std::vector<std::pair<std::string, Ids>> prompts;
  for (const char* p : {"muse_short_prose", "muse_4k", "muse_8k"}) {
    Ids ids = muse_rig::read_ids(pdir + "/" + p + ".ids", pre.vocab);
    if (!ids.empty()) prompts.emplace_back(p, std::move(ids));
  }
  if (prompts.size() != 3) {
    std::printf("SKIP: %s/muse_{short_prose,4k,8k}.ids are not all there\n", pdir.c_str());
    return 77;
  }
  o.max_len = muse_rig::len_for(prompts.back().second.size() + kGen + 1);
  muse_rig::Rig rig;
  if (mode == "pp") {
    const int rc = run_pp(rig, snapdir, o, split, prompts[2].second);
    std::printf("muse_prefill_test pp %s\n", rc == 0 ? "OK" : "FAILED");
    return rc;
  }
  o.split = split;
  muse_rig::build(rig, snapdir, devices, o);
  rm::MuseEngine& e = *rig.eng;
  std::printf("muse_prefill_test %s: %u layers, lm_head %s, %s attention (B70_MUSE_ATTN, both halves), %u device(s)%s; "
              "%zu decode launches, max_len %u\n", mode.c_str(), e.weights().desc.layers, o.int8_head ? "int8" : "bf16",
              rm::muse_attn_name(e.attention()), e.devices(),
              e.devices() == 2 ? (std::string(", split ") + std::to_string(e.split())).c_str() : "", e.launches(),
              e.max_len());
  const int rc = mode == "split" ? run_split(e, prompts[2].second) : run_all(e, prompts);
  std::printf("muse_prefill_test %s %s\n", mode.c_str(), rc == 0 ? "OK" : "FAILED");
  return rc;
}
