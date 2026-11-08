// Spec 13c (plan 13c Tasks 2-3): runtime::Engine's sequence slots on the card - several requests on one B70.
//
// One engine: Qwen3.8 with its MTP head and the int8 lm_head (the served form), slots 16384 + 3 x 8192 (40960
// positions), the KV form B70_KV_CACHE's (slots_kv8_test: int8). Prompts: long32k / code / prose / long, cut per slot.
//
//   S1  a slot alone is today's engine: the same session (prefill, 32 plain steps, 8 MTP iterations at K = 3) in slot 0
//       and in slot 2 gives the same ids and every step's logits bitwise (each slot's lists are today's kernels over
//       its own state and KV range).
//   S2  isolation (Review Focus 3-4): a slot's prefill, decode, reset, and a snapshot restored into it leave every
//       other slot's bytes - its GDN states (the speculative ones too), conv ring, hh rows, Control and the KV rows it
//       holds (main and head) - byte-identical; save_state / save_kv of slot 1 loaded into slot 3 continues exactly as
//       slot 1 does.
//   S3  B1 for the row kernels, bitwise, at mixed K and mixed depth (spec 13a's check through the engine): slots 0 (deep,
//       K = 3), 1 (K = 1) and 2 (K = 1) in ONE verify_rows step (8 rows) against each run alone (verify_rows of its own
//       rows): every row's logits and id, the GDN state each row wrote, the KV rows it wrote (main and head); and B3 -
//       nothing else of any slot moved (slot 3 whole, the other slots' sources and KV windows), and commit after the
//       batched step leaves each slot as after its run alone.
//   S4  B1 by the f32 judge (spec 13 §11.3): four requests decoded together (32 batched plain steps, then 8 batched
//       greedy MTP iterations at K = 1 each - 8 rows) against each alone (the slot's own lists): identical, or
//       spec_judge::streams' rule (tag slots_<kv>_<n>: the int8 head's references) - a divergence whose context has no
//       f32 reference yet AWAITS it (exit 77).
//   S5  determinism: the S4 batched run twice from the same state gives the same ids.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "check.h"
#include "golden_common.h"
#include "greedy_tie.h"
#include "l0/cmdlist.h"
#include "l0/memory.h"
#include "loader/loader.h"
#include "model/qwen35.h"
#include "runtime/control.h"
#include "runtime/engine.h"
#include "spec_judge.h"

namespace {
using runtime::Engine;
constexpr size_t V = model::Qwen35::kVocab;
const std::vector<uint32_t> kSlots = {16384, 8192, 8192, 8192};

uint64_t fnv(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
  const uint8_t* b = static_cast<const uint8_t*>(p);
  for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
  return h;
}

struct Ctx {
  l0::Context& ctx;
  Engine& e;
  l0::CmdList imm;
  l0::Mem host;   // pinned scratch for readbacks
  uint32_t VU;
  Ctx(l0::Context& c, Engine& en)
      : ctx(c), e(en), imm(l0::CmdList::immediate(c)), host(c, l0::MemKind::Host, size_t(160) << 20),
        VU(en.model().desc->vocab_used) {}

  uint64_t dev_hash(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const size_t chunk = host.size();
    for (size_t off = 0; off < n; off += chunk) {
      const size_t m = std::min(chunk, n - off);
      imm.copy(host.ptr(), static_cast<const uint8_t*>(p) + off, m);
      h = fnv(host.ptr(), m, h);
    }
    return h;
  }
  size_t state_bytes() const { return e.buffers().gdn_state.size(); }
  // Every byte of slot q's persistent state, its KV rows [0, npos) of every layer (main and head) included.
  uint64_t slot_hash(uint32_t q, uint32_t npos) {
    const Engine::SlotView v = e.slot_view(q);
    uint64_t h = fnv(v.control, sizeof(runtime::Control));
    h = dev_hash(v.state0, state_bytes(), h);
    if (v.spec) h = dev_hash(v.spec, state_bytes() * (v.states - 1), h);
    h = dev_hash(v.ring, e.buffers().conv_ring.size(), h);
    if (v.hh) h = dev_hash(v.hh, size_t(runtime::DecodeScratchDims::kM + 1) * e.model().desc->hidden * 2, h);
    return kv_hash(q, 0, npos, h);
  }
  // KV rows [p, p + n) of slot q, every layer, main and head, rows and scales.
  uint64_t kv_hash(uint32_t q, uint32_t p, uint32_t n, uint64_t h = 1469598103934665603ull) {
    const Engine::SlotView v = e.slot_view(q);
    const auto one = [&](const runtime::KvLayout& L, void* k, void* vv) {
      for (uint32_t l = 0; l < L.layers; ++l) {
        const runtime::KvLayer x = L.at_base(L.layer(k, vv, l), v.base + p);
        h = dev_hash(x.k, size_t(n) * L.row_bytes(), h);
        h = dev_hash(x.v, size_t(n) * L.row_bytes(), h);
        if (x.int8()) {
          h = dev_hash(x.ks, size_t(n) * L.scale_row_bytes(), h);
          h = dev_hash(x.vs, size_t(n) * L.scale_row_bytes(), h);
        }
      }
    };
    one(e.buffers().kv_lay, e.buffers().kv_k.ptr(), e.buffers().kv_v.ptr());
    if (e.mtp_buffers()) one(e.mtp_buffers()->kv_lay, e.mtp_buffers()->kv_k.ptr(), e.mtp_buffers()->kv_v.ptr());
    return h;
  }
  std::vector<float> rows(const float* dev, uint32_t first, uint32_t n) {
    std::vector<float> v(size_t(n) * V);
    imm.copy(v.data(), dev + size_t(first) * V, v.size() * 4);
    return v;
  }
};

std::vector<uint32_t> cut(const std::vector<uint32_t>& src, size_t from, size_t n) {
  std::vector<uint32_t> out;
  for (size_t i = 0; i < n; ++i) out.push_back(src[(from + i) % src.size()]);
  return out;
}

// One greedy step of the selected slot through its own lists: k = 0 plain (verify(0) + commit), else draft + verify +
// the longest agreeing prefix. Returns the ids emitted; `logits` gets every verify row's logits hash.
std::vector<uint32_t> lone_step(Ctx& c, uint32_t k, std::vector<uint64_t>* logits = nullptr) {
  Engine& e = c.e;
  std::vector<uint32_t> out{e.control()->cur_token[0]};
  if (k > 0) e.draft(k);
  e.verify(k);
  if (logits) {
    const std::vector<float> r = c.rows(e.verify_logits_device(), 0, k + 1);
    logits->push_back(fnv(r.data(), r.size() * 4));
  }
  uint32_t j = 0;
  while (j < k && e.draft_ids()[j] == e.verify_ids()[j]) ++j;
  for (uint32_t i = 0; i < j; ++i) out.push_back(e.draft_ids()[i]);
  e.commit(j, e.verify_ids()[j]);
  return out;
}

// S1.
bool s1(Ctx& c, const std::vector<uint32_t>& prompt) {
  Engine& e = c.e;
  auto run = [&](uint32_t q, std::vector<uint64_t>& lg) {
    e.select(q);
    e.reset();
    e.prefill(prompt);
    std::vector<uint32_t> ids;
    for (int i = 0; i < 32; ++i)
      for (uint32_t id : lone_step(c, 0, &lg)) ids.push_back(id);
    for (int i = 0; i < 8; ++i)
      for (uint32_t id : lone_step(c, 3, &lg)) ids.push_back(id);
    return ids;
  };
  std::vector<uint64_t> l0, l2;
  const std::vector<uint32_t> a = run(0, l0), b = run(2, l2);
  const bool ok = a == b && l0 == l2;
  std::printf("S1 slot 0 vs slot 2, the same session alone (%zu ids, %zu verify steps): %s\n", a.size(), l0.size(),
              ok ? "PASS (ids and every step's logits bitwise)" : "FAIL");
  return ok;
}

// S2.
bool s2(Ctx& c, const std::vector<uint32_t>& p1, const std::vector<uint32_t>& p2, const std::vector<uint32_t>& p3) {
  Engine& e = c.e;
  bool ok = true;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.reset();
  }
  e.select(1);
  e.prefill(p1);
  for (int i = 0; i < 6; ++i) lone_step(c, i % 2 ? 3 : 0);   // a rejected draft or two leave speculative states
  const uint32_t n1 = e.pos() + 16;
  auto others = [&](uint32_t busy) {
    std::vector<uint64_t> h;
    for (uint32_t q = 0; q < e.slots(); ++q) h.push_back(q == busy ? 0 : c.slot_hash(q, q == 1 ? n1 : 64));
    return h;
  };
  // A prefill and a decode of slot 2.
  std::vector<uint64_t> before = others(2);
  e.select(2);
  e.prefill(p2);
  for (int i = 0; i < 6; ++i) lone_step(c, i % 2 ? 3 : 0);
  bool r = others(2) == before;
  std::printf("S2 slot 2's prefill + decode: the other slots %s\n", r ? "byte-identical" : "MOVED - FAIL");
  ok &= r;
  // A reset of slot 3 (after a prefill).
  e.select(3);
  e.prefill(p3);
  before = others(3);
  e.select(3);
  e.reset();
  r = others(3) == before && e.pos() == 0;
  std::printf("S2 reset(slot 3): the other slots %s, its pos %u\n", r ? "byte-identical" : "MOVED - FAIL", e.pos());
  ok &= r;
  // Slot 1's snapshot into slot 3.
  e.select(1);
  const uint32_t p = e.pos();
  l0::Mem st(c.ctx, l0::MemKind::Host, e.state_bytes()), kv(c.ctx, l0::MemKind::Host, e.kv_bytes(p));
  e.save_state(st.ptr());
  e.save_kv(0, p, kv.ptr());
  const uint32_t pending = e.control()->cur_token[0];
  before = others(3);
  e.select(3);
  e.load_state(st.ptr(), p);
  e.load_kv(0, p, kv.ptr());
  e.control()->cur_token[0] = pending;   // cur_token is not snapshotted (Engine::load_state)
  r = others(3) == before;
  std::vector<uint32_t> a, b;
  e.select(1);
  for (int i = 0; i < 16; ++i)
    for (uint32_t id : lone_step(c, i % 2 ? 2 : 0)) a.push_back(id);
  e.select(3);
  for (int i = 0; i < 16; ++i)
    for (uint32_t id : lone_step(c, i % 2 ? 2 : 0)) b.push_back(id);
  std::printf("S2 slot 1's snapshot loaded into slot 3: the other slots %s; slot 3 continues as slot 1: %s\n",
              r ? "byte-identical" : "MOVED - FAIL", a == b ? "yes (ids bitwise)" : "NO - FAIL");
  return ok && r && a == b;
}

// S3.
struct RowOut {
  std::vector<float> logits;
  uint32_t id = 0;
  uint64_t dst = 0, kv = 0;
};
bool s3(Ctx& c, const std::vector<uint32_t>& deep, const std::vector<uint32_t>& p1, const std::vector<uint32_t>& p2,
        const std::vector<uint32_t>& p3) {
  Engine& e = c.e;
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.reset();
  }
  const std::vector<std::vector<uint32_t>> prompts = {deep, p1, p2, p3};
  for (uint32_t q = 0; q < e.slots(); ++q) {
    e.select(q);
    e.prefill(prompts[q]);
  }
  const std::vector<uint32_t> ks = {3, 1, 1};
  // Every slot's state before the step (live state, ring, hh row 0), its pending id and its drafts.
  std::vector<l0::Mem> st;
  std::vector<std::vector<uint32_t>> drafts(3);
  std::vector<uint32_t> pend(3), pos(3);
  for (uint32_t q = 0; q < 3; ++q) {
    e.select(q);
    e.draft(ks[q]);
    drafts[q] = e.draft_ids();
    pend[q] = e.control()->cur_token[0];
    pos[q] = e.pos();
    st.emplace_back(c.ctx, l0::MemKind::Host, e.state_bytes());
    e.save_state(st.back().ptr());
  }
  const uint32_t n3 = uint32_t(p3.size()) + 16;
  const uint64_t slot3 = c.slot_hash(3, n3);
  auto restore = [&](uint32_t q) {
    e.select(q);
    e.load_state(st[q].ptr(), pos[q]);
    e.control()->cur_token[0] = pend[q];
    for (uint32_t i = 0; i < ks[q]; ++i) e.control()->cur_token[1 + i] = drafts[q][i];
  };
  // The rows each run writes: its GDN states (indices (live + j) % 4 of its group; live is 0 after load_state) and
  // its KV rows at pos .. pos + k (the head's at pos - 1 ..).
  auto outputs = [&](const std::vector<uint32_t>& slots, uint32_t first_row) {
    std::vector<RowOut> o;
    uint32_t m = first_row;
    for (uint32_t q : slots) {
      const Engine::SlotView v = e.slot_view(q);
      for (uint32_t j = 0; j <= ks[q]; ++j, ++m) {
        RowOut r;
        r.logits = c.rows(e.rows_logits_device(), m, 1);
        r.id = e.rows_ids()[m];
        r.dst = c.dev_hash(j == 0 ? v.state0 : v.spec + size_t(j - 1) * c.state_bytes(), c.state_bytes());
        r.kv = c.kv_hash(q, pos[q] - 1 + j, 2);
        o.push_back(r);
      }
    }
    return o;
  };
  // KV rows the step must not write: [pos - 9, pos - 1) and [pos + 8, pos + 16) (it writes the main rows pos .. pos + k
  // and the head's pos - 1 .. pos + k - 1).
  auto window = [&](uint32_t q) { return c.kv_hash(q, pos[q] - 9, 8) ^ (c.kv_hash(q, pos[q] + 8, 8) * 31); };
  for (uint32_t q = 0; q < 3; ++q) restore(q);
  std::vector<uint64_t> win0;
  for (uint32_t q = 0; q < 3; ++q) win0.push_back(window(q));
  e.verify_rows({{0, 3}, {1, 1}, {2, 1}});
  const std::vector<RowOut> batch = outputs({0, 1, 2}, 0);
  bool b3 = c.slot_hash(3, n3) == slot3;
  for (uint32_t q = 0; q < 3; ++q) b3 = b3 && window(q) == win0[q];
  // Commit after the batched step: each slot's state after it (as the run alone leaves it, below).
  std::vector<uint64_t> after_batch;
  for (uint32_t q = 0; q < 3; ++q) {
    e.select(q);
    e.commit(1, e.rows_ids()[q == 0 ? 1 : q == 1 ? 5 : 7]);
    after_batch.push_back(c.slot_hash(q, pos[q] + 8));
  }
  // Each run alone.
  bool b1 = true;
  std::string why;
  uint32_t m0 = 0;
  for (uint32_t q = 0; q < 3; ++q) {
    for (uint32_t r = 0; r < 3; ++r) restore(r);
    e.verify_rows({{q, ks[q]}});
    const std::vector<RowOut> alone = outputs({q}, 0);
    for (uint32_t j = 0; j <= ks[q]; ++j) {
      const RowOut &x = batch[m0 + j], &y = alone[j];
      const std::string w = " slot" + std::to_string(q) + ".row" + std::to_string(j);
      if (std::memcmp(x.logits.data(), y.logits.data(), V * 4) != 0) why += w + ":logits";
      if (x.id != y.id) why += w + ":id";
      if (x.dst != y.dst) why += w + ":gdn-state";
      if (x.kv != y.kv) why += w + ":kv";
    }
    e.select(q);
    e.commit(1, e.rows_ids()[1]);
    if (c.slot_hash(q, pos[q] + 8) != after_batch[q]) why += " slot" + std::to_string(q) + ":commit";
    m0 += ks[q] + 1;
  }
  b1 = why.empty();
  std::printf("S3 B1 (bitwise, the row kernels) 1x4+2x2 at %u / %u / %u: %s%s\n", pos[0], pos[1], pos[2],
              b1 ? "PASS (logits, ids, GDN states, KV rows and the commit equal each run alone)" : "FAIL", why.c_str());
  std::printf("S3 B3 (isolation): %s\n", b3 ? "PASS (slot 3 and every slot's KV window byte-identical)" : "FAIL");
  return b1 && b3;
}

// S4 / S5: four requests alone and together.
struct Streams {
  std::vector<std::vector<uint32_t>> ids;
};
bool s4(Ctx& c, const std::vector<std::vector<uint32_t>>& prompts, const std::string& tag) {
  Engine& e = c.e;
  const uint32_t S = e.slots();
  const golden::ForcedJudge fj;
  // Alone: each slot's own lists, 32 plain steps then 8 greedy MTP iterations at K = 1; the plain steps' rows for the
  // tie rule.
  std::vector<std::vector<uint32_t>> plain(S);
  std::vector<std::vector<greedy_tie::RefStep>> steps(S);
  std::vector<std::vector<uint32_t>> lone(S);
  for (uint32_t q = 0; q < S; ++q) {
    e.select(q);
    e.reset();
    e.prefill(prompts[q]);
    plain[q].push_back(e.control()->cur_token[0]);
    for (int i = 0; i < 48; ++i) {
      e.verify(0);
      const std::vector<float> row = c.rows(e.verify_logits_device(), 0, 1);
      steps[q].push_back(greedy_tie::ref_step(row.data(), c.VU));
      e.commit(0, e.verify_ids()[0]);
      plain[q].push_back(e.control()->cur_token[0]);
    }
  }
  auto together = [&]() {
    std::vector<std::vector<uint32_t>> out(S);
    for (uint32_t q = 0; q < S; ++q) {
      e.select(q);
      e.reset();
      e.prefill(prompts[q]);
    }
    for (int i = 0; i < 32; ++i) {
      std::vector<Engine::Run> runs;
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        out[q].push_back(e.control()->cur_token[0]);
        runs.push_back({q, 0});
      }
      e.verify_rows(runs);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        e.commit(0, e.rows_ids()[q]);
      }
    }
    for (int i = 0; i < 8; ++i) {   // greedy MTP, K = 1 each: 8 rows
      std::vector<Engine::Run> runs;
      std::vector<std::vector<uint32_t>> d(S);
      for (uint32_t q = 0; q < S; ++q) {
        e.select(q);
        out[q].push_back(e.control()->cur_token[0]);
        e.draft(1);
        d[q] = e.draft_ids();
        runs.push_back({q, 1});
      }
      e.verify_rows(runs);
      for (uint32_t q = 0; q < S; ++q) {
        const uint32_t* ids = e.rows_ids() + 2 * q;
        const uint32_t j = d[q][0] == ids[0] ? 1 : 0;
        if (j) out[q].push_back(d[q][0]);
        e.select(q);
        e.commit(j, ids[j]);
      }
    }
    for (auto& o : out) o.resize(std::min<size_t>(o.size(), 48));
    return out;
  };
  const std::vector<std::vector<uint32_t>> a = together(), b = together();
  bool ok = a == b;
  std::printf("S5 determinism: the batched run twice from the same state: %s\n", ok ? "PASS (identical ids)" : "FAIL");
  for (uint32_t q = 0; q < S; ++q) {
    // Both streams start at the prefill's pending id (the context's last id); steps[i] decided plain[i + 1].
    const size_t n = a[q].size();
    ok &= n >= 2 && a[q][0] == plain[q][0];
    std::vector<uint32_t> ctx = prompts[q];
    ctx.push_back(plain[q][0]);
    const std::vector<uint32_t> pl(plain[q].begin() + 1, plain[q].begin() + n), sp(a[q].begin() + 1, a[q].end());
    const std::vector<greedy_tie::RefStep> ps(steps[q].begin(), steps[q].begin() + (n - 1));
    const spec_judge::StreamVerdict v =
        spec_judge::streams(fj, tag + "_" + std::to_string(q), ctx, pl, ps, sp, uint32_t(V), c.VU);
    std::printf("S4 B1 slot %u (%zu ids, batched vs alone): %s\n", q, n, v.line().c_str());
    ok &= v.pass;
  }
  return ok;
}
}  // namespace

int main(int argc, char** argv) {
  const std::string checkpoint = argc > 1 ? argv[1] : "urakozz/Qwen3.8-27B-W4A16-g64-AutoRound-GPTQ";
  uint32_t total = 0;
  for (uint32_t l : kSlots) total += l;
  l0::Context ctx(0);
  Engine e(ctx, loader::load(ctx, checkpoint, total, /*mtp=*/true, loader::LmHeadForm::Int8), total, false,
           runtime::default_kv_cache(), nullptr, kSlots);
  e.prepare_prefill();
  std::printf("slots_test: %u slots (%s), %s KV; %s\n", e.slots(), runtime::slot_layout_name(kSlots).c_str(),
              runtime::kv_cache_name(e.kv_cache()), e.memory_line().c_str());
  CHECK_EQ(e.slots(), 4u);
  CHECK_EQ(e.slot_base(2), 16384u + 8192u);
  CHECK_EQ(e.max_len(), 16384u);
  const auto long32k = golden::read_ids("tests/golden/prompts/long32k.ids");
  const auto code = golden::read_ids("tests/golden/prompts/code.ids");
  const auto prose = golden::read_ids("tests/golden/prompts/prose.ids");
  const auto lng = golden::read_ids("tests/golden/prompts/long.ids");
  CHECK(long32k.size() > 9000 && !code.empty() && !prose.empty() && !lng.empty());
  const std::vector<uint32_t> p_deep = cut(long32k, 0, 6000), p1 = cut(code, 0, 1200), p2 = cut(prose, 0, 900),
                              p3 = cut(lng, 0, 1000);
  Ctx c(ctx, e);
  bool ok = true;
  ok &= s1(c, p1);
  ok &= s2(c, p1, p2, p3);
  ok &= s3(c, p_deep, p1, p2, p3);
  const std::string tag = std::string("slots_") + runtime::kv_cache_name(e.kv_cache());
  ok &= s4(c, {p_deep, p1, p2, p3}, tag);
  std::printf("slots_test: %s\n", ok ? "PASS" : "FAIL");
  return spec_judge::exit_code(ok, "slots_test");
}
