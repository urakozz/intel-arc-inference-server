// Spec 21e Task 2 (host): Qwen3.8-Flash-Next's prefix-cache snapshot layouts (runtime/qwen4exp/qwen4exp_sizes.h
// state_runs / kv_runs) over SIMULATED devices - one host buffer per device allocation, sized as the engine's
// (snap_tensor_bytes = persistent_sizes' groups and the head's), every byte a function of (the absolute layer, the
// offset inside that layer's slice) only, so the same model state looks the same whichever device holds a layer:
//
//   1. the bytes: the runs' total is state_snapshot_bytes / kv_snapshot_bytes at pos 0, 1, 3, 4, 2049..2052, 5000
//      (and every run inside its allocation), with and without the MTP head; the real model's 115,651,592 B of
//      state and 25,344 B a position (derived, spec 21 §3), + 21,248 / + 2,112 with the head;
//   2. one host layout for --pp 1 and --pp 2 (plan 21e Review Focus 5): the host image saved under
//      Q4Placement::one is byte for byte the one saved under two(d, 1), two(d, 2) and two(d, 3) at --layers 4, and
//      under one / two(d, 24) at 48 layers - the state and the kv of [0, 2051), [2048, 4096), [4096, 5000);
//   3. round trips: load (into zeroed devices) then save gives the same host bytes, under each placement and across
//      them (saved under two, loaded under one, and the reverse);
//   4. positions before 0 (pos 0, 1, 3): the PLE ids are EOS 248044 on the host (a cold run's history - what
//      q4_ple_gather reads for a missing predecessor), the conv rows zeros (`zero` runs), and a load writes those
//      bytes back into their slots;
//   5. the open block's raw keys: 3 rows at 2051 (positions 2048, 2049, 2050 at tail slots 0, 1, 2), 1 at 2049, 0
//      at 2052 and 5000 (3 pad rows); the head's, one position behind: 3 rows at 2052 (2048..2050), 0 at 2049;
//   6. the GDN conv rows are p - 3 .. p - 1 at slots p % 16 (wrapping at 16 = 2048 + 16k), the PLE's p - 9 .. p - 1.
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

#include "check.h"
#include "model/qwen4exp.h"
#include "runtime/qwen4exp/qwen4exp_sizes.h"

namespace {

namespace rq = runtime::qwen4exp;
using model::Q4Placement;
using model::Qwen4ExpDesc;
using rq::SnapRun;
using rq::SnapTensor;

constexpr uint32_t kEos = 248044;
constexpr uint32_t kHeadLayer = 1000;   // the MTP head's pseudo layer id in the fill

// The allocation a tensor lives in (PleIds / PleRing share the PLE layer's `ple`).
int alloc_of(SnapTensor t) { return t == SnapTensor::PleRing ? int(SnapTensor::PleIds) : int(t); }

struct SimDevices {
  const Qwen4ExpDesc& d;
  Q4Placement p;
  uint32_t max_len;
  bool mtp;
  std::vector<std::map<int, std::vector<uint8_t>>> mem;   // [device][allocation]

  SimDevices(const Qwen4ExpDesc& desc, const Q4Placement& pl, uint32_t len, bool with_mtp)
      : d(desc), p(pl), max_len(len), mtp(with_mtp), mem(pl.devices) {
    for (uint32_t dev = 0; dev < p.devices; ++dev)
      for (SnapTensor t : {SnapTensor::Kv, SnapTensor::IdxKeys, SnapTensor::IdxTail, SnapTensor::GdnState,
                           SnapTensor::ConvRing, SnapTensor::PleIds, SnapTensor::MtpKv, SnapTensor::MtpIdxKeys,
                           SnapTensor::MtpIdxTail, SnapTensor::MtpHidden}) {
        const bool head = t == SnapTensor::MtpKv || t == SnapTensor::MtpIdxKeys || t == SnapTensor::MtpIdxTail ||
                          t == SnapTensor::MtpHidden;
        if (head && (!mtp || dev + 1 != p.devices)) continue;
        mem[dev][alloc_of(t)].assign(rq::snap_tensor_bytes(d, p, dev, max_len, t), 0);
      }
  }

  // The per-layer slice of an allocation and the absolute layers it holds, in slice order.
  void layout(uint32_t dev, int a, size_t& slice, std::vector<uint32_t>& layers) const {
    layers.clear();
    const SnapTensor t = SnapTensor(a);
    const size_t key = size_t(d.idx_dim) * 2;
    for (uint32_t l = p.first(dev); l < p.end(dev); ++l) {
      const bool qsa = d.is_qsa(l);
      if ((t == SnapTensor::Kv || t == SnapTensor::IdxKeys || t == SnapTensor::IdxTail) && qsa) layers.push_back(l);
      if ((t == SnapTensor::GdnState || t == SnapTensor::ConvRing) && !qsa) layers.push_back(l);
      if (t == SnapTensor::PleIds && l == d.ple_layer) layers.push_back(l);
    }
    switch (t) {
      case SnapTensor::Kv: slice = size_t(max_len) * 2 * d.kv_n() * 2; break;
      case SnapTensor::IdxKeys: slice = size_t(max_len / d.idx_compress) * key; break;
      case SnapTensor::IdxTail: slice = size_t(rq::kIdxTail) * key; break;
      case SnapTensor::GdnState: slice = rq::gdn_state_bytes_per_layer(d); break;
      case SnapTensor::ConvRing: slice = rq::conv_ring_bytes_per_layer(d); break;
      case SnapTensor::PleIds: slice = rq::ple_state_bytes(d); break;
      default:   // the head's allocations: one slice
        slice = mem[dev].at(a).size();
        layers = {kHeadLayer};
    }
  }

  // Every byte a function of (absolute layer, offset in the layer's slice, allocation kind).
  void fill() {
    for (uint32_t dev = 0; dev < p.devices; ++dev)
      for (auto& [a, buf] : mem[dev]) {
        size_t slice = 0;
        std::vector<uint32_t> layers;
        layout(dev, a, slice, layers);
        for (size_t i = 0; i < layers.size(); ++i)
          for (size_t o = 0; o < slice && i * slice + o < buf.size(); ++o) {
            uint64_t h = (uint64_t(layers[i]) * 0x9E3779B97F4A7C15ull) ^ (uint64_t(o) * 0xC2B2AE3D27D4EB4Full) ^
                         (uint64_t(a + 1) << 56);
            h ^= h >> 29;
            h *= 0xBF58476D1CE4E5B9ull;
            buf[i * slice + o] = uint8_t(h >> 32);
          }
      }
  }

  std::vector<uint8_t>& at(const SnapRun& r) {
    std::vector<uint8_t>& b = mem.at(r.device).at(alloc_of(r.tensor));
    CHECK(r.pad || r.offset + r.bytes <= b.size());
    return b;
  }

  // save_state / save_kv's copies, the zero runs' host filler.
  std::vector<uint8_t> save(const std::vector<SnapRun>& runs) {
    std::vector<uint8_t> host;
    for (const SnapRun& r : runs) {
      const size_t at0 = host.size();
      host.resize(at0 + r.bytes, 0);
      if (r.pad) continue;
      if (r.zero) {
        if (r.tensor == SnapTensor::PleIds) std::memcpy(host.data() + at0, &kEos, 4);
        continue;
      }
      const std::vector<uint8_t>& b = at(r);
      std::memcpy(host.data() + at0, b.data() + r.offset, r.bytes);
    }
    return host;
  }
  void load(const std::vector<SnapRun>& runs, const std::vector<uint8_t>& host) {
    size_t h = 0;
    for (const SnapRun& r : runs) {
      if (!r.pad) std::memcpy(at(r).data() + r.offset, host.data() + h, r.bytes);
      h += r.bytes;
    }
    CHECK_EQ(h, host.size());
  }
};

size_t total(const std::vector<SnapRun>& runs) {
  size_t n = 0;
  for (const SnapRun& r : runs) n += r.bytes;
  return n;
}

std::vector<SnapRun> of(const std::vector<SnapRun>& runs, SnapTensor t) {
  std::vector<SnapRun> out;
  for (const SnapRun& r : runs)
    if (r.tensor == t) out.push_back(r);
  return out;
}

const uint32_t kPositions[] = {0, 1, 3, 4, 2049, 2050, 2051, 2052, 4096, 5000};
const uint32_t kRanges[][2] = {{0, 2051}, {0, 2048}, {2048, 4096}, {4096, 5000}, {2048, 2052}, {4096, 4096}};

// 1-3: the bytes, one layout for every placement, round trips.
void layouts(const Qwen4ExpDesc& d, const std::vector<Q4Placement>& pls, uint32_t max_len, bool mtp) {
  for (uint32_t pos : kPositions) {
    std::vector<uint8_t> first;
    for (const Q4Placement& p : pls) {
      const std::vector<SnapRun> runs = rq::state_runs(d, p, pos, mtp);
      CHECK_EQ(total(runs), rq::state_snapshot_bytes(d, mtp));
      SimDevices sim(d, p, max_len, mtp);
      sim.fill();
      const std::vector<uint8_t> host = sim.save(runs);
      if (first.empty()) first = host;
      CHECK(host == first);   // --pp 2's host layout is --pp 1's byte for byte
      SimDevices back(d, p, max_len, mtp);
      back.load(runs, host);
      CHECK(back.save(runs) == host);   // a round trip
      // ... and across placements: saved under the first, loaded under this one
      SimDevices cross(d, p, max_len, mtp);
      cross.load(runs, first);
      CHECK(cross.save(runs) == first);
    }
  }
  for (const auto& rg : kRanges) {
    std::vector<uint8_t> first;
    for (const Q4Placement& p : pls) {
      const std::vector<SnapRun> runs = rq::kv_runs(d, p, max_len, rg[0], rg[1], mtp);
      CHECK_EQ(total(runs), rq::kv_snapshot_bytes(d, rg[0], rg[1], mtp));
      SimDevices sim(d, p, max_len, mtp);
      sim.fill();
      const std::vector<uint8_t> host = sim.save(runs);
      if (first.empty()) first = host;
      CHECK(host == first);
      SimDevices back(d, p, max_len, mtp);
      back.load(runs, host);
      CHECK(back.save(runs) == host);
    }
  }
  std::printf("layouts: %u layers, %zu placements, mtp %d: state %zu B, kv %zu B a position - one host layout, round "
              "trips bitwise\n", d.layers, pls.size(), mtp, rq::state_snapshot_bytes(d, mtp),
              rq::kv_snapshot_bytes(d, 0, 4, mtp) / 4);
}

// 4-6 at --layers 4 on one card: the history before 0, the tails, the conv rows.
void history(const Qwen4ExpDesc& d, uint32_t max_len) {
  const Q4Placement one = Q4Placement::one(d);
  const uint32_t gdn = d.gdn_before(d.layers), qsa = d.qsa_before(d.layers);
  const size_t key = size_t(d.idx_dim) * 2, conv = size_t(d.conv_rows()) * 2, ple = size_t(d.hc_n()) * 2;
  {   // pos 0: every predecessor is before 0
    const std::vector<SnapRun> runs = rq::state_runs(d, one, 0, true);
    for (const SnapRun& r : of(runs, SnapTensor::ConvRing)) CHECK(r.zero && r.bytes == conv);
    CHECK_EQ(of(runs, SnapTensor::ConvRing).size(), size_t(gdn) * 3);
    for (const SnapRun& r : of(runs, SnapTensor::PleRing)) CHECK(r.zero && r.bytes == ple);
    CHECK_EQ(of(runs, SnapTensor::PleRing).size(), size_t(d.ple_ring()));
    const std::vector<SnapRun> ids = of(runs, SnapTensor::PleIds);
    CHECK_EQ(ids.size(), size_t(2));
    CHECK(ids[0].zero && ids[1].zero && ids[0].offset == 15 * 4 && ids[1].offset == 14 * 4);
    for (const SnapRun& r : of(runs, SnapTensor::IdxTail)) CHECK(r.pad && r.bytes == 3 * key);
    SimDevices sim(d, one, max_len, true);
    sim.fill();
    const std::vector<uint8_t> host = sim.save(runs);
    // the PLE ids sit after the GDN states and their conv rows: EOS, EOS
    const size_t at = size_t(gdn) * (rq::gdn_state_bytes_per_layer(d) + 3 * conv);
    uint32_t w[2];
    std::memcpy(w, host.data() + at, 8);
    CHECK_EQ(w[0], kEos);
    CHECK_EQ(w[1], kEos);
    for (size_t i = size_t(gdn) * rq::gdn_state_bytes_per_layer(d); i < at; ++i) CHECK_EQ(host[i], uint8_t(0));
    for (size_t i = at + 8; i < at + 8 + 9 * ple; ++i) CHECK_EQ(host[i], uint8_t(0));
    // ... and a load writes them back into their slots (EOS in id slots 15 and 14, zeros in the conv slots)
    SimDevices back(d, one, max_len, true);
    back.fill();
    back.load(runs, host);
    uint32_t s15 = 0, s14 = 0;
    std::memcpy(&s15, back.mem[0].at(int(SnapTensor::PleIds)).data() + 15 * 4, 4);
    std::memcpy(&s14, back.mem[0].at(int(SnapTensor::PleIds)).data() + 14 * 4, 4);
    CHECK_EQ(s15, kEos);
    CHECK_EQ(s14, kEos);
    std::printf("pos 0: PLE ids EOS, EOS on the host and loaded back; %u GDN x 3 + 9 PLE conv rows zero\n", gdn);
  }
  {   // pos 3: GDN rows 0..2 real, the PLE ids 2, 1, its conv rows -6..-1 zero then 0..2
    const std::vector<SnapRun> runs = rq::state_runs(d, one, 3, false);
    for (const SnapRun& r : of(runs, SnapTensor::ConvRing)) CHECK(!r.zero && r.bytes == 3 * conv && r.offset % rq::conv_ring_bytes_per_layer(d) == 0);
    const std::vector<SnapRun> ids = of(runs, SnapTensor::PleIds);
    CHECK(!ids[0].zero && ids[0].offset == 2 * 4 && !ids[1].zero && ids[1].offset == 1 * 4);
    const std::vector<SnapRun> pr = of(runs, SnapTensor::PleRing);
    CHECK_EQ(pr.size(), size_t(7));   // six zero rows (slots 10..15), then slots 0..2 in one run
    for (size_t i = 0; i < 6; ++i) CHECK(pr[i].zero && pr[i].offset == rq::kPleConvOff + (10 + i) * ple);
    CHECK(!pr[6].zero && pr[6].offset == rq::kPleConvOff && pr[6].bytes == 3 * ple);
    const std::vector<SnapRun> t = of(runs, SnapTensor::IdxTail);
    CHECK_EQ(t.size(), size_t(qsa));   // positions 0..2 at slots 0..2, one run, no pad
    for (const SnapRun& r : t) CHECK(!r.pad && r.bytes == 3 * key);
  }
  struct Tail {
    uint32_t pos, rows, slot, head_rows;
  };
  for (const Tail& c : {Tail{2049, 1, 0, 0}, Tail{2050, 2, 0, 1}, Tail{2051, 3, 0, 2}, Tail{2052, 0, 0, 3},
                        Tail{4, 0, 0, 3}, Tail{5000, 0, 0, 3}, Tail{4099, 3, 0, 2}, Tail{4103, 3, 4, 2}}) {
    const std::vector<SnapRun> runs = rq::state_runs(d, one, c.pos, true);
    size_t real = 0, pad = 0;
    for (const SnapRun& r : of(runs, SnapTensor::IdxTail)) {
      (r.pad ? pad : real) += r.bytes;
      if (!r.pad) CHECK_EQ(r.offset % (rq::kIdxTail * key), size_t(c.slot) * key);
    }
    CHECK_EQ(real, size_t(qsa) * c.rows * key);
    CHECK_EQ(pad, size_t(qsa) * (3 - c.rows) * key);
    size_t hreal = 0;
    for (const SnapRun& r : of(runs, SnapTensor::MtpIdxTail)) hreal += r.pad ? 0 : r.bytes;
    CHECK_EQ(hreal, size_t(c.head_rows) * key);
    std::printf("pos %u: %u raw key rows a QSA layer (from slot %u), the head's %u\n", c.pos, c.rows, c.slot, c.head_rows);
  }
  {   // the GDN window wraps: pos 2050 reads positions 2047..2049, slots 15, 0, 1 - two runs
    const std::vector<SnapRun> runs = rq::state_runs(d, one, 2050, false);
    const std::vector<SnapRun> cr = of(runs, SnapTensor::ConvRing);
    CHECK_EQ(cr.size(), size_t(gdn) * 2);
    CHECK(cr[0].offset == 15 * conv && cr[0].bytes == conv && cr[1].offset == 0 && cr[1].bytes == 2 * conv);
    // the PLE's p - 9 .. p - 1 = 2041..2049: slots 9..15 then 0..1
    const std::vector<SnapRun> pr = of(runs, SnapTensor::PleRing);
    CHECK_EQ(pr.size(), size_t(2));
    CHECK(pr[0].offset == rq::kPleConvOff + 9 * ple && pr[0].bytes == 7 * ple && pr[1].offset == rq::kPleConvOff &&
          pr[1].bytes == 2 * ple);
  }
  // The kv runs: [2048, 2052) holds blocks 512 only; [4096, 5000) blocks 1024..1249; a begin off the block grid throws.
  CHECK_EQ(rq::kv_snapshot_bytes(d, 2048, 2052, false), size_t(qsa) * (4 * 2048 + 256));
  CHECK_EQ(rq::kv_snapshot_bytes(d, 2048, 2051, false), size_t(qsa) * 3 * 2048);
  CHECK_EQ(rq::kv_snapshot_bytes(d, 4096, 5000, true), size_t(qsa + 1) * (904 * 2048 + 226 * 256));
  bool threw = false;
  try {
    rq::kv_runs(d, one, max_len, 2, 8, false);
  } catch (const std::invalid_argument&) {
    threw = true;
  }
  CHECK(threw);
}

}  // namespace

int main() {
  const Qwen4ExpDesc& full = model::qwen4exp();
  // The real model (derived): 36 x 3,145,728 + 36 x 61,440 + 8 + 184,320 + 12 x 768 = 115,651,592 B.
  CHECK_EQ(rq::state_snapshot_bytes(full, false), size_t(115651592));
  CHECK_EQ(rq::state_snapshot_bytes(full, true), size_t(115651592 + 768 + 20480));
  CHECK_EQ(rq::kv_snapshot_bytes(full, 0, 4096, false), size_t(4096) * 25344);
  CHECK_EQ(rq::kv_snapshot_bytes(full, 0, 4096, false) / 4096, rq::kv_bytes_per_pos(full));
  CHECK_EQ(rq::kv_snapshot_bytes(full, 0, 4096, true), size_t(4096) * (25344 + 2112));
  std::printf("the real model: state %zu B (%.1f MB; + %zu with the head), blocks %zu B a position (+ %zu)\n",
              rq::state_snapshot_bytes(full, false), rq::state_snapshot_bytes(full, false) / 1e6,
              rq::state_snapshot_bytes(full, true) - rq::state_snapshot_bytes(full, false),
              rq::kv_snapshot_bytes(full, 0, 4, false) / 4,
              (rq::kv_snapshot_bytes(full, 0, 4, true) - rq::kv_snapshot_bytes(full, 0, 4, false)) / 4);

  const uint32_t len = 5120;
  const Qwen4ExpDesc d4 = rq::truncated(full, 4);
  for (bool mtp : {false, true}) {
    layouts(d4, {Q4Placement::one(d4), Q4Placement::two(d4, 1), Q4Placement::two(d4, 2), Q4Placement::two(d4, 3)}, len,
            mtp);
    const Qwen4ExpDesc d8 = rq::truncated(full, 8);
    layouts(d8, {Q4Placement::one(d8), Q4Placement::two(d8, 4), Q4Placement::two(d8, 7)}, len, mtp);
  }
  history(d4, len);
  // 48 layers: sizes only for the placements (the simulated devices would be ~1.4 GB at max_len 8192).
  for (const Q4Placement& p : {Q4Placement::one(full), Q4Placement::two(full, 24), Q4Placement::two(full, 1)})
    for (bool mtp : {false, true}) {
      CHECK_EQ(total(rq::state_runs(full, p, 5000, mtp)), rq::state_snapshot_bytes(full, mtp));
      CHECK_EQ(total(rq::kv_runs(full, p, 262144, 4096, 5000, mtp)), rq::kv_snapshot_bytes(full, 4096, 5000, mtp));
    }
  std::puts("qwen4exp_snapshot_test OK");
  return 0;
}
